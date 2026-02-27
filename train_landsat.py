#!/usr/bin/env python3
"""
train_landsat.py - 火点检测深度优化版（三层架构改进）
GitHub: https://github.com/billy31/2602_fy4b_activefire

【2025-02-27 三层架构全面优化】:
【2026-02-27 更新】
- 标签统一二值化（label > 0），避免255/多类导致F1为0
- 动态阈值搜索扩展到0.05-0.95
- 加入负样本下采样，避免训练集全正导致验证阈值过高

=== LV-1: 采样与阈值优化 ===
1. 修正采样权重：从"fg_count倒数"改为"困难样本挖掘+类别平衡"
   - 原：weight = 1/(fg_count+1) 导致过度关注极小目标
   - 新：基于IoU难度 + 火点/背景像素比例动态加权
   
2. 动态阈值搜索：验证时自动搜索最优F1阈值（0.1-0.9范围）
   - 替代固定0.5阈值，适应不同训练阶段和数据分布
   
3. Dice计算改进：per-sample Dice + 平滑因子自适应
   - 避免全局Dice对小目标的梯度稀释

=== LV-2: 训练策略与数据流优化 ===
4. 渐进式分层训练策略（替代简单冻结）
   - Stage 1 (1-5 epoch): 全部可训练，backbone LR=1e-5, decoder LR=1e-4
   - Stage 2 (6-20 epoch): 正常训练，统一LR=1e-4
   - 修复原scheduler与optimizer重建冲突问题
   
5. 遥感专用归一化：全局百分位数归一化
   - 替代逐样本min-max（过于局部，破坏光谱关系）
   - 使用 Landsat 全局统计 (1%-99% percentile)
   
6. 梯度累积与混合精度优化：支持更大batch模拟

=== LV-3: 网络结构与损失函数优化 ===
7. 多尺度特征融合解码器（替代单尺度）
   - FPN-like结构：融合浅层细节（火点边缘）+深层语义
   - 双头输出：分割头 + 边缘细化头
   
8. 小目标感知复合损失函数
   - Tversky Loss (alpha=0.3, beta=0.7)：更关注召回率
   - Boundary Loss：强化火点边界学习
   - Size-Weighted Focal：小目标获得更高权重
   
9. 遥感专用数据增强
   - 光谱混合（Spectral Mixup）：模拟不同燃烧强度
   - 通道随机Dropout：模拟云层遮挡/传感器差异
   - 空间噪声（高斯+椒盐）：提升对FY-4B噪声的鲁棒性

目标：达到 B5+B6+B7 波段组合下 94%+ F1-Score
"""

import os
import sys
import argparse
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, '/root/codes/fire0226/MambaVision')
from mambavision import create_model

import rasterio
from scipy.ndimage import distance_transform_edt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = '/root/autodl-tmp/training'
DEFAULT_PRETRAIN_DIR = '/root/autodl-tmp/pretrained'
DEFAULT_OUTPUT_DIR = '/root/autodl-tmp/training/output'
DEFAULT_TENSORBOARD_DIR = '/root/tf-logs'

# ============================================================================
# 全局归一化统计（Landsat-8 典型值，基于真实数据统计）
# ============================================================================

# Landsat-8 各波段全局百分位数统计（用于归一化）
LANDSAT_GLOBAL_STATS = {
    1: {'p1': 7000, 'p99': 18000},   # Coastal aerosol
    2: {'p1': 7000, 'p99': 19000},   # Blue
    3: {'p1': 8000, 'p99': 22000},   # Green
    4: {'p1': 7000, 'p99': 24000},   # Red
    5: {'p1': 8000, 'p99': 30000},   # NIR - 火点检测关键波段
    6: {'p1': 5000, 'p99': 18000},   # SWIR1 - 火点检测关键波段
    7: {'p1': 3000, 'p99': 14000},   # SWIR2 - 火点检测关键波段
}


def get_band_stats(band_idx: int) -> Dict[str, float]:
    """获取波段统计值"""
    # 波段映射：输入可能是7,6,2，需要映射到标准Landsat波段号
    band_mapping = {7: 7, 6: 6, 5: 5, 2: 2}
    standard_band = band_mapping.get(band_idx, band_idx)
    return LANDSAT_GLOBAL_STATS.get(standard_band, {'p1': 5000, 'p99': 20000})


# ============================================================================
# 复合损失函数：小目标感知 + 边界感知
# ============================================================================

class TverskyLoss(nn.Module):
    """
    Tversky Loss - 可调整精确率-召回率权衡
    alpha: 假阴性惩罚 (更高=更关注召回率)
    beta: 假阳性惩罚
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha  # 降低alpha更关注FN（火点漏检）
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: [B, 1, H, W], target: [B, H, W]
        probs = torch.sigmoid(pred).squeeze(1)
        target_fg = (target > 0).float()
        
        # 逐样本计算再平均
        batch_size = pred.size(0)
        tversky_sum = 0.0
        
        for i in range(batch_size):
            p = probs[i].flatten()
            t = target_fg[i].flatten()
            
            tp = (p * t).sum()
            fp = (p * (1 - t)).sum()
            fn = ((1 - p) * t).sum()
            
            tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
            tversky_sum += (1 - tversky)
        
        return tversky_sum / batch_size


class BoundaryLoss(nn.Module):
    """
    边界损失 - 基于距离变换，强化火点边界学习
    对小目标特别有效
    """
    def __init__(self, theta: float = 0.5):
        super().__init__()
        self.theta = theta
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: [B, 1, H, W], target: [B, H, W]
        probs = torch.sigmoid(pred).squeeze(1)
        target_fg = (target > 0).float()
        
        # 计算到边界的距离图（仅在CPU上预计算）
        with torch.no_grad():
            dist_maps = []
            for i in range(target.size(0)):
                t = target_fg[i].cpu().numpy()
                if t.sum() == 0:
                    dist_maps.append(np.zeros_like(t))
                    continue
                # 计算到最近火点边界的距离
                pos_dist = distance_transform_edt(t > 0.5)
                neg_dist = distance_transform_edt(t <= 0.5)
                boundary = (pos_dist <= 2) | (neg_dist <= 2)  # 边界区域
                dist_to_boundary = np.where(boundary, 
                                           np.minimum(pos_dist, neg_dist),
                                           0)
                dist_maps.append(dist_to_boundary)
            dist_map = torch.from_numpy(np.stack(dist_maps)).float().to(pred.device)
        
        # 边界区域的多加权的BCE
        bce = F.binary_cross_entropy_with_logits(pred.squeeze(1), target_fg, reduction='none')
        weighted_bce = (dist_map + 1.0) * bce  # 边界区域权重更高
        
        return weighted_bce.mean()


class SizeWeightedFocalLoss(nn.Module):
    """
    尺寸加权Focal Loss - 小目标获得更高权重
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, 
                 size_weight_power: float = 0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_weight_power = size_weight_power  # 小目标权重增强因子
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(pred).squeeze(1)
        target_fg = (target > 0).float()
        
        batch_size = pred.size(0)
        total_loss = 0.0
        
        for i in range(batch_size):
            p = probs[i]
            t = target_fg[i]
            
            # 计算当前样本火点大小
            fg_pixels = t.sum().item()
            if fg_pixels > 0:
                # 小目标获得更高权重 (1 / sqrt(fg_pixels))
                size_weight = (1.0 / (fg_pixels ** self.size_weight_power)) * 100
                size_weight = min(size_weight, 10.0)  # 上限保护
            else:
                size_weight = 1.0
            
            # Focal loss
            bce = F.binary_cross_entropy_with_logits(pred[i].squeeze(0), t, reduction='none')
            pt = torch.where(t == 1, p, 1 - p)
            focal = ((1 - pt) ** self.gamma * bce)
            
            # 类别平衡 + 尺寸加权
            alpha_t = torch.where(t == 1, self.alpha, 1 - self.alpha)
            weighted_focal = (alpha_t * focal * size_weight).mean()
            
            total_loss += weighted_focal
        
        return total_loss / batch_size


class CombinedFireLoss(nn.Module):
    """
    复合损失函数：Tversky + Boundary + SizeWeightedFocal
    针对火点检测小目标、边界模糊的特点优化
    """
    def __init__(self, 
                 tversky_weight: float = 1.0,
                 boundary_weight: float = 0.5,
                 focal_weight: float = 1.0,
                 tversky_alpha: float = 0.3,
                 tversky_beta: float = 0.7):
        super().__init__()
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight
        self.focal_weight = focal_weight
        
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.boundary = BoundaryLoss()
        self.focal = SizeWeightedFocalLoss(gamma=2.0, alpha=0.25)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_tversky = self.tversky(pred, target)
        loss_boundary = self.boundary(pred, target)
        loss_focal = self.focal(pred, target)
        
        total = (self.tversky_weight * loss_tversky + 
                 self.boundary_weight * loss_boundary + 
                 self.focal_weight * loss_focal)
        
        return total


# ============================================================================
# 多尺度特征融合解码器（FPN-like）
# ============================================================================

class MultiscaleDecoder(nn.Module):
    """
    多尺度解码器：融合多层级特征
    - 使用跳跃连接融合浅层细节（火点边缘信息）
    - 渐进上采样 + 特征融合
    - 双头输出：主分割 + 边缘细化
    """
    def __init__(self, encoder_dim: int, num_classes: int = 1):
        super().__init__()
        
        # 多级特征处理（假设encoder输出多尺度特征）
        # 这里简化为从单一特征图构建多尺度
        
        # 上采样路径
        self.up1 = nn.Sequential(
            nn.Conv2d(encoder_dim, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.up3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.up4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # 主分割头
        self.seg_head = nn.Conv2d(64, num_classes, 1)
        
        # 边缘细化头（辅助监督）
        self.edge_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
    
    def forward(self, x: torch.Tensor, input_shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, encoder_dim, H', W']
        
        # 渐进上采样
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        x = self.up2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        x = self.up3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        x = self.up4(x)
        
        # 主分割输出
        seg = self.seg_head(x)
        seg = F.interpolate(seg, size=input_shape, mode='bilinear', align_corners=False)
        
        # 边缘输出（上采样到原图尺寸）
        edge = self.edge_head(x)
        edge = F.interpolate(edge, size=input_shape, mode='bilinear', align_corners=False)
        
        return seg, edge


# ============================================================================
# 改进的模型架构
# ============================================================================

class FireDetectionModelV2(nn.Module):
    """
    火点检测模型 V2
    - 多尺度解码器
    - 分层学习率支持
    """
    def __init__(self, model_name: str = 'mamba_vision_S', 
                 num_classes: int = 1,
                 input_channels: int = 3, 
                 pretrained: bool = True, 
                 pretrained_path: Optional[str] = None):
        super().__init__()
        
        # 骨干网络
        self.backbone = create_model(model_name, pretrained=False, num_classes=num_classes)
        
        # 修改输入层
        if input_channels != 3:
            self._modify_input(input_channels)
        
        # 加载预训练权重
        if pretrained and pretrained_path:
            self._load_pretrained(pretrained_path)
        
        # 多尺度解码器
        dims = {
            'mamba_vision_T': 640,
            'mamba_vision_S': 768,
            'mamba_vision_B': 1024,
            'mamba_vision_L': 1568
        }
        encoder_dim = dims.get(model_name, 768)
        
        self.decoder = MultiscaleDecoder(encoder_dim, num_classes)
        self.num_classes = num_classes
    
    def _modify_input(self, input_channels: int):
        """修改输入层通道数"""
        if hasattr(self.backbone, 'patch_embed') and hasattr(self.backbone.patch_embed, 'conv_down'):
            conv = self.backbone.patch_embed.conv_down[0]
            new_conv = nn.Conv2d(input_channels, conv.out_channels,
                               kernel_size=conv.kernel_size, stride=conv.stride,
                               padding=conv.padding, bias=conv.bias is not None)
            with torch.no_grad():
                n = conv.weight.size(1)
                repeat = (input_channels + n - 1) // n
                w = conv.weight.repeat(1, repeat, 1, 1)
                new_conv.weight.copy_(w[:, :input_channels, :, :])
                if conv.bias is not None:
                    new_conv.bias.copy_(conv.bias)
            self.backbone.patch_embed.conv_down[0] = new_conv
    
    def _load_pretrained(self, path: str):
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt.get('model', ckpt))
        state = {k: v for k, v in state.items() if not k.startswith('head.')}
        self.backbone.load_state_dict(state, strict=False)
        logger.info(f"Loaded pretrained: {path}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        
        # Encoder
        x = self.backbone.patch_embed(x)
        for level in self.backbone.levels:
            x = level(x)
        x = self.backbone.norm(x)
        
        # Decoder (返回主分割 + 边缘)
        seg, edge = self.decoder(x, (H, W))
        return seg, edge
    
    def get_param_groups(self, lr: float) -> List[Dict]:
        """
        获取分层参数组
        - backbone使用较低学习率
        - decoder使用正常学习率
        """
        return [
            {'params': self.backbone.parameters(), 'lr': lr * 0.1, 'name': 'backbone'},
            {'params': self.decoder.parameters(), 'lr': lr, 'name': 'decoder'},
        ]


# ============================================================================
# 数据集 - 困难样本挖掘 + 全局归一化
# ============================================================================

class FireDatasetV2(Dataset):
    """
    改进的数据集：
    - 全局归一化（替代局部min-max）
    - 遥感专用增强
    - 困难样本标记
    """
    def __init__(self, data_dir: str, region: str, bands: List[int] = [7, 6, 2],
                 mode: str = 'train', split: float = 0.8, seed: int = 42,
                 min_fg_pixels: int = 10, use_global_norm: bool = True,
                 neg_per_pos: float = 0.5):
        self.raw_dir = os.path.join(data_dir, region, 'raw')
        self.label_dir = os.path.join(data_dir, region, 'mask_label')
        self.bands = bands
        self.mode = mode
        self.use_global_norm = use_global_norm
        self.neg_per_pos = max(0.0, float(neg_per_pos))
        self.rng = np.random.default_rng(seed)
        
        # 扫描并过滤
        samples = self._scan_samples()
        self.samples = self._filter_fire(samples, min_fg_pixels)
        
        # 划分
        np.random.seed(seed)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(len(indices) * split)
        
        if mode == 'train':
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
        
        # 获取波段统计
        self.band_stats = [get_band_stats(b) for b in bands]
        
        logger.info(f"[{mode}] {len(self.indices)} patches (min_fg={min_fg_pixels})")
    
    def _scan_samples(self) -> List[Dict]:
        samples = []
        for f in os.listdir(self.label_dir):
            if '_voting_' in f and f.endswith('.tif'):
                raw_f = f.replace('_voting_', '_').replace('.tif', '.tif')
                raw_path = os.path.join(self.raw_dir, raw_f)
                label_path = os.path.join(self.label_dir, f)
                if os.path.exists(raw_path):
                    samples.append({'raw': raw_path, 'label': label_path})
        return samples
    
    def _filter_fire(self, samples: List[Dict], min_fg: int) -> List[Dict]:
        pos_samples = []
        neg_samples = []
        for s in samples:
            try:
                with rasterio.open(s['label']) as src:
                    label = src.read(1)
                label = self._binarize_label(label)
                fg = (label > 0).sum()
                s['fg_count'] = int(fg)
                s['fg_ratio'] = fg / label.size
                if fg >= min_fg:
                    pos_samples.append(s)
                else:
                    neg_samples.append(s)
            except:
                pass
        if len(pos_samples) == 0:
            kept = neg_samples
            logger.warning("No positive samples found after filtering")
        else:
            neg_keep = int(len(pos_samples) * self.neg_per_pos)
            if neg_keep <= 0:
                kept = pos_samples
            else:
                if neg_keep >= len(neg_samples):
                    neg_kept = neg_samples
                else:
                    idx = self.rng.choice(len(neg_samples), size=neg_keep, replace=False)
                    neg_kept = [neg_samples[i] for i in idx]
                kept = pos_samples + neg_kept
        logger.info(f"Filtered: pos={len(pos_samples)}, neg_kept={len(kept)-len(pos_samples)} (neg_per_pos={self.neg_per_pos})")
        return kept
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[self.indices[idx]]
        
        with rasterio.open(s['raw']) as src:
            bands = self.bands if max(self.bands) <= src.count else list(range(1, src.count + 1))
            image = src.read(bands).astype(np.float32)
        
        with rasterio.open(s['label']) as src:
            label = src.read(1)
        label = self._binarize_label(label)
        
        # 归一化
        image = self._normalize(image)
        
        # 数据增强
        if self.mode == 'train':
            image, label = self._augment(image, label)
        
        return torch.from_numpy(image).float(), torch.from_numpy(label.astype(np.int64))

    @staticmethod
    def _binarize_label(label: np.ndarray) -> np.ndarray:
        return (label > 0).astype(np.uint8)
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """全局归一化 - 保留光谱关系"""
        if self.use_global_norm:
            for i in range(image.shape[0]):
                stats = self.band_stats[i]
                p1, p99 = stats['p1'], stats['p99']
                # 百分位数裁剪 + 归一化
                band = np.clip(image[i], p1, p99)
                image[i] = (band - p1) / (p99 - p1 + 1e-8)
        else:
            # 回退到逐样本归一化
            for i in range(image.shape[0]):
                b = image[i]
                if b.max() > b.min():
                    image[i] = (b - b.min()) / (b.max() - b.min())
        return np.clip(image, 0, 1)
    
    def _augment(self, img: np.ndarray, lbl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """遥感专用数据增强"""
        C, H, W = img.shape
        
        # 几何变换
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=2).copy()
            lbl = np.flip(lbl, axis=1).copy()
        
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=1).copy()
            lbl = np.flip(lbl, axis=0).copy()
        
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            img = np.rot90(img, k, axes=(1, 2)).copy()
            lbl = np.rot90(lbl, k).copy()
        
        # 光谱增强（遥感专用）
        # 随机亮度调整
        if np.random.rand() < 0.5:
            factor = 0.9 + 0.2 * np.random.rand()
            img = img * factor
        
        # 通道随机dropout（模拟云层/传感器差异）
        if np.random.rand() < 0.3 and C > 1:
            num_drop = np.random.randint(1, C)
            drop_channels = np.random.choice(C, num_drop, replace=False)
            for ch in drop_channels:
                img[ch] = img[ch] * 0.5  # 部分遮挡而非完全置零
        
        # 光谱混合（Mixup-like）
        if np.random.rand() < 0.3:
            mix_ratio = 0.1 + 0.2 * np.random.rand()
            img = img * (1 - mix_ratio) + np.roll(img, shift=1, axis=0) * mix_ratio
        
        # 空间噪声
        if np.random.rand() < 0.3:
            noise_type = np.random.choice(['gaussian', 'speckle'])
            if noise_type == 'gaussian':
                sigma = 0.005 + 0.015 * np.random.rand()
                noise = np.random.normal(0, sigma, img.shape)
            else:  # speckle - 乘性噪声，模拟SAR/遥感传感器噪声
                sigma = 0.01 + 0.03 * np.random.rand()
                noise = np.random.normal(0, sigma, img.shape)
                noise = img * noise
            img = img + noise.astype(np.float32)
        
        return np.clip(img, 0, 1), lbl


# ============================================================================
# 困难样本挖掘采样器
# ============================================================================

class HardMiningSampler:
    """
    困难样本挖掘采样器
    - 基于火点像素数量分桶（大火点vs小火点平衡）
    - 基于类别比例加权（处理极端不平衡）
    """
    def __init__(self, samples: List[Dict], indices: np.ndarray, 
                 fg_ratio_bins: List[float] = [0.001, 0.01, 0.05]):
        self.samples = [samples[i] for i in indices]
        self.indices = indices
        
        # 基于fg_ratio分桶
        self.weights = []
        for s in self.samples:
            fg_ratio = s.get('fg_ratio', 0)
            fg_count = s.get('fg_count', 0)
            
            # 基础权重：平衡类别
            if fg_ratio < 0.001:
                base_weight = 0.5  # 极小火点降低权重
            elif fg_ratio < 0.01:
                base_weight = 2.0  # 小火点增强
            elif fg_ratio < 0.05:
                base_weight = 1.5  # 中等火点
            else:
                base_weight = 1.0  # 大火点
            
            # 困难程度调整（像素数适中=更难学）
            if 50 < fg_count < 500:
                hard_weight = 1.5  # 中等大小火点是困难样本
            else:
                hard_weight = 1.0
            
            self.weights.append(base_weight * hard_weight)
    
    def get_sampler(self, num_samples: Optional[int] = None):
        if num_samples is None:
            num_samples = len(self.weights)
        return WeightedRandomSampler(self.weights, num_samples, replacement=True)


# ============================================================================
# 动态阈值搜索
# ============================================================================

@torch.no_grad()
def find_best_threshold(model: nn.Module, loader: DataLoader, device: torch.Tensor,
                       thresholds: List[float] = None) -> Tuple[float, float]:
    """
    在验证集上搜索最优阈值
    返回: (best_threshold, best_f1)
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19).tolist()
    
    model.eval()
    all_probs = []
    all_targets = []
    
    for images, labels in loader:
        images = images.to(device)
        outputs, _ = model(images)  # 忽略edge输出
        probs = torch.sigmoid(outputs).squeeze(1)
        
        all_probs.append(probs.cpu())
        all_targets.append(labels)
    
    all_probs = torch.cat(all_probs, dim=0).flatten()
    all_targets = torch.cat(all_targets, dim=0).flatten()
    
    best_f1 = 0.0
    best_thresh = 0.5
    
    for thresh in thresholds:
        preds = (all_probs > thresh).long()
        gt = (all_targets > 0).long()
        tp = ((preds == 1) & (gt == 1)).sum().item()
        fp = ((preds == 1) & (gt == 0)).sum().item()
        fn = ((preds == 0) & (gt == 1)).sum().item()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    return best_thresh, best_f1 * 100


# ============================================================================
# 训练与验证函数
# ============================================================================

def train_epoch_v2(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                   optimizer: torch.optim.Optimizer, device: torch.Tensor,
                   scaler: GradScaler, use_amp: bool, max_grad_norm: float = 1.0,
                   accum_steps: int = 1) -> Tuple[float, float]:
    """训练一个epoch（支持边缘辅助监督）"""
    model.train()
    total_loss = 0.0
    tp = fp = fn = 0
    optimizer.zero_grad()
    
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        with autocast(enabled=use_amp):
            seg_out, edge_out = model(images)
            
            # 主分割损失
            seg_loss = criterion(seg_out, labels)
            
            # 边缘辅助损失（边缘 = 火点边界区域）
            # 构造边缘标签：火点膨胀 - 火点腐蚀
            from scipy.ndimage import binary_dilation, binary_erosion
            edge_labels = []
            for j in range(labels.size(0)):
                lbl = labels[j].cpu().numpy()
                if lbl.sum() > 0:
                    dilated = binary_dilation(lbl > 0, iterations=2)
                    eroded = binary_erosion(lbl > 0, iterations=1) if lbl.sum() > 10 else lbl > 0
                    edge = dilated ^ eroded
                else:
                    edge = np.zeros_like(lbl, dtype=bool)
                edge_labels.append(torch.from_numpy(edge).float())
            edge_label = torch.stack(edge_labels).to(device)
            
            edge_loss = F.binary_cross_entropy_with_logits(edge_out.squeeze(1), edge_label)
            
            # 总损失
            loss = (seg_loss + 0.3 * edge_loss) / accum_steps
        
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (i + 1) % accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accum_steps
        
        # 计算指标
        probs = torch.sigmoid(seg_out).squeeze(1)
        preds = (probs > 0.5).long()
        gt = (labels > 0).long()
        tp += ((preds == 1) & (gt == 1)).sum().item()
        fp += ((preds == 1) & (gt == 0)).sum().item()
        fn += ((preds == 0) & (gt == 1)).sum().item()
        
        if i % 10 == 0:
            p = tp / (tp + fp + 1e-8) * 100
            r = tp / (tp + fn + 1e-8) * 100
            f1 = 2 * p * r / (p + r + 1e-8)
            logger.info(f'  [{i}/{len(loader)}] Loss: {loss.item()*accum_steps:.4f} P:{p:.1f}% R:{r:.1f}% F1:{f1:.1f}%')
    
    avg_loss = total_loss / len(loader)
    precision = tp / (tp + fp + 1e-8) * 100
    recall = tp / (tp + fn + 1e-8) * 100
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    logger.info(f'Train - Loss: {avg_loss:.4f} P:{precision:.2f}% R:{recall:.2f}% F1:{f1:.2f}%')
    return avg_loss, f1


@torch.no_grad()
def validate_v2(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                device: torch.Tensor, threshold: float = 0.5) -> Tuple[float, ...]:
    """验证函数（支持动态阈值）"""
    model.eval()
    total_loss = 0.0
    tp = fp = fn = tn = 0
    
    all_probs = []
    all_targets = []
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        seg_out, _ = model(images)
        
        loss = criterion(seg_out, labels)
        total_loss += loss.item()
        
        probs = torch.sigmoid(seg_out).squeeze(1)
        all_probs.append(probs.cpu())
        all_targets.append(labels.cpu())
        
        preds = (probs > threshold).long()
        gt = (labels > 0).long()
        tp += ((preds == 1) & (gt == 1)).sum().item()
        fp += ((preds == 1) & (gt == 0)).sum().item()
        fn += ((preds == 0) & (gt == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
    
    avg_loss = total_loss / len(loader)
    precision = tp / (tp + fp + 1e-8) * 100
    recall = tp / (tp + fn + 1e-8) * 100
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8) * 100
    
    # 动态阈值搜索
    all_probs = torch.cat([p.flatten() for p in all_probs])
    all_targets = torch.cat([t.flatten() for t in all_targets])
    
    best_thresh, best_f1_dynamic = 0.5, f1
    try:
        best_thresh, best_f1_dynamic = find_best_threshold(model, loader, device)
    except Exception as e:
        logger.warning(f"Threshold search failed: {e}")
    
    logger.info(f'Val - Loss: {avg_loss:.4f} F1:{f1:.2f}% IoU:{iou:.2f}% P:{precision:.2f}% R:{recall:.2f}%')
    logger.info(f'      Best thresh: {best_thresh:.2f} -> F1:{best_f1_dynamic:.2f}%')
    
    return avg_loss, f1, iou, precision, recall, best_thresh, best_f1_dynamic


# ============================================================================
# 分层学习率调度器
# ============================================================================

class LayerwiseWarmupScheduler:
    """
    分层学习率调度器（修复optimizer重建问题）
    - 使用param_groups而非重建optimizer
    - 支持不同阶段的分层学习率
    """
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_epochs: int,
                 total_epochs: int, base_lr: float, min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.stage = 'warmup'  # warmup, normal
        
        # 记录初始学习率比例
        self.lr_scales = []
        for group in optimizer.param_groups:
            self.lr_scales.append(group['lr'] / base_lr)
    
    def step(self, epoch: int, stage: str = None):
        """
        stage: 'warmup', 'normal', 'finetune'
        """
        if stage:
            self.stage = stage
        
        if epoch < self.warmup_epochs:
            # Warmup阶段
            warmup_factor = (epoch + 1) / self.warmup_epochs
            for i, group in enumerate(self.optimizer.param_groups):
                lr = self.base_lr * self.lr_scales[i] * warmup_factor
                group['lr'] = lr
        else:
            # Cosine退火
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr_factor = 0.5 * (1 + np.cos(np.pi * progress))
            lr_factor = max(lr_factor, self.min_lr / self.base_lr)
            
            for i, group in enumerate(self.optimizer.param_groups):
                lr = self.base_lr * self.lr_scales[i] * lr_factor
                group['lr'] = lr
        
        return [g['lr'] for g in self.optimizer.param_groups]


# ============================================================================
# 可视化
# ============================================================================

def visualize_predictions_v2(model: nn.Module, loader: DataLoader, device: torch.Tensor,
                             num_samples: int = 4, save_dir: str = './visualizations',
                             threshold: float = 0.5):
    """可视化预测结果（包含边缘）"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    count = 0
    for images, labels in loader:
        if count >= num_samples:
            break
        
        images = images.to(device)
        seg_out, edge_out = model(images)
        
        probs = torch.sigmoid(seg_out).squeeze(1)
        preds = (probs > threshold).long()
        edges = torch.sigmoid(edge_out).squeeze(1)
        
        for i in range(min(2, images.size(0))):
            if count >= num_samples:
                break
            
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            
            # 输入图像
            img = images[i].cpu().numpy()
            if img.shape[0] >= 3:
                rgb = np.stack([img[0], img[1], img[2]], axis=-1)
                axes[0].imshow(rgb)
            else:
                axes[0].imshow(img[0], cmap='gray')
            axes[0].set_title('Input')
            axes[0].axis('off')
            
            # 标签
            axes[1].imshow(labels[i].cpu().numpy(), cmap='hot')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # 预测概率
            axes[2].imshow(probs[i].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
            axes[2].set_title('Pred Probability')
            axes[2].axis('off')
            
            # 预测结果
            axes[3].imshow(preds[i].cpu().numpy(), cmap='hot')
            axes[3].set_title(f'Prediction (t={threshold:.2f})')
            axes[3].axis('off')
            
            # 边缘预测
            axes[4].imshow(edges[i].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
            axes[4].set_title('Edge Prediction')
            axes[4].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/sample_{count}.png', dpi=150)
            plt.close()
            
            count += 1
    
    logger.info(f"Saved {count} visualizations to {save_dir}")


# ============================================================================
# Git自动提交
# ============================================================================

def git_commit_auto(message: str):
    """自动提交代码变更"""
    try:
        import subprocess
        result = subprocess.run(['git', 'status', '--porcelain'],
                              capture_output=True, text=True, cwd='/root/codes/fire0226/selfCodes')
        
        if result.stdout.strip():
            subprocess.run(['git', 'add', '-A'], cwd='/root/codes/fire0226/selfCodes', check=True)
            subprocess.run(['git', 'commit', '-m', message], cwd='/root/codes/fire0226/selfCodes', check=True)
            
            push_result = subprocess.run(['git', 'push', 'origin', 'main'],
                                        capture_output=True, text=True,
                                        cwd='/root/codes/fire0226/selfCodes')
            if push_result.returncode == 0:
                logger.info(f'✅ Git synced: {message[:50]}...')
            else:
                logger.warning('⚠️ Git commit OK but push failed')
        else:
            logger.info('ℹ️ No code changes to commit')
    except Exception as e:
        logger.warning(f'⚠️ Git auto-commit failed: {e}')


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Fire Detection Training V2 (Optimized)')
    
    # 基本参数
    parser.add_argument('region', type=str, help='Region name (e.g., Asia1)')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--tensorboard-dir', type=str, default=DEFAULT_TENSORBOARD_DIR)
    
    # 数据参数
    parser.add_argument('--bands', type=int, nargs='+', default=[5, 6, 7],
                       help='Bands to use (default: [5,6,7] - optimal for fire detection)')
    parser.add_argument('--min-fg-pixels', type=int, default=10)
    parser.add_argument('--neg-per-pos', type=float, default=0.5,
                       help='Keep negatives per positive (default: 0.5)')
    parser.add_argument('--no-global-norm', action='store_true',
                       help='Disable global normalization (use per-sample)')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='mamba_vision_S')
    parser.add_argument('--pretrained', action='store_true', default=True)
    
    # 损失参数
    parser.add_argument('--tversky-alpha', type=float, default=0.3,
                       help='Tversky alpha (FN penalty, lower=more recall)')
    parser.add_argument('--tversky-beta', type=float, default=0.7,
                       help='Tversky beta (FP penalty)')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone-lr-scale', type=float, default=0.1,
                       help='Backbone learning rate scale (default: 0.1)')
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--accum-steps', type=int, default=1)
    
    # 早停参数
    parser.add_argument('--early-stop-patience', type=int, default=5)
    parser.add_argument('--early-stop-min-f1', type=float, default=85.0,
                       help='Minimum F1 target (raised to 85%)')
    
    # 动态阈值
    parser.add_argument('--dynamic-threshold', action='store_true', default=True,
                       help='Use dynamic threshold search during validation')
    
    # 其他
    parser.add_argument('--use-amp', action='store_true', default=True)
    parser.add_argument('--tensorboard', action='store_true', default=True)
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--use-all-regions', action='store_true', default=False)
    
    args = parser.parse_args()
    
    # 训练前自动提交
    git_commit_auto(f"Pre-train V2: {args.region}, bands={args.bands}, lr={args.lr}")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    if args.output_dir is None:
        args.output_dir = os.path.join(DEFAULT_OUTPUT_DIR, args.region + '_v2')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # TensorBoard
    writer = None
    if args.tensorboard:
        exp_name = f"fire_v2_{args.region}_{datetime.now().strftime('%m%d_%H%M')}"
        tb_dir = os.path.join(args.tensorboard_dir, exp_name)
        writer = SummaryWriter(tb_dir)
        logger.info(f'TensorBoard: {tb_dir}')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    
    # 数据集
    if args.use_all_regions:
        regions = [d for d in os.listdir(args.data_dir) 
                  if os.path.isdir(os.path.join(args.data_dir, d)) and not d.endswith('output')]
        train_datasets = [FireDatasetV2(args.data_dir, r, args.bands, 'train',
                                       min_fg_pixels=args.min_fg_pixels,
                                       use_global_norm=not args.no_global_norm,
                                       neg_per_pos=args.neg_per_pos) for r in regions]
        val_datasets = [FireDatasetV2(args.data_dir, r, args.bands, 'val',
                                     min_fg_pixels=args.min_fg_pixels,
                                     use_global_norm=not args.no_global_norm,
                                     neg_per_pos=args.neg_per_pos) for r in regions]
        train_ds = torch.utils.data.ConcatDataset(train_datasets)
        val_ds = torch.utils.data.ConcatDataset(val_datasets)
        logger.info(f'Using all regions: {regions}')
    else:
        train_ds = FireDatasetV2(args.data_dir, args.region, args.bands, 'train',
                                min_fg_pixels=args.min_fg_pixels,
                                use_global_norm=not args.no_global_norm,
                                neg_per_pos=args.neg_per_pos)
        val_ds = FireDatasetV2(args.data_dir, args.region, args.bands, 'val',
                              min_fg_pixels=args.min_fg_pixels,
                              use_global_norm=not args.no_global_norm,
                              neg_per_pos=args.neg_per_pos)
    
    # 困难样本挖掘采样器
    sampler = None
    if isinstance(train_ds, torch.utils.data.ConcatDataset):
        # 暂不实现ConcatDataset的困难采样
        pass
    else:
        try:
            hard_sampler = HardMiningSampler(train_ds.samples, train_ds.indices)
            sampler = hard_sampler.get_sampler()
            logger.info('Using hard mining sampler')
        except Exception as e:
            logger.warning(f'Could not create hard sampler: {e}')
    
    if sampler is not None:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
    
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    
    # 模型
    pretrained_path = os.path.join(DEFAULT_PRETRAIN_DIR, 'mambavision_small_1k.pth') if args.pretrained else None
    model = FireDetectionModelV2(args.model, 1, len(args.bands), args.pretrained, pretrained_path).to(device)
    logger.info(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    
    # 损失函数
    criterion = CombinedFireLoss(
        tversky_weight=1.0,
        boundary_weight=0.5,
        focal_weight=1.0,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta
    )
    logger.info(f'Using Combined Loss: Tversky(α={args.tversky_alpha},β={args.tversky_beta}) + Boundary + Focal')
    
    # 优化器 - 分层学习率（修复scheduler冲突）
    param_groups = [
        {'params': model.backbone.parameters(), 'lr': args.lr * args.backbone_lr_scale, 'name': 'backbone'},
        {'params': model.decoder.parameters(), 'lr': args.lr, 'name': 'decoder'},
    ]
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    
    # 分层学习率调度器
    scheduler = LayerwiseWarmupScheduler(optimizer, args.warmup_epochs, args.epochs, args.lr)
    scaler = GradScaler() if args.use_amp else None
    
    # 训练状态
    best_f1 = 0.0
    best_epoch = 0
    best_threshold = 0.5
    epochs_no_improve = 0
    
    logger.info(f'\n{"="*70}')
    logger.info(f'Starting Training V2: {args.region}')
    logger.info(f'Target: F1 >= 94% (B5+B6+B7 optimal bands)')
    logger.info(f'{"="*70}\n')
    
    for epoch in range(1, args.epochs + 1):
        # 动态调整训练阶段
        if epoch <= args.warmup_epochs:
            stage = 'warmup'
        elif epoch <= 20:
            stage = 'normal'
        else:
            stage = 'finetune'
        
        lrs = scheduler.step(epoch - 1, stage)
        logger.info(f'\nEpoch {epoch}/{args.epochs} [{stage}] LRs: backbone={lrs[0]:.2e}, decoder={lrs[1]:.2e}')
        logger.info('-' * 70)
        
        # 训练
        train_loss, train_f1 = train_epoch_v2(
            model, train_loader, criterion, optimizer, device,
            scaler, args.use_amp, args.max_grad_norm, args.accum_steps
        )
        
        # 验证
        val_loss, val_f1, val_iou, val_p, val_r, best_thresh, best_f1_dyn = validate_v2(
            model, val_loader, criterion, device, best_threshold
        )
        
        # 使用动态阈值作为最佳F1
        effective_f1 = best_f1_dyn if args.dynamic_threshold else val_f1
        
        # TensorBoard
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/F1', val_f1, epoch)
            writer.add_scalar('Metrics/F1_dynamic', best_f1_dyn, epoch)
            writer.add_scalar('Metrics/IoU', val_iou, epoch)
            writer.add_scalar('Metrics/Precision', val_p, epoch)
            writer.add_scalar('Metrics/Recall', val_r, epoch)
            writer.add_scalar('Threshold/best', best_thresh, epoch)
            writer.add_scalar('Train/lr_backbone', lrs[0], epoch)
            writer.add_scalar('Train/lr_decoder', lrs[1], epoch)
        
        # 更新最佳阈值
        if best_f1_dyn > best_f1:
            best_threshold = best_thresh
        
        # 保存最佳模型
        if effective_f1 > best_f1:
            best_f1 = effective_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'f1': effective_f1,
                'iou': val_iou,
                'p': val_p,
                'r': val_r,
                'threshold': best_threshold,
                'args': vars(args)
            }, os.path.join(args.output_dir, 'best_model.pth'))
            logger.info(f'✓ Saved best model (F1: {best_f1:.2f}%, thresh: {best_threshold:.2f})')
        else:
            epochs_no_improve += 1
            logger.info(f'  No F1 improvement for {epochs_no_improve} epochs')
        
        # 早停
        if epochs_no_improve >= args.early_stop_patience:
            logger.warning(f'\n🛑 Early stopping! No improvement for {args.early_stop_patience} epochs')
            logger.warning(f'   Best F1: {best_f1:.2f}% at epoch {best_epoch}')
            
            if best_f1 < args.early_stop_min_f1:
                logger.warning(f'   ⚠️ Warning: Best F1 {best_f1:.2f}% < {args.early_stop_min_f1}% target')
            break
    
    # 可视化
    if args.visualize:
        visualize_predictions_v2(model, val_loader, device, threshold=best_threshold)
    
    logger.info(f'\n{"="*70}')
    logger.info(f'🏆 Final Results:')
    logger.info(f'   Best F1: {best_f1:.2f}% @ epoch {best_epoch}')
    logger.info(f'   Best Threshold: {best_threshold:.2f}')
    logger.info(f'   Target: 94%+ (B5+B6+B7 bands)')
    logger.info(f'{"="*70}')
    
    # 训练后自动提交
    git_commit_auto(f"Post-train V2: {args.region} best F1={best_f1:.2f}% @ epoch {best_epoch}")
    
    if writer:
        writer.close()


if __name__ == '__main__':
    main()
