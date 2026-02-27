#!/usr/bin/env python3
"""
train_landsat_v3.py - 火点检测综合优化版
GitHub: https://github.com/billy31/2602_fy4b_activefire

【2025-02-27 V3重构版】
【2026-02-27 更新】
- 标签统一二值化（label > 0），避免255/多类导致F1为0
- EMA加入warmup控制，避免早期EMA导致验证全负
- 验证阈值搜索扩展到0.05-0.95
- 加入负样本下采样，避免训练集全正导致验证阈值过高
- 正例增强：火点区域裁剪放大（Crop-to-Fire）
- 训练采用均衡采样（WeightedRandomSampler）
- 训练/验证输出mIoU、F1、Recall、Precision

=== 核心改进 ===
1. 灵活区域选择: 支持单区域、多区域(逗号分隔)、ALL全部区域
2. 自适应波段: 支持任意波段组合 (推荐: 5,6,7 或 7,6,2)
3. 架构优化: 
   - DeepLabV3+ 风格解码器 (ASPP + Skip Connection)
   - 可选UNet风格轻量解码器
   - 多尺度特征融合
4. 训练策略:
   - EMA (指数移动平均) 权重平滑
   - 更精细的学习率调度 (Cosine + Warmup + ReduceLROnPlateau)
   - 分层训练: 先冻结backbone训练decoder, 再联合训练
5. 损失函数:
   - BCE + Dice + Focal 组合
   - Tversky Loss 选项 (可调FP/FN比例)
   - OHEM (在线难样本挖掘)
6. 数据增强:
   - 火点专属增强 (Copy-Paste, Crop-to-Fire)
   - MixUp/CutMix支持

使用方式:
  python train_landsat_v3.py Asia1                    # 单区域
  python train_landsat_v3.py Asia1,Asia2,Asia3        # 多区域
  python train_landsat_v3.py ALL                      # 全部区域
  python train_landsat_v3.py Asia1 --bands 7 6 2      # 自定义波段
"""

import os
import sys
import argparse
import logging
import numpy as np
import cv2
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, '/root/codes/fire0226/MambaVision')
from mambavision import create_model

import rasterio
from scipy.ndimage import distance_transform_edt, gaussian_filter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = '/root/autodl-tmp/training'
DEFAULT_PRETRAIN_DIR = '/root/autodl-tmp/pretrained'
DEFAULT_OUTPUT_DIR = '/root/autodl-tmp/training/output'
DEFAULT_TENSORBOARD_DIR = '/root/tf-logs'

# ============================================================================
# 全局归一化统计
# ============================================================================

LANDSAT_GLOBAL_STATS = {
    1: {'p1': 7000, 'p99': 18000, 'name': 'Coastal'},
    2: {'p1': 7000, 'p99': 19000, 'name': 'Blue'},
    3: {'p1': 8000, 'p99': 22000, 'name': 'Green'},
    4: {'p1': 7000, 'p99': 24000, 'name': 'Red'},
    5: {'p1': 8000, 'p99': 30000, 'name': 'NIR'},
    6: {'p1': 5000, 'p99': 18000, 'name': 'SWIR1'},
    7: {'p1': 3000, 'p99': 14000, 'name': 'SWIR2'},
}


def git_auto_commit(message: str):
    """自动提交代码到GitHub"""
    try:
        import subprocess
        
        # 检查Git仓库
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, 
                              cwd='/root/codes/fire0226')
        
        if result.returncode == 0:
            # 有变更则提交
            if result.stdout.strip():
                subprocess.run(['git', 'add', '-A'], 
                             cwd='/root/codes/fire0226', check=True)
                commit_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {message}"
                subprocess.run(['git', 'commit', '-m', commit_msg], 
                             cwd='/root/codes/fire0226', check=True)
                
                # 尝试推送
                push_result = subprocess.run(['git', 'push'], 
                                           capture_output=True, text=True,
                                           cwd='/root/codes/fire0226')
                if push_result.returncode == 0:
                    logger.info(f'✅ GitHub synced: {message[:50]}...')
                else:
                    logger.warning(f'⚠️ Git commit OK but push failed')
            else:
                logger.info('ℹ️ No code changes to commit')
        else:
            logger.warning('⚠️ Not a git repository or git error')
    except Exception as e:
        logger.warning(f'⚠️ Git auto-commit failed: {e}')


def get_band_stats(band_idx: int) -> Dict[str, float]:
    return LANDSAT_GLOBAL_STATS.get(band_idx, {'p1': 5000, 'p99': 20000})


# ============================================================================
# EMA (指数移动平均) - 提升模型稳定性
# ============================================================================

class ModelEMA:
    """模型参数的指数移动平均"""
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
    
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        return self.shadow
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict


# ============================================================================
# 损失函数组合
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss - 聚焦难样本"""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        probs = torch.sigmoid(pred)
        pt = torch.where(target > 0, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class DiceLoss(nn.Module):
    """Dice Loss - 直接优化IoU"""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(pred)
        intersection = (probs * target).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return (1 - dice).mean()


class TverskyLoss(nn.Module):
    """Tversky Loss - 可调FP/FN权重"""
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha  # FP权重
        self.beta = beta    # FN权重（降低更关注FN/召回）
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(pred)
        tp = (probs * target).sum(dim=(1, 2))
        fp = (probs * (1 - target)).sum(dim=(1, 2))
        fn = ((1 - probs) * target).sum(dim=(1, 2))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - tversky).mean()


class OHEMLoss(nn.Module):
    """在线难样本挖掘"""
    def __init__(self, ratio: float = 0.25):
        super().__init__()
        self.ratio = ratio
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.bce(pred, target)
        # 选择损失最大的样本
        num_hard = int(self.ratio * loss.numel())
        hard_loss, _ = torch.topk(loss.view(-1), num_hard)
        return hard_loss.mean()


class CombinedLoss(nn.Module):
    """组合损失：BCE(pos_weight) + Dice + Focal + Tversky(可选)"""
    def __init__(self, 
                 pos_weight: float = 10.0,
                 bce_weight: float = 1.0,
                 dice_weight: float = 1.0,
                 focal_weight: float = 0.5,
                 tversky_weight: float = 0.0,
                 tversky_alpha: float = 0.3,
                 use_ohem: bool = False):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        
        self.pos_weight = torch.tensor([pos_weight])
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        
        if tversky_weight > 0:
            self.tversky = TverskyLoss(alpha=tversky_alpha, beta=1-tversky_alpha)
        
        if use_ohem:
            self.ohem = OHEMLoss()
        else:
            self.ohem = None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        
        # BCE with pos_weight
        if self.pos_weight.device != pred.device:
            self.pos_weight = self.pos_weight.to(pred.device)
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=self.pos_weight)
        
        total = self.bce_weight * bce_loss
        
        if self.dice_weight > 0:
            total += self.dice_weight * self.dice(pred, target)
        
        if self.focal_weight > 0:
            total += self.focal_weight * self.focal(pred, target)
        
        if self.tversky_weight > 0 and hasattr(self, 'tversky'):
            total += self.tversky_weight * self.tversky(pred, target)
        
        if self.ohem is not None:
            total += self.ohem(pred, target)
        
        return total


# ============================================================================
# ASPP (Atrous Spatial Pyramid Pooling) - DeepLabV3+核心
# ============================================================================

class ASPP(nn.Module):
    """空洞空间金字塔池化"""
    def __init__(self, in_ch: int, out_ch: int = 256, rates: List[int] = [6, 12, 18]):
        super().__init__()
        
        # 1x1卷积分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
        # 空洞卷积分支
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ) for rate in rates
        ])
        
        # 全局平均池化分支
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
        # 融合
        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_ch * (2 + len(rates)), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        
        # 各分支
        feat1 = self.branch1(x)
        feats = [branch(x) for branch in self.branches]
        
        # 全局分支需要上采样
        global_feat = self.global_branch(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', align_corners=False)
        
        # 拼接
        concat_feat = torch.cat([feat1] + feats + [global_feat], dim=1)
        
        return self.conv_cat(concat_feat)


# ============================================================================
# 解码器选择
# ============================================================================

class DeepLabV3PlusDecoder(nn.Module):
    """DeepLabV3+风格解码器 - ASPP + Skip Connection"""
    def __init__(self, encoder_dim: int, num_classes: int = 1, low_level_dim: int = 96):
        super().__init__()
        
        # ASPP模块处理最深特征
        self.aspp = ASPP(encoder_dim, 256, rates=[6, 12, 18])
        
        # 处理浅层特征 (low-level features)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_dim, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # 融合后的卷积 - 增加Dropout防过拟合
        self.classifier = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(128, num_classes, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
    
    def forward(self, features: List[torch.Tensor], input_shape: Tuple[int, int]) -> torch.Tensor:
        """
        features: [f_low, f_high] 
        f_low: 浅层特征 (1/4分辨率)
        f_high: 深层特征 (1/16或1/32)
        """
        f_low, f_high = features[0], features[-1]
        
        # ASPP处理深层特征
        x = self.aspp(f_high)
        
        # 上采样4倍
        x = F.interpolate(x, size=f_low.shape[2:], mode='bilinear', align_corners=False)
        
        # 处理浅层特征
        low_level_feat = self.low_level_conv(f_low)
        
        # 拼接
        x = torch.cat([x, low_level_feat], dim=1)
        
        # 分类
        x = self.classifier(x)
        
        # 上采样到原图尺寸
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        return x


class FPNFireDecoder(nn.Module):
    """FPN风格解码器 - 多尺度特征融合"""
    def __init__(self, encoder_dims: List[int], num_classes: int = 1):
        super().__init__()
        
        # 自顶向下路径
        self.lateral4 = nn.Conv2d(encoder_dims[3], 256, 1)
        self.lateral3 = nn.Conv2d(encoder_dims[2], 256, 1)
        self.lateral2 = nn.Conv2d(encoder_dims[1], 256, 1)
        self.lateral1 = nn.Conv2d(encoder_dims[0], 128, 1)
        
        # 平滑卷积
        self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth1 = nn.Conv2d(128, 128, 3, padding=1)
        
        # 上采样
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # 分割头
        self.seg_head = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, num_classes, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
    
    def forward(self, features: List[torch.Tensor], input_shape: Tuple[int, int]) -> torch.Tensor:
        f1, f2, f3, f4 = features
        
        # 自顶向下
        p4 = self.lateral4(f4)
        
        p3 = self.lateral3(f3) + self.upsample(p4)
        p3 = self.smooth3(p3)
        
        p2 = self.lateral2(f2) + self.upsample(p3)
        p2 = self.smooth2(p2)
        
        p1 = self.lateral1(f1) + self.upsample(p2)
        p1 = self.smooth1(p1)
        
        # 分割
        out = self.seg_head(p1)
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        
        return out


# ============================================================================
# 模型架构
# ============================================================================

def extract_features(backbone: nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
    """提取多尺度特征 [f1, f2, f3, f4]"""
    features = []
    
    x = backbone.patch_embed(x)
    features.append(x)  # f1: 1/4
    
    for i, level in enumerate(backbone.levels):
        x = level(x)
        if i < 3:
            features.append(x)
    
    x = backbone.norm(x)
    features.append(x)  # f4: 1/32
    
    return features[:4]


class FireDetectionModel(nn.Module):
    """火点检测模型 - 支持多种解码器"""
    def __init__(self, 
                 model_name: str = 'mamba_vision_S',
                 num_classes: int = 1,
                 input_channels: int = 3,
                 decoder_type: str = 'deeplabv3plus',
                 pretrained: bool = True,
                 pretrained_path: Optional[str] = None):
        super().__init__()
        
        self.decoder_type = decoder_type
        
        # Backbone
        self.backbone = create_model(model_name, pretrained=False, num_classes=num_classes)
        
        if input_channels != 3:
            self._modify_input(input_channels)
        
        if pretrained and pretrained_path:
            self._load_pretrained(pretrained_path)
        
        # 维度配置
        dims_map = {
            'mamba_vision_T': ([96, 192, 384, 384], 384),
            'mamba_vision_S': ([96, 192, 384, 768], 768),
            'mamba_vision_B': ([128, 256, 512, 1024], 1024),
            'mamba_vision_L': ([128, 256, 512, 1568], 1568),
        }
        self.encoder_dims, deepest_dim = dims_map.get(model_name, ([96, 192, 384, 768], 768))
        
        # 解码器选择
        if decoder_type == 'deeplabv3plus':
            self.decoder = DeepLabV3PlusDecoder(deepest_dim, num_classes, low_level_dim=self.encoder_dims[0])
        elif decoder_type == 'fpn':
            self.decoder = FPNFireDecoder(self.encoder_dims, num_classes)
        else:
            raise ValueError(f"Unknown decoder: {decoder_type}")
    
    def _modify_input(self, input_channels: int):
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
            logger.info(f"Modified input layer to {input_channels} channels")
    
    def _load_pretrained(self, path: str):
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt.get('model', ckpt))
        state = {k: v for k, v in state.items() if not k.startswith('head.')}
        
        model_state = self.backbone.state_dict()
        matched = 0
        
        filtered_state = {}
        for k, v in state.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered_state[k] = v
                matched += 1
        
        self.backbone.load_state_dict(filtered_state, strict=False)
        total = len(model_state)
        logger.info(f"Loaded pretrained: {matched}/{total} ({100*matched/total:.1f}%)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2:]
        features = extract_features(self.backbone, x)
        out = self.decoder(features, (H, W))
        return out
    
    def get_param_groups(self, lr: float, backbone_lr_scale: float) -> List[Dict]:
        return [
            {'params': self.backbone.parameters(), 'lr': lr * backbone_lr_scale, 'name': 'backbone'},
            {'params': self.decoder.parameters(), 'lr': lr, 'name': 'decoder'},
        ]


# ============================================================================
# 数据集
# ============================================================================

class FireDataset(Dataset):
    """火点数据集 - 支持任意波段组合"""
    def __init__(self, data_dir: str, region: str, bands: List[int] = [5, 6, 7],
                 mode: str = 'train', split: float = 0.8, seed: int = 42,
                 min_fg_pixels: int = 5, neg_per_pos: float = 0.5,
                 crop_to_fire: bool = True, crop_scale_min: float = 0.6, crop_scale_max: float = 1.0):
        self.raw_dir = os.path.join(data_dir, region, 'raw')
        self.label_dir = os.path.join(data_dir, region, 'mask_label')
        self.bands = bands
        self.mode = mode
        self.neg_per_pos = max(0.0, float(neg_per_pos))
        self.rng = np.random.default_rng(seed)
        self.crop_to_fire = crop_to_fire
        self.crop_scale_min = float(crop_scale_min)
        self.crop_scale_max = float(crop_scale_max)
        
        samples = self._scan_samples()
        self.samples = self._filter_fire(samples, min_fg_pixels)
        
        np.random.seed(seed)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(len(indices) * split)
        
        if mode == 'train':
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
        
        self.band_stats = [get_band_stats(b) for b in bands]
        self._compute_stats()
        
        band_names = [LANDSAT_GLOBAL_STATS.get(b, {}).get('name', f'B{b}') for b in bands]
        logger.info(f"[{mode}] {region}: {len(self.indices)} patches, bands={band_names}, neg/pos≈{self.neg_pos_ratio:.0f}:1")
    
    def _scan_samples(self) -> List[Dict]:
        samples = []
        if not os.path.exists(self.label_dir):
            return samples
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
        # 负样本下采样：避免训练集几乎全正
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
    
    def _compute_stats(self):
        total_fg = 0
        total_pixels = 0
        for idx in self.indices[:min(50, len(self.indices))]:
            try:
                with rasterio.open(self.samples[idx]['label']) as src:
                    label = src.read(1)
                label = self._binarize_label(label)
                total_fg += (label > 0).sum()
                total_pixels += label.size
            except:
                pass
        self.neg_pos_ratio = (total_pixels - total_fg) / max(total_fg, 1)
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[self.indices[idx]]
        
        with rasterio.open(s['raw']) as src:
            available_bands = list(range(1, src.count + 1))
            bands_to_read = [min(b, src.count) for b in self.bands]
            image = src.read(bands_to_read).astype(np.float32)
        
        with rasterio.open(s['label']) as src:
            label = src.read(1)
        label = self._binarize_label(label)
        
        image = self._normalize(image)
        
        if self.mode == 'train':
            image, label = self._augment(image, label)
        
        return torch.from_numpy(image).float(), torch.from_numpy(label.astype(np.int64))

    @staticmethod
    def _binarize_label(label: np.ndarray) -> np.ndarray:
        # 兼容0/1、0/255、或多值标签
        return (label > 0).astype(np.uint8)
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        for i in range(image.shape[0]):
            p1, p99 = self.band_stats[i]['p1'], self.band_stats[i]['p99']
            band = np.clip(image[i], p1, p99)
            image[i] = (band - p1) / (p99 - p1 + 1e-8)
        return np.clip(image, 0, 1)
    
    def _augment(self, img: np.ndarray, lbl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        C, H, W = img.shape
        has_fire = (lbl > 0).any()
        
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
        
        # 火点专属增强
        if has_fire and self.mode == 'train':
            # Crop-to-Fire: 放大火点区域，提高正例像素占比
            if self.crop_to_fire and np.random.rand() < 0.6:
                img, lbl = self._crop_to_fire(img, lbl)
            # Copy-Paste增强
            if np.random.rand() < 0.3:
                img, lbl = self._fire_copy_paste(img, lbl)
            
            # 燃烧强度增强
            if np.random.rand() < 0.4:
                fire_mask = lbl > 0
                intensity = 1.1 + 0.4 * np.random.rand()
                for c in range(min(3, C)):
                    img_c = img[c].copy()
                    img_c[fire_mask] = np.clip(img_c[fire_mask] * intensity, 0, 1)
                    img[c] = img_c
        
        # 通用增强
        if np.random.rand() < 0.3:
            img = img * (0.9 + 0.2 * np.random.rand())
        if np.random.rand() < 0.3:
            img = img + np.random.normal(0, 0.01, img.shape)
        
        return np.clip(img, 0, 1), lbl

    def _crop_to_fire(self, img: np.ndarray, lbl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        C, H, W = img.shape
        ys, xs = np.where(lbl > 0)
        if len(ys) == 0:
            return img, lbl
        # 以火点为中心随机裁剪并缩放回原尺寸
        center_idx = self.rng.integers(0, len(ys))
        cy, cx = int(ys[center_idx]), int(xs[center_idx])
        scale = self.rng.uniform(self.crop_scale_min, self.crop_scale_max)
        ch = max(16, int(H * scale))
        cw = max(16, int(W * scale))
        y1 = np.clip(cy - ch // 2, 0, H - ch)
        x1 = np.clip(cx - cw // 2, 0, W - cw)
        y2 = y1 + ch
        x2 = x1 + cw
        img_crop = img[:, y1:y2, x1:x2]
        lbl_crop = lbl[y1:y2, x1:x2]
        img_resized = np.stack([cv2.resize(img_crop[c], (W, H), interpolation=cv2.INTER_LINEAR) for c in range(C)], axis=0)
        lbl_resized = cv2.resize(lbl_crop.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        return img_resized, lbl_resized

    def get_sample_weights(self) -> List[float]:
        # 按是否含火点进行均衡采样
        weights = []
        for s in self.samples:
            weights.append(1.0 if s.get('fg_count', 0) > 0 else 0.25)
        return weights
    
    def _fire_copy_paste(self, img: np.ndarray, lbl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        C, H, W = img.shape
        fire_mask = lbl > 0
        if not fire_mask.any():
            return img, lbl
        
        result_img, result_lbl = img.copy(), lbl.copy()
        
        for _ in range(np.random.randint(1, 3)):
            y_offset = np.random.randint(-H//4, H//4)
            x_offset = np.random.randint(-W//4, W//4)
            
            M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
            shifted_mask = cv2.warpAffine(fire_mask.astype(np.uint8), M, (W, H)) > 0
            
            overlap = (result_lbl > 0) & shifted_mask
            if overlap.sum() < shifted_mask.sum() * 0.3:
                for c in range(C):
                    shifted_band = cv2.warpAffine(img[c], M, (W, H))
                    result_img[c] = np.where(shifted_mask, shifted_band, result_img[c])
                result_lbl = np.maximum(result_lbl, shifted_mask.astype(np.int64))
        
        return result_img, result_lbl


# ============================================================================
# 训练函数
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device, scaler, use_amp, max_grad_norm,
                ema=None, update_ema: bool = True):
    model.train()
    total_loss = 0.0
    tp = fp = fn = 0
    
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs.squeeze(1), labels.float())
        
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        # EMA更新
        if ema is not None and update_ema:
            ema.update()
        
        total_loss += loss.item()
        
        # 统计
        with torch.no_grad():
            probs = torch.sigmoid(outputs).squeeze(1)
            preds = (probs > 0.5).long()
            gt = (labels > 0).long()
            tp += ((preds == 1) & (gt == 1)).sum().item()
            fp += ((preds == 1) & (gt == 0)).sum().item()
            fn += ((preds == 0) & (gt == 1)).sum().item()
        
        if i % 20 == 0:
            p = tp / (tp + fp + 1e-8) * 100
            r = tp / (tp + fn + 1e-8) * 100
            f1 = 2 * p * r / (p + r + 1e-8)
            logger.info(f'  [{i}/{len(loader)}] Loss:{loss.item():.4f} P:{p:.1f}% R:{r:.1f}% F1:{f1:.1f}%')
    
    avg_loss = total_loss / len(loader)
    precision = tp / (tp + fp + 1e-8) * 100
    recall = tp / (tp + fn + 1e-8) * 100
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8) * 100
    logger.info(f'Train - Loss:{avg_loss:.4f} mIoU:{iou:.2f}% P:{precision:.2f}% R:{recall:.2f}% F1:{f1:.2f}%')
    return avg_loss, f1


@torch.no_grad()
def validate(model, loader, criterion, device, ema=None, use_ema: bool = True):
    if ema is not None and use_ema:
        ema.apply_shadow()
    
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs.squeeze(1), labels.float())
        total_loss += loss.item()
        
        probs = torch.sigmoid(outputs).squeeze(1)
        all_probs.append(probs.cpu())
        all_targets.append(labels.cpu())
    
    all_probs = torch.cat([p.flatten() for p in all_probs])
    all_targets = torch.cat([t.flatten() for t in all_targets])
    
    # 阈值搜索
    best_f1 = 0
    best_thresh = 0.5
    best_tp = best_fp = best_fn = 0
    thresholds = np.linspace(0.05, 0.95, 19).tolist()
    for thresh in thresholds:
        preds = (all_probs > thresh).long()
        gt = (all_targets > 0).long()
        tp = ((preds == 1) & (gt == 1)).sum().item()
        fp = ((preds == 1) & (gt == 0)).sum().item()
        fn = ((preds == 0) & (gt == 1)).sum().item()
        
        p = tp / (tp + fp + 1e-8) * 100
        r = tp / (tp + fn + 1e-8) * 100
        f1 = 2 * p * r / (p + r + 1e-8)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_tp, best_fp, best_fn = tp, fp, fn
    
    if ema is not None and use_ema:
        ema.restore()

    avg_loss = total_loss / len(loader)
    pos_ratio = all_targets.float().mean().item() * 100
    # best_f1对应的最后一次tp/fp/fn
    iou = best_tp / (best_tp + best_fp + best_fn + 1e-8) * 100
    precision = best_tp / (best_tp + best_fp + 1e-8) * 100
    recall = best_tp / (best_tp + best_fn + 1e-8) * 100
    logger.info(f'Val - Loss:{avg_loss:.4f} mIoU:{iou:.2f}% P:{precision:.2f}% R:{recall:.2f}% '
                f'Best@{best_thresh:.2f} F1:{best_f1:.2f}% (pos={pos_ratio:.4f}%)')
    return avg_loss, best_f1, best_thresh


# ============================================================================
# 主函数
# ============================================================================

def parse_regions_arg(regions_arg: str, data_dir: str) -> List[str]:
    """解析区域参数"""
    if regions_arg.upper() == 'ALL':
        # 扫描所有有效区域
        regions = []
        for d in sorted(os.listdir(data_dir)):
            dir_path = os.path.join(data_dir, d)
            if os.path.isdir(dir_path) and d not in ['output', 'meta', '.git'] and not d.startswith('.'):
                # 检查是否有有效数据
                raw_dir = os.path.join(dir_path, 'raw')
                label_dir = os.path.join(dir_path, 'mask_label')
                if os.path.exists(raw_dir) and os.path.exists(label_dir):
                    regions.append(d)
        logger.info(f"Auto-detected regions: {regions}")
        return regions
    else:
        # 逗号分隔的多区域
        return [r.strip() for r in regions_arg.split(',')]


def main():
    parser = argparse.ArgumentParser(description='Fire Detection Training V3')
    
    # 区域参数 - 支持单区域、多区域(逗号分隔)、ALL全部
    parser.add_argument('regions', type=str, help='Region(s): Asia1 | Asia1,Asia2,Asia3 | ALL')
    
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--tensorboard-dir', type=str, default=DEFAULT_TENSORBOARD_DIR)
    
    # 数据参数 - 支持任意波段组合
    parser.add_argument('--bands', type=int, nargs='+', default=[5, 6, 7],
                       help='Bands to use (default: 5,6,7). Options: 1-7')
    parser.add_argument('--min-fg-pixels', type=int, default=5)
    parser.add_argument('--neg-per-pos', type=float, default=0.5,
                       help='Keep negatives per positive (default: 0.5)')
    parser.add_argument('--crop-to-fire', action='store_true', default=True,
                       help='Enable crop-to-fire augmentation')
    parser.add_argument('--crop-scale-min', type=float, default=0.6)
    parser.add_argument('--crop-scale-max', type=float, default=1.0)
    
    # 模型参数
    parser.add_argument('--model', type=str, default='mamba_vision_S',
                       choices=['mamba_vision_T', 'mamba_vision_S', 'mamba_vision_B', 'mamba_vision_L'])
    parser.add_argument('--decoder', type=str, default='deeplabv3plus',
                       choices=['deeplabv3plus', 'fpn'],
                       help='Decoder type: deeplabv3plus (ASPP+Skip) or fpn')
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--pretrained-path', type=str, default=None)
    
    # 损失参数
    parser.add_argument('--pos-weight', type=float, default=5.0)
    parser.add_argument('--bce-weight', type=float, default=1.0)
    parser.add_argument('--dice-weight', type=float, default=1.0)
    parser.add_argument('--focal-weight', type=float, default=0.5)
    parser.add_argument('--tversky-weight', type=float, default=0.0)
    parser.add_argument('--tversky-alpha', type=float, default=0.3)
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4) - 降低防止过拟合')
    parser.add_argument('--backbone-lr-scale', type=float, default=0.05,
                       help='Backbone LR scale (default: 0.05) - 更低保护预训练')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                       help='Weight decay (default: 0.05) - 增强正则化')
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    
    # 高级训练策略
    parser.add_argument('--use-ema', action='store_true', default=True,
                       help='Use EMA (Exponential Moving Average)')
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--ema-warmup-epochs', type=int, default=5,
                       help='Delay EMA updates/validation to avoid early underfit')
    parser.add_argument('--use-amp', action='store_true', default=True)
    
    # 早停
    parser.add_argument('--early-stop-patience', type=int, default=10)
    parser.add_argument('--early-stop-min-f1', type=float, default=85.0)
    
    # 其他
    parser.add_argument('--tensorboard', action='store_true', default=True)
    
    args = parser.parse_args()
    
    # 解析区域
    regions = parse_regions_arg(args.regions, args.data_dir)
    if len(regions) == 0:
        raise ValueError(f"No valid regions found")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 输出目录
    if args.output_dir is None:
        region_name = regions[0] if len(regions) == 1 else f"multi{len(regions)}"
        args.output_dir = os.path.join(DEFAULT_OUTPUT_DIR, f'{region_name}_v3_{args.decoder}')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Git自动提交 - 训练前
    git_auto_commit(f"Start training: {regions}, bands={args.bands}, decoder={args.decoder}")
    
    # TensorBoard - 命名规则: 时间+fire+trainingLandsat
    writer = None
    if args.tensorboard:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"{timestamp}_fire_trainingLandsat"
        tb_dir = os.path.join(args.tensorboard_dir, exp_name)
        writer = SummaryWriter(tb_dir)
        logger.info(f'TensorBoard: {tb_dir}')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    logger.info(f'Model: {args.model} + {args.decoder}')
    logger.info(f'Bands: {args.bands}')
    logger.info(f'Regions: {regions}')
    
    # 加载数据集
    train_datasets = []
    val_datasets = []
    neg_pos_ratios = []
    
    for region in regions:
        try:
            train_ds = FireDataset(args.data_dir, region, args.bands, 'train',
                                   min_fg_pixels=args.min_fg_pixels, neg_per_pos=args.neg_per_pos,
                                   crop_to_fire=args.crop_to_fire,
                                   crop_scale_min=args.crop_scale_min, crop_scale_max=args.crop_scale_max)
            val_ds = FireDataset(args.data_dir, region, args.bands, 'val',
                                 min_fg_pixels=args.min_fg_pixels, neg_per_pos=args.neg_per_pos,
                                 crop_to_fire=False,
                                 crop_scale_min=args.crop_scale_min, crop_scale_max=args.crop_scale_max)
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            neg_pos_ratios.append(train_ds.neg_pos_ratio)
        except Exception as e:
            logger.warning(f'Skip region {region}: {e}')
    
    if len(train_datasets) == 0:
        raise ValueError("No valid datasets")
    
    train_ds = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_ds = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
    
    avg_neg_pos = np.mean(neg_pos_ratios)
    logger.info(f'Total: train={len(train_ds)}, val={len(val_ds)}, avg neg/pos={avg_neg_pos:.0f}:1')
    
    sampler = None
    weights = None
    if hasattr(train_ds, 'get_sample_weights'):
        weights = train_ds.get_sample_weights()
    elif isinstance(train_ds, ConcatDataset):
        all_weights = []
        for ds in train_ds.datasets:
            if hasattr(ds, 'get_sample_weights'):
                all_weights.extend(ds.get_sample_weights())
        if len(all_weights) == len(train_ds):
            weights = all_weights
    if weights is not None and len(weights) == len(train_ds):
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              shuffle=(sampler is None), num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    
    # 模型
    pretrained_path = args.pretrained_path or (os.path.join(DEFAULT_PRETRAIN_DIR, 'mambavision_small_1k.pth') 
                                               if args.pretrained else None)
    model = FireDetectionModel(args.model, 1, len(args.bands), args.decoder, args.pretrained, pretrained_path)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Params: {total_params/1e6:.2f}M')
    
    # 损失函数 - pos_weight限制在合理范围
    # neg_pos_ratio通常很大(2000:1)，但pos_weight过大会导致梯度爆炸和过拟合
    # 使用log缩放：sqrt(neg_pos_ratio) 更合理
    calculated_pw = min(np.sqrt(avg_neg_pos), 20.0)  # 限制最大20
    pos_weight = min(max(args.pos_weight, 1.0), calculated_pw)
    criterion = CombinedLoss(
        pos_weight=pos_weight,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight,
        tversky_weight=args.tversky_weight,
        tversky_alpha=args.tversky_alpha
    )
    logger.info(f'Loss: BCE(pw={pos_weight:.1f}, calc={calculated_pw:.1f}) + Dice({args.dice_weight}) + Focal({args.focal_weight})')
    
    # 优化器
    param_groups = model.get_param_groups(args.lr, args.backbone_lr_scale)
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    logger.info(f'LR: decoder={args.lr}, backbone={args.lr * args.backbone_lr_scale}')
    
    # 调度器
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6)
    scheduler_plateau = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # EMA
    ema = ModelEMA(model, args.ema_decay) if args.use_ema else None
    
    # AMP
    scaler = GradScaler() if args.use_amp else None
    
    # 训练循环
    best_f1 = 0.0
    best_epoch = 0
    best_thresh = 0.5
    epochs_no_improve = 0
    
    logger.info(f'\n{"="*60}')
    logger.info(f'Start Training: {len(regions)} region(s), {args.decoder} decoder')
    logger.info(f'{"="*60}\n')
    
    for epoch in range(1, args.epochs + 1):
        # Warmup
        if epoch <= args.warmup_epochs:
            lr = args.lr * epoch / args.warmup_epochs
            for param_group in optimizer.param_groups:
                if param_group.get('name') == 'backbone':
                    param_group['lr'] = lr * args.backbone_lr_scale
                else:
                    param_group['lr'] = lr
            logger.info(f'\nEpoch {epoch}/{args.epochs} [Warmup] lr={lr:.2e}')
        else:
            scheduler_cosine.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f'\nEpoch {epoch}/{args.epochs} lr={current_lr:.2e}')
        
        logger.info('-' * 60)
        
        # 训练
        use_ema_now = ema is not None and epoch > args.ema_warmup_epochs
        train_loss, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            args.use_amp, args.max_grad_norm, ema, update_ema=use_ema_now
        )
        
        # 验证
        val_loss, val_f1, val_thresh = validate(model, val_loader, criterion, device, ema, use_ema=use_ema_now)
        
        # Plateau调度
        if epoch > args.warmup_epochs:
            scheduler_plateau.step(val_f1)
        
        # TensorBoard
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/F1', val_f1, epoch)
            writer.add_scalar('Threshold/best', val_thresh, epoch)
            writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], epoch)
        
        # 保存
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_thresh = val_thresh
            epochs_no_improve = 0
            
            save_dict = {
                'epoch': epoch,
                'model': model.state_dict(),
                'ema': ema.state_dict() if ema else None,
                'f1': val_f1,
                'threshold': val_thresh,
                'args': vars(args)
            }
            torch.save(save_dict, os.path.join(args.output_dir, 'best_model.pth'))
            logger.info(f'✓ Saved best model (F1: {best_f1:.2f}%, thresh: {best_thresh:.2f})')
        else:
            epochs_no_improve += 1
            logger.info(f'  No improvement for {epochs_no_improve} epochs')
        
        # 早停
        if epochs_no_improve >= args.early_stop_patience:
            logger.warning(f'\n🛑 Early stopping! Best F1: {best_f1:.2f}% at epoch {best_epoch}')
            break
    
    logger.info(f'\n{"="*60}')
    logger.info(f'🏆 Best F1: {best_f1:.2f}% @ epoch {best_epoch}, threshold: {best_thresh:.2f}')
    logger.info(f'{"="*60}')
    
    # Git自动提交 - 训练后
    git_auto_commit(f"Complete: F1={best_f1:.2f}% @ epoch {best_epoch}, regions={regions}")
    
    if writer:
        writer.close()


if __name__ == '__main__':
    main()
