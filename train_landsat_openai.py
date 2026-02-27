#!/usr/bin/env python3
"""
train_landsat_v3.py - 火点检测综合优化版
GitHub: https://github.com/billy31/2602_fy4b_activefire

【2025-02-27 V3重构版】:

【OpenAI修订版说明（2026-02-27）】:
- 修复极端不平衡场景：训练集不再默认过滤全部负样本，新增 train_neg_ratio 控制负样本下采样
- 标签统一二值化(label==1)，避免非0/1标签导致损失/指标异常
- 验证阈值搜索从固定[0.3,0.4,0.5,0.6]扩展为低阈值优先的广覆盖搜索，并输出概率分布统计
- 训练策略改为“Warmup + 单调度器(默认Cosine)”并在epoch末step，避免warmup/调度冲突
- 新增冻结backbone前几轮训练decoder的阶段训练策略（默认3轮）
- 修复CLI开关风格：支持 --no-pretrained / --no-ema / --no-amp / --no-tensorboard


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
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import random
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
    """类不平衡友好的 Focal Loss（支持正负类不同alpha）"""
    def __init__(self, alpha_pos: float = 0.75, alpha_neg: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target shape: [B, H, W]
        target = (target > 0.5).float()
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        probs = torch.sigmoid(pred)
        pt = torch.where(target > 0.5, probs, 1 - probs)
        alpha = torch.where(target > 0.5,
                            torch.full_like(target, self.alpha_pos),
                            torch.full_like(target, self.alpha_neg))
        focal_weight = alpha * (1 - pt).pow(self.gamma)
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
    """组合损失：BCE(pos_weight) + Dice + Focal + Tversky(可选)

    改动说明：
    - 强制将标签二值化为 {0,1}，避免掩膜中存在非0/1值时损失异常
    - Focal支持正负类不同alpha，更适合极端不平衡火点像元
    """
    def __init__(self,
                 pos_weight: float = 10.0,
                 bce_weight: float = 1.0,
                 dice_weight: float = 1.0,
                 focal_weight: float = 0.5,
                 tversky_weight: float = 0.0,
                 tversky_alpha: float = 0.3,
                 focal_alpha_pos: float = 0.75,
                 focal_alpha_neg: float = 0.25,
                 use_ohem: bool = False):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight

        self.pos_weight = torch.tensor([pos_weight])
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha_pos=focal_alpha_pos, alpha_neg=focal_alpha_neg)

        if tversky_weight > 0:
            self.tversky = TverskyLoss(alpha=tversky_alpha, beta=1 - tversky_alpha)

        if use_ohem:
            self.ohem = OHEMLoss()
        else:
            self.ohem = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target shape: [B,H,W]
        target = (target == 1).float() if target.dtype != torch.float32 else (target > 0.5).float()

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
    """火点数据集 - 支持任意波段组合（修复极端不平衡场景）

    关键改动：
    1) 不再默认丢弃全部负样本（原实现会导致训练精度/泛化严重偏移）
    2) 训练集支持按负样本比例下采样（train_neg_ratio）
    3) 标签读取后统一二值化(label==1)，避免非0/1标签干扰损失与指标
    4) 验证集默认保留全部样本，评估更接近真实分布
    """
    def __init__(self, data_dir: str, region: str, bands: List[int] = [5, 6, 7],
                 mode: str = 'train', split: float = 0.8, seed: int = 42,
                 min_fg_pixels: int = 5,
                 train_neg_ratio: float = 2.0,
                 keep_all_neg_in_val: bool = True):
        self.raw_dir = os.path.join(data_dir, region, 'raw')
        self.label_dir = os.path.join(data_dir, region, 'mask_label')
        self.bands = bands
        self.mode = mode
        self.train_neg_ratio = train_neg_ratio
        self.keep_all_neg_in_val = keep_all_neg_in_val

        raw_samples = self._scan_samples()
        self.samples = self._annotate_samples(raw_samples, min_fg_pixels=min_fg_pixels)

        # 分层切分：正负样本分别切分，避免val全是极小火点/训练分布偏移
        pos_ids = [i for i, s in enumerate(self.samples) if s['is_pos']]
        neg_ids = [i for i, s in enumerate(self.samples) if not s['is_pos']]
        rng = np.random.RandomState(seed)
        pos_ids = list(rng.permutation(pos_ids))
        neg_ids = list(rng.permutation(neg_ids))

        pos_split = int(len(pos_ids) * split)
        neg_split = int(len(neg_ids) * split)

        if mode == 'train':
            sel_pos = pos_ids[:pos_split]
            sel_neg = neg_ids[:neg_split]
            if train_neg_ratio is not None and train_neg_ratio >= 0 and len(sel_pos) > 0:
                max_neg = int(max(1, round(len(sel_pos) * train_neg_ratio)))
                if len(sel_neg) > max_neg:
                    sel_neg = sel_neg[:max_neg]
            self.indices = np.array(sel_pos + sel_neg, dtype=np.int64)
            rng.shuffle(self.indices)
        else:
            sel_pos = pos_ids[pos_split:]
            sel_neg = neg_ids[neg_split:]
            if not keep_all_neg_in_val and len(sel_pos) > 0 and len(sel_neg) > len(sel_pos) * 5:
                sel_neg = sel_neg[:len(sel_pos) * 5]
            self.indices = np.array(sel_pos + sel_neg, dtype=np.int64)

        self.band_stats = [get_band_stats(b) for b in bands]
        self._compute_stats()

        band_names = [LANDSAT_GLOBAL_STATS.get(b, {}).get('name', f'B{b}') for b in bands]
        n_pos = sum(1 for i in self.indices if self.samples[int(i)]['is_pos'])
        n_neg = len(self.indices) - n_pos
        logger.info(
            f"[{mode}] {region}: {len(self.indices)} patches (pos={n_pos}, neg={n_neg}), "
            f"bands={band_names}, pixel neg/pos≈{self.neg_pos_ratio:.0f}:1"
        )

    def _scan_samples(self) -> List[Dict]:
        samples = []
        if not os.path.exists(self.label_dir):
            return samples
        for f in sorted(os.listdir(self.label_dir)):
            if '_voting_' in f and f.endswith('.tif'):
                raw_f = f.replace('_voting_', '_').replace('.tif', '.tif')
                raw_path = os.path.join(self.raw_dir, raw_f)
                label_path = os.path.join(self.label_dir, f)
                if os.path.exists(raw_path):
                    samples.append({'raw': raw_path, 'label': label_path, 'name': f})
        return samples

    def _read_binary_label(self, path: str) -> np.ndarray:
        with rasterio.open(path) as src:
            label = src.read(1)
        # 明确二值化：仅 label==1 视为正类，其他全部视为背景
        return (label == 1).astype(np.uint8)

    def _annotate_samples(self, samples: List[Dict], min_fg_pixels: int) -> List[Dict]:
        annotated = []
        for s in samples:
            try:
                label_bin = self._read_binary_label(s['label'])
                fg = int(label_bin.sum())
                total = int(label_bin.size)
                item = dict(s)
                item['fg_count'] = fg
                item['fg_ratio'] = fg / max(total, 1)
                item['is_pos'] = fg >= int(min_fg_pixels)
                annotated.append(item)
            except Exception as e:
                logger.warning(f"Skip bad sample {s.get('label')}: {e}")
        return annotated

    def _compute_stats(self):
        total_fg = 0
        total_pixels = 0
        for idx in self.indices[:min(100, len(self.indices))]:
            try:
                label = self._read_binary_label(self.samples[int(idx)]['label'])
                total_fg += int(label.sum())
                total_pixels += int(label.size)
            except Exception:
                pass
        self.neg_pos_ratio = (total_pixels - total_fg) / max(total_fg, 1)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[int(self.indices[idx])]

        with rasterio.open(s['raw']) as src:
            if any(b > src.count or b < 1 for b in self.bands):
                raise ValueError(f"Requested bands {self.bands} exceed file channel count={src.count}: {s['raw']}")
            image = src.read(self.bands).astype(np.float32)

        label = self._read_binary_label(s['label'])
        image = self._normalize(image)

        if self.mode == 'train':
            image, label = self._augment(image, label)

        return torch.from_numpy(image).float(), torch.from_numpy(label.astype(np.int64))

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

        # 火点专属增强（仅对正样本patch）
        if has_fire:
            if np.random.rand() < 0.3:
                img, lbl = self._fire_copy_paste(img, lbl)
            if np.random.rand() < 0.4:
                fire_mask = lbl > 0
                intensity = 1.05 + 0.30 * np.random.rand()
                for c in range(min(3, C)):
                    img_c = img[c].copy()
                    img_c[fire_mask] = np.clip(img_c[fire_mask] * intensity, 0, 1)
                    img[c] = img_c

        # 通用增强
        if np.random.rand() < 0.3:
            img = img * (0.9 + 0.2 * np.random.rand())
        if np.random.rand() < 0.3:
            img = img + np.random.normal(0, 0.01, img.shape)

        return np.clip(img, 0, 1), (lbl > 0).astype(np.uint8)

    def _fire_copy_paste(self, img: np.ndarray, lbl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        C, H, W = img.shape
        fire_mask = lbl > 0
        if not fire_mask.any():
            return img, lbl

        result_img, result_lbl = img.copy(), lbl.copy()
        for _ in range(np.random.randint(1, 3)):
            y_offset = np.random.randint(-H // 4, H // 4)
            x_offset = np.random.randint(-W // 4, W // 4)
            M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
            shifted_mask = cv2.warpAffine(fire_mask.astype(np.uint8), M, (W, H)) > 0

            overlap = (result_lbl > 0) & shifted_mask
            if shifted_mask.sum() > 0 and overlap.sum() < shifted_mask.sum() * 0.3:
                for c in range(C):
                    shifted_band = cv2.warpAffine(img[c], M, (W, H))
                    result_img[c] = np.where(shifted_mask, shifted_band, result_img[c])
                result_lbl = np.maximum(result_lbl, shifted_mask.astype(np.uint8))

        return result_img, result_lbl


# ============================================================================
# 训练函数
# ============================================================================


def train_epoch(model, loader, criterion, optimizer, device, scaler, use_amp, max_grad_norm, ema=None):
    model.train()
    total_loss = 0.0
    tp = fp = fn = 0

    for i, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        labels_bin = (labels == 1).float()

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            outputs = model(images)
            logits = outputs.squeeze(1)
            loss = criterion(logits, labels_bin)

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

        if ema is not None:
            ema.update()

        total_loss += float(loss.item())

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            labels_eval = labels_bin.long()
            tp += ((preds == 1) & (labels_eval == 1)).sum().item()
            fp += ((preds == 1) & (labels_eval == 0)).sum().item()
            fn += ((preds == 0) & (labels_eval == 1)).sum().item()

        if i % 20 == 0:
            p = tp / (tp + fp + 1e-8) * 100
            r = tp / (tp + fn + 1e-8) * 100
            f1 = 2 * p * r / (p + r + 1e-8)
            logger.info(f'  [{i}/{len(loader)}] Loss:{loss.item():.4f} P:{p:.1f}% R:{r:.1f}% F1:{f1:.1f}%')

    avg_loss = total_loss / max(len(loader), 1)
    precision = tp / (tp + fp + 1e-8) * 100
    recall = tp / (tp + fn + 1e-8) * 100
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    logger.info(f'Train - Loss:{avg_loss:.4f} P:{precision:.2f}% R:{recall:.2f}% F1:{f1:.2f}%')
    return avg_loss, f1


@torch.no_grad()
def validate(model, loader, criterion, device, ema=None, threshold_candidates=None):
    if ema is not None:
        ema.apply_shadow()

    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        labels_bin = (labels == 1).float()

        outputs = model(images)
        logits = outputs.squeeze(1)
        loss = criterion(logits, labels_bin)
        total_loss += float(loss.item())

        probs = torch.sigmoid(logits)
        all_probs.append(probs.detach().cpu())
        all_targets.append(labels_bin.detach().cpu())

    if len(all_probs) == 0:
        if ema is not None:
            ema.restore()
        logger.warning('Val loader is empty, returning zero metrics')
        return 0.0, 0.0, 0.5

    all_probs = torch.cat([p.flatten() for p in all_probs])
    all_targets = torch.cat([t.flatten() for t in all_targets]).long()

    if threshold_candidates is None:
        threshold_candidates = [
            0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05,
            0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
            0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90
        ]

    pos_pixels = int((all_targets == 1).sum().item())
    total_pixels = int(all_targets.numel())
    logger.info(
        'Val prob stats: mean=%.4f p95=%.4f p99=%.4f max=%.4f, pos_pixels=%d/%d (%.6f%%)' % (
            float(all_probs.mean().item()),
            float(torch.quantile(all_probs, 0.95).item()),
            float(torch.quantile(all_probs, 0.99).item()),
            float(all_probs.max().item()),
            pos_pixels, total_pixels, 100.0 * pos_pixels / max(total_pixels, 1)
        )
    )

    best_f1 = 0.0
    best_thresh = 0.5
    best_p = 0.0
    best_r = 0.0

    for thresh in threshold_candidates:
        preds = (all_probs > thresh).long()
        tp = ((preds == 1) & (all_targets == 1)).sum().item()
        fp = ((preds == 1) & (all_targets == 0)).sum().item()
        fn = ((preds == 0) & (all_targets == 1)).sum().item()

        p = tp / (tp + fp + 1e-8) * 100
        r = tp / (tp + fn + 1e-8) * 100
        f1 = 2 * p * r / (p + r + 1e-8)

        if f1 > best_f1:
            best_f1, best_thresh, best_p, best_r = f1, float(thresh), p, r

    if ema is not None:
        ema.restore()

    avg_loss = total_loss / max(len(loader), 1)
    logger.info(f'Val - Loss:{avg_loss:.4f} Best@{best_thresh:.3f} P:{best_p:.2f}% R:{best_r:.2f}% F1:{best_f1:.2f}%')
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



def _set_backbone_trainable(model: nn.Module, trainable: bool):
    for p in model.backbone.parameters():
        p.requires_grad = trainable


def _get_group_lr(optimizer, group_name: str, default_idx: int = 0) -> float:
    for i, g in enumerate(optimizer.param_groups):
        if g.get('name') == group_name:
            return float(g['lr'])
    return float(optimizer.param_groups[default_idx]['lr'])


def main():
    parser = argparse.ArgumentParser(description='Fire Detection Training OpenAI (improved for severe imbalance)')

    parser.add_argument('regions', type=str, help='Region(s): Asia1 | Asia1,Asia2 | ALL')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--tensorboard-dir', type=str, default=DEFAULT_TENSORBOARD_DIR)

    # 数据
    parser.add_argument('--bands', type=int, nargs='+', default=[5, 6, 7], help='Bands to use (1-7)')
    parser.add_argument('--min-fg-pixels', type=int, default=5, help='Patch is positive if fg pixels >= this value')
    parser.add_argument('--train-neg-ratio', type=float, default=2.0,
                        help='Train negative patches kept per positive patch (e.g., 2.0). <0 means keep all negatives')
    parser.add_argument('--keep-all-neg-in-val', dest='keep_all_neg_in_val', action='store_true')
    parser.add_argument('--no-keep-all-neg-in-val', dest='keep_all_neg_in_val', action='store_false')
    parser.set_defaults(keep_all_neg_in_val=True)

    # 模型
    parser.add_argument('--model', type=str, default='mamba_vision_S',
                        choices=['mamba_vision_T', 'mamba_vision_S', 'mamba_vision_B', 'mamba_vision_L'])
    parser.add_argument('--decoder', type=str, default='deeplabv3plus', choices=['deeplabv3plus', 'fpn'])
    parser.add_argument('--pretrained-path', type=str, default=None)
    parser.add_argument('--pretrained', dest='pretrained', action='store_true')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    parser.set_defaults(pretrained=True)

    # 损失
    parser.add_argument('--pos-weight', type=float, default=-1.0,
                        help='<=0 auto from pixel imbalance; >0 use manual value')
    parser.add_argument('--bce-weight', type=float, default=1.0)
    parser.add_argument('--dice-weight', type=float, default=0.5)
    parser.add_argument('--focal-weight', type=float, default=1.0)
    parser.add_argument('--tversky-weight', type=float, default=0.5)
    parser.add_argument('--tversky-alpha', type=float, default=0.3)

    # 训练
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone-lr-scale', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--freeze-backbone-epochs', type=int, default=3,
                        help='First N epochs train decoder only (stabilize under severe imbalance)')
    parser.add_argument('--drop-last', action='store_true', default=True,
                        help='Drop last train batch for BN stability (default True)')

    # 调度器
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'plateau'])
    parser.add_argument('--plateau-factor', type=float, default=0.5)
    parser.add_argument('--plateau-patience', type=int, default=4)

    # EMA/AMP/TB
    parser.add_argument('--use-ema', dest='use_ema', action='store_true')
    parser.add_argument('--no-ema', dest='use_ema', action='store_false')
    parser.set_defaults(use_ema=True)
    parser.add_argument('--ema-decay', type=float, default=0.999)

    parser.add_argument('--use-amp', dest='use_amp', action='store_true')
    parser.add_argument('--no-amp', dest='use_amp', action='store_false')
    parser.set_defaults(use_amp=True)

    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true')
    parser.add_argument('--no-tensorboard', dest='tensorboard', action='store_false')
    parser.set_defaults(tensorboard=True)

    # 早停
    parser.add_argument('--early-stop-patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    regions = parse_regions_arg(args.regions, args.data_dir)
    if len(regions) == 0:
        raise ValueError('No valid regions found')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.output_dir is None:
        region_name = regions[0] if len(regions) == 1 else f'multi{len(regions)}'
        args.output_dir = os.path.join(DEFAULT_OUTPUT_DIR, f'{region_name}_openai_{args.decoder}')
    os.makedirs(args.output_dir, exist_ok=True)

    # 训练前自动提交（可失败，不影响训练）
    git_auto_commit(f"Start training(openai): {regions}, bands={args.bands}, decoder={args.decoder}")

    writer = None
    if args.tensorboard:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"{timestamp}_fire_trainingLandsat_openai"
        tb_dir = os.path.join(args.tensorboard_dir, exp_name)
        writer = SummaryWriter(tb_dir)
        logger.info(f'TensorBoard: {tb_dir}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    logger.info(f'Model: {args.model} + {args.decoder}')
    logger.info(f'Bands: {args.bands}')
    logger.info(f'Regions: {regions}')
    logger.info(f'train_neg_ratio={args.train_neg_ratio}, min_fg_pixels={args.min_fg_pixels}')

    # 数据集
    train_datasets, val_datasets, neg_pos_ratios = [], [], []
    for region in regions:
        try:
            train_ds = FireDataset(
                args.data_dir, region, args.bands, 'train', seed=args.seed,
                min_fg_pixels=args.min_fg_pixels,
                train_neg_ratio=None if args.train_neg_ratio < 0 else args.train_neg_ratio,
                keep_all_neg_in_val=args.keep_all_neg_in_val,
            )
            val_ds = FireDataset(
                args.data_dir, region, args.bands, 'val', seed=args.seed,
                min_fg_pixels=args.min_fg_pixels,
                train_neg_ratio=None if args.train_neg_ratio < 0 else args.train_neg_ratio,
                keep_all_neg_in_val=args.keep_all_neg_in_val,
            )
            if len(train_ds) == 0 or len(val_ds) == 0:
                logger.warning(f'Skip region {region}: empty train/val after filtering')
                continue
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            neg_pos_ratios.append(train_ds.neg_pos_ratio)
        except Exception as e:
            logger.warning(f'Skip region {region}: {e}')

    if len(train_datasets) == 0:
        raise ValueError('No valid datasets after preparation')

    train_ds = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_ds = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
    avg_neg_pos = float(np.mean(neg_pos_ratios)) if len(neg_pos_ratios) > 0 else 100.0
    logger.info(f'Total: train={len(train_ds)}, val={len(val_ds)}, avg pixel neg/pos={avg_neg_pos:.0f}:1')

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=args.drop_last,
        persistent_workers=(args.num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )

    pretrained_path = args.pretrained_path or (
        os.path.join(DEFAULT_PRETRAIN_DIR, 'mambavision_small_1k.pth') if args.pretrained else None
    )
    model = FireDetectionModel(args.model, 1, len(args.bands), args.decoder, args.pretrained, pretrained_path).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Params: {total_params/1e6:.2f}M')

    # pos_weight 自动估计（极端不平衡场景建议不要过大）
    if args.pos_weight is None or args.pos_weight <= 0:
        pos_weight = float(np.clip(np.sqrt(max(avg_neg_pos, 1.0)), 5.0, 40.0))
    else:
        pos_weight = float(args.pos_weight)

    criterion = CombinedLoss(
        pos_weight=pos_weight,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight,
        tversky_weight=args.tversky_weight,
        tversky_alpha=args.tversky_alpha,
        focal_alpha_pos=0.85,
        focal_alpha_neg=0.15,
    )
    logger.info(
        f'Loss config: BCE({args.bce_weight}) + Dice({args.dice_weight}) + '
        f'Focal({args.focal_weight}) + Tversky({args.tversky_weight}), pos_weight={pos_weight:.2f}'
    )

    optimizer = AdamW(model.get_param_groups(args.lr, args.backbone_lr_scale), weight_decay=args.weight_decay)
    logger.info(f'Initial LR: decoder={args.lr:.2e}, backbone={args.lr*args.backbone_lr_scale:.2e}')

    scheduler_cosine = None
    scheduler_plateau = None
    if args.scheduler == 'cosine':
        tmax = max(1, args.epochs - args.warmup_epochs)
        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=tmax, eta_min=1e-6)
    else:
        scheduler_plateau = ReduceLROnPlateau(
            optimizer, mode='max', factor=args.plateau_factor, patience=args.plateau_patience, verbose=True
        )

    ema = ModelEMA(model, args.ema_decay) if args.use_ema else None
    scaler = GradScaler() if args.use_amp else None

    best_f1, best_epoch, best_thresh = 0.0, 0, 0.5
    epochs_no_improve = 0

    logger.info('\n' + '=' * 70)
    logger.info('Start Training (OpenAI improved): severe imbalance friendly setup')
    logger.info('=' * 70 + '\n')



    for epoch in range(1, args.epochs + 1):
        # 阶段训练：先训decoder，再联合训练
        if args.freeze_backbone_epochs > 0 and epoch == 1:
            _set_backbone_trainable(model, False)
            logger.info(f'Freeze backbone for first {args.freeze_backbone_epochs} epoch(s)')
        if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs + 1:
            _set_backbone_trainable(model, True)
            logger.info('Unfreeze backbone (joint training starts)')

        # Warmup（仅设置LR，不调用scheduler.step）
        if epoch <= args.warmup_epochs:
            base_lr = args.lr * epoch / max(args.warmup_epochs, 1)
            for pg in optimizer.param_groups:
                if pg.get('name') == 'backbone':
                    pg['lr'] = base_lr * args.backbone_lr_scale
                else:
                    pg['lr'] = base_lr
            logger.info(
                f'\nEpoch {epoch}/{args.epochs} [Warmup] '
                f"decoder_lr={_get_group_lr(optimizer,'decoder',1):.2e}, "
                f"backbone_lr={_get_group_lr(optimizer,'backbone',0):.2e}"
            )
        else:
            logger.info(
                f'\nEpoch {epoch}/{args.epochs} '
                f"decoder_lr={_get_group_lr(optimizer,'decoder',1):.2e}, "
                f"backbone_lr={_get_group_lr(optimizer,'backbone',0):.2e}"
            )



        logger.info('-' * 70)

        train_loss, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            args.use_amp, args.max_grad_norm, ema
        )
        val_loss, val_f1, val_thresh = validate(model, val_loader, criterion, device, ema)

        # 调度器在epoch末更新（避免与warmup冲突）
        if epoch > args.warmup_epochs:
            if scheduler_cosine is not None:
                scheduler_cosine.step()
            if scheduler_plateau is not None:
                scheduler_plateau.step(val_f1)

        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/train_F1@0.5', train_f1, epoch)
            writer.add_scalar('Metrics/val_best_F1', val_f1, epoch)
            writer.add_scalar('Threshold/val_best', val_thresh, epoch)
            writer.add_scalar('LR/decoder', _get_group_lr(optimizer, 'decoder', 1), epoch)
            writer.add_scalar('LR/backbone', _get_group_lr(optimizer, 'backbone', 0), epoch)

        if val_f1 > best_f1:
            best_f1, best_epoch, best_thresh = val_f1, epoch, val_thresh
            epochs_no_improve = 0
            save_dict = {
                'epoch': epoch,
                'model': model.state_dict(),
                'ema': ema.state_dict() if ema else None,
                'f1': val_f1,
                'threshold': val_thresh,
                'args': vars(args),
            }
            torch.save(save_dict, os.path.join(args.output_dir, 'best_model.pth'))
            logger.info(f'✓ Saved best model (F1: {best_f1:.2f}%, thresh: {best_thresh:.3f})')
        else:
            epochs_no_improve += 1
            logger.info(f'No improvement for {epochs_no_improve} epoch(s)')

        if epochs_no_improve >= args.early_stop_patience:
            logger.warning(f'🛑 Early stopping! Best F1: {best_f1:.2f}% at epoch {best_epoch}')
            break

    logger.info('\n' + '=' * 70)
    logger.info(f'🏆 Best F1: {best_f1:.2f}% @ epoch {best_epoch}, threshold: {best_thresh:.3f}')
    logger.info('=' * 70)

    git_auto_commit(f"Complete(openai): F1={best_f1:.2f}% @ epoch {best_epoch}, regions={regions}")
    if writer:
        writer.close()


if __name__ == '__main__':
    main()
