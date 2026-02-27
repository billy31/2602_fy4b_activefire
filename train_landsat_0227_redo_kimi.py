#!/usr/env/bin python3
"""
Update 0227_redo_kimi_v4: 
- 修正TypeError: 处理is_positive为None的情况
- 优化正负样本统计逻辑
- 保持其他功能不变
"""

import os
import sys
import argparse
import subprocess
import datetime
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
import numpy as np
import json
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
import rasterio
from rasterio.windows import Window
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, confusion_matrix
from tqdm import tqdm
import git

warnings.filterwarnings('ignore')

# ==================== 硬件环境检测 ====================
def check_hardware():
    """检测硬件环境"""
    print("=" * 60)
    print("Hardware Environment Check")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"✓ CUDA Available: {torch.cuda.is_available()}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ PyTorch Version: {torch.__version__}")
        print(f"✓ GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  - GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
        print(f"✓ cuDNN Version: {torch.backends.cudnn.version()}")
    else:
        print("✗ CUDA not available, using CPU")

    try:
        import mamba_ssm
        print(f"✓ mamba-ssm installed: {mamba_ssm.__version__}")
    except ImportError:
        print("✗ mamba-ssm not installed (using fallback implementation)")

    deps = ['rasterio', 'gitpython', 'tqdm', 'sklearn', 'tensorboard']
    for dep in deps:
        try:
            __import__(dep)
            print(f"✓ {dep} available")
        except ImportError:
            print(f"✗ {dep} missing")

    print("=" * 60)

# ==================== 配置参数 ====================
DATA_ROOT = Path("/root/autodl-tmp")
TRAIN_ROOT = DATA_ROOT / "training"
OUTPUT_ROOT = DATA_ROOT / "output"
TENSORBOARD_ROOT = Path("/root/tf-logs")
PRETRAINED_ROOT = Path("/root/autodl-tmp/pretrained")

# Landsat-8 7/6/2波段 (SWIR2/SWIR1/Blue)
LANDSAT_BANDS = [7, 6, 2]
NUM_CLASSES = 2
CLASS_NAMES = ['Background', 'Target']

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==================== Git同步 ====================
def sync_to_git(comment: str):
    """自动同步代码到Git仓库"""
    try:
        repo = git.Repo(search_parent_directories=True)
        repo.git.add('--all')
        commit_msg = f"[{datetime.datetime.now().strftime('%m%d')}] {comment}"
        repo.index.commit(commit_msg)
        origin = repo.remote(name='origin')
        origin.push()
        print(f"[Git] Synced: {commit_msg}")
    except Exception as e:
        print(f"[Git] Warning: {e}")

# ==================== 数据增强 ====================
class StrongAugmentation:
    """强数据增强，特别针对正例样本"""

    def __init__(self, is_training=True):
        self.is_training = is_training

    def __call__(self, image, mask):
        if not self.is_training:
            return image, mask

        # 随机翻转
        if random.random() > 0.5:
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()

        # 随机旋转
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            image = np.rot90(image, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k).copy()

        # 随机缩放
        if random.random() > 0.7:
            scale = random.uniform(0.9, 1.1)
            h, w = image.shape[1], image.shape[2]
            new_h, new_w = int(h * scale), int(w * scale)

            image = torch.from_numpy(image).unsqueeze(0)
            image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)

            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
            mask = F.interpolate(mask, size=(new_h, new_w), mode='nearest')

            if new_h > h:
                start_y = (new_h - h) // 2
                image = image[:, :, start_y:start_y+h, start_y:start_y+w]
                mask = mask[:, :, start_y:start_y+h, start_y:start_y+w]
            elif new_h < h:
                pad_y, pad_x = (h - new_h) // 2, (w - new_w) // 2
                image = F.pad(image, (pad_x, pad_x, pad_y, pad_y))
                mask = F.pad(mask, (pad_x, pad_x, pad_y, pad_y))

            image = image.squeeze(0).numpy()
            mask = mask.squeeze(0).squeeze(0).numpy().astype(np.int64)

        # 颜色抖动
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 1)

        if random.random() > 0.5:
            mean = image.mean()
            factor = random.uniform(0.8, 1.2)
            image = np.clip((image - mean) * factor + mean, 0, 1)

        # 高斯噪声
        if random.random() > 0.7:
            noise = np.random.normal(0, 0.02, image.shape)
            image = np.clip(image + noise, 0, 1)

        return image, mask

# ==================== 数据集 ====================
class LandsatDataset(Dataset):
    """Landsat-8数据集 - 修正版"""

    def __init__(self, regions, patch_size=256, stride=128, is_training=True):
        self.regions = regions if isinstance(regions, list) else [regions]
        self.patch_size = patch_size
        self.stride = stride
        self.is_training = is_training
        self.augment = StrongAugmentation(is_training)

        self.samples = []
        self._load_data()

    def _get_mask_name(self, raw_name: str) -> str:
        """
        根据raw文件名生成对应的mask文件名
        raw: LC08_L1TP_170024_20200812_20200812_01_RT_p00602.tif
        mask: LC08_L1TP_170024_20200812_20200812_01_RT_voting_p00602.tif
        """
        # 在_p之前插入_voting
        pattern = r'(.+)(_p\d+\.tif)$'
        match = re.match(pattern, raw_name)
        if match:
            prefix = match.group(1)
            suffix = match.group(2)
            return f"{prefix}_voting{suffix}"
        else:
            # 如果匹配失败，尝试在.tif前插入_voting
            return raw_name.replace('.tif', '_voting.tif')

    def _load_data(self):
        print(f"[Data] Loading regions: {self.regions}")

        for region in self.regions:
            region_path = TRAIN_ROOT / region
            raw_dir = region_path / "raw"
            mask_dir = region_path / "mask_label"

            if not raw_dir.exists():
                print(f"[Warning] Raw directory not found: {raw_dir}")
                continue

            if not mask_dir.exists():
                print(f"[Warning] Mask directory not found: {mask_dir}")
                continue

            raw_files = list(raw_dir.glob("*.tif"))
            print(f"[Data] {region}: Found {len(raw_files)} raw images")

            matched_count = 0
            for raw_file in raw_files:
                # 生成对应的mask文件名
                mask_name = self._get_mask_name(raw_file.name)
                mask_file = mask_dir / mask_name

                if mask_file.exists():
                    self._extract_patches(raw_file, mask_file, region)
                    matched_count += 1
                else:
                    print(f"[Warning] Mask not found for {raw_file.name}")
                    print(f"         Expected: {mask_name}")

            print(f"[Data] {region}: {matched_count}/{len(raw_files)} images matched with masks")

        print(f"[Data] Total patches: {len(self.samples)}")

        # 统计正负样本 - 修正：避免None值
        if self.samples:
            # 延迟加载时不计算，只统计已知的
            known_positives = [s.get('is_positive') for s in self.samples if s.get('is_positive') is not None]
            pos_count = sum(known_positives) if known_positives else 0
            known_count = len(known_positives)

            if known_count > 0:
                print(f"[Data] Known positive patches: {pos_count}/{known_count} ({pos_count/max(known_count,1)*100:.2f}%)")
            else:
                print(f"[Data] Positive patches: not yet calculated (lazy loading)")

    def _extract_patches(self, raw_file: Path, mask_file: Path, region: str):
        """从影像中提取patch位置"""
        try:
            with rasterio.open(raw_file) as src:
                h, w = src.height, src.width

                # 检查波段数
                if src.count < max(LANDSAT_BANDS):
                    print(f"[Warning] {raw_file.name} has only {src.count} bands, expected at least {max(LANDSAT_BANDS)}")
                    return

            # 生成patch坐标
            for y in range(0, h - self.patch_size + 1, self.stride):
                for x in range(0, w - self.patch_size + 1, self.stride):
                    self.samples.append({
                        'raw': raw_file,
                        'mask': mask_file,
                        'region': region,
                        'x': x,
                        'y': y,
                        'width': self.patch_size,
                        'height': self.patch_size,
                        'is_positive': False  # 默认False，避免None
                    })
        except Exception as e:
            print(f"[Error] Failed to process {raw_file}: {e}")

    def check_positive(self, idx: int) -> bool:
        """检查patch是否包含正例"""
        sample = self.samples[idx]
        if sample.get('is_positive') is not None and sample['is_positive'] is not False:
            return sample['is_positive']

        try:
            with rasterio.open(sample['mask']) as src:
                window = Window(sample['x'], sample['y'], self.patch_size, self.patch_size)
                mask_data = src.read(1, window=window)
                is_pos = bool((mask_data > 0).any())  # 确保是bool类型
                sample['is_positive'] = is_pos
                return is_pos
        except Exception as e:
            print(f"[Error] Failed to read mask {sample['mask']}: {e}")
            sample['is_positive'] = False
            return False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 读取影像数据 (raw文件)
        try:
            with rasterio.open(sample['raw']) as src:
                window = Window(sample['x'], sample['y'], self.patch_size, self.patch_size)

                # 读取指定波段 (7,6,2)
                bands = []
                for b in LANDSAT_BANDS:
                    try:
                        band_data = src.read(b, window=window)
                        bands.append(band_data)
                    except IndexError:
                        print(f"[Error] Band {b} not found in {sample['raw']}")
                        raise

                image = np.stack(bands, axis=0).astype(np.float32)

                # 归一化
                if image.max() > 1.0:
                    image = image / 10000.0
                image = np.clip(image, 0, 1)
        except Exception as e:
            print(f"[Error] Failed to read raw image {sample['raw']}: {e}")
            # 返回空数据
            image = np.zeros((3, self.patch_size, self.patch_size), dtype=np.float32)

        # 读取标签数据 (mask_label文件，带voting)
        try:
            with rasterio.open(sample['mask']) as src:
                window = Window(sample['x'], sample['y'], self.patch_size, self.patch_size)
                mask = src.read(1, window=window)
                # 二值化：大于0为正类
                mask = (mask > 0).astype(np.int64)
        except Exception as e:
            print(f"[Error] Failed to read mask {sample['mask']}: {e}")
            mask = np.zeros((self.patch_size, self.patch_size), dtype=np.int64)

        # 数据增强
        image, mask = self.augment(image, mask)

        return {
            'image': torch.from_numpy(image),
            'mask': torch.from_numpy(mask),
            'is_positive': bool((mask > 0).any()),  # 确保是bool
            'region': sample['region']
        }

# ==================== OHEM Loss ====================
class OHEMFocalLoss(nn.Module):
    """在线困难样本挖掘 + Focal Loss"""

    def __init__(self, alpha=0.25, gamma=2.0, weight=None, ohem_ratio=0.7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ohem_ratio = ohem_ratio

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.training:
            num_hard = int(self.ohem_ratio * focal_loss.numel())
            hard_loss, _ = torch.topk(focal_loss.view(-1), num_hard)
            return hard_loss.mean()
        else:
            return focal_loss.mean()

# ==================== MambaVision模型 ====================
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class MambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv1 = ConvBNAct(dim, dim//2, 1, padding=0)
        self.dwconv = ConvBNAct(dim//2, dim//2, 3, groups=dim//2)
        self.conv2 = nn.Conv2d(dim//2, dim, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.conv2(x)

        return residual + x

class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x

        x = x.permute(0, 2, 3, 1).view(B, H*W, C)
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        return residual + x

class MambaVisionBackbone(nn.Module):
    def __init__(self, in_ch=3, embed_dims=[64, 128, 256, 512]):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNAct(in_ch, embed_dims[0]//2, 3, stride=2, padding=1),
            ConvBNAct(embed_dims[0]//2, embed_dims[0], 3, stride=2, padding=1),
        )

        self.stage1 = nn.Sequential(
            ConvBNAct(embed_dims[0], embed_dims[0]),
            ConvBNAct(embed_dims[0], embed_dims[0]),
        )
        self.down1 = ConvBNAct(embed_dims[0], embed_dims[1], 3, stride=2)

        self.stage2 = nn.Sequential(
            ConvBNAct(embed_dims[1], embed_dims[1]),
            ConvBNAct(embed_dims[1], embed_dims[1]),
        )
        self.down2 = ConvBNAct(embed_dims[1], embed_dims[2], 3, stride=2)

        self.stage3_mamba = nn.Sequential(*[MambaBlock(embed_dims[2]) for _ in range(2)])
        self.stage3_attn = AttentionBlock(embed_dims[2])
        self.down3 = ConvBNAct(embed_dims[2], embed_dims[3], 3, stride=2)

        self.stage4_mamba = nn.Sequential(*[MambaBlock(embed_dims[3]) for _ in range(2)])
        self.stage4_attn = AttentionBlock(embed_dims[3])

    def forward(self, x):
        x = self.stem(x)

        x1 = self.stage1(x)
        x = self.down1(x1)

        x2 = self.stage2(x)
        x = self.down2(x2)

        x3 = self.stage3_mamba(x) + self.stage3_attn(x)
        x = self.down3(x3)

        x4 = self.stage4_mamba(x) + self.stage4_attn(x)

        return [x1, x2, x3, x4]

class SegmentationHead(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()

        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_ch, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.project = nn.Sequential(
            nn.Conv2d(256*4, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.project(x)
        return x

class MambaVisionSegmentor(nn.Module):
    def __init__(self, in_ch=3, num_classes=2, embed_dims=[64, 128, 256, 512]):
        super().__init__()

        self.backbone = MambaVisionBackbone(in_ch, embed_dims)

        self.fpn4 = nn.Conv2d(embed_dims[3], 256, 1)
        self.fpn3 = nn.Conv2d(embed_dims[2], 256, 1)
        self.fpn2 = nn.Conv2d(embed_dims[1], 256, 1)
        self.fpn1 = nn.Conv2d(embed_dims[0], 256, 1)

        self.up3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(256, 256, 2, stride=2)

        self.fuse3 = ConvBNAct(256, 256, 3)
        self.fuse2 = ConvBNAct(256, 256, 3)
        self.fuse1 = ConvBNAct(256, 256, 3)

        self.final_up = nn.ConvTranspose2d(256, 256, 4, stride=4)
        self.seg_head = SegmentationHead(256, num_classes)

    def forward(self, x):
        c1, c2, c3, c4 = self.backbone(x)

        p4 = self.fpn4(c4)

        p3 = self.up3(p4) + self.fpn3(c3)
        p3 = self.fuse3(p3)

        p2 = self.up2(p3) + self.fpn2(c2)
        p2 = self.fuse2(p2)

        p1 = self.up1(p2) + self.fpn1(c1)
        p1 = self.fuse1(p1)

        p0 = self.final_up(p1)
        out = self.seg_head(p0)

        return out

# ==================== 指标计算 ====================
class Metrics:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.conf_mat = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred, target):
        pred = pred.flatten()
        target = target.flatten()

        for p, t in zip(pred, target):
            self.conf_mat[t, p] += 1

    def compute(self):
        ious = []
        for i in range(self.num_classes):
            tp = self.conf_mat[i, i]
            fp = self.conf_mat[:, i].sum() - tp
            fn = self.conf_mat[i, :].sum() - tp
            iou = tp / (tp + fp + fn + 1e-8)
            ious.append(iou)

        miou = np.mean(ious)

        tp = self.conf_mat[1, 1]
        fp = self.conf_mat[0, 1]
        fn = self.conf_mat[1, 0]

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            'mIoU': miou,
            'F1-Score': f1,
            'Recall': recall,
            'Precision': precision,
            'IoU_BG': ious[0],
            'IoU_FG': ious[1]
        }

# ==================== 训练器 ====================
class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = DEVICE

        self.exp_name = f"landsat_mamba_{datetime.datetime.now().strftime('%m%d_%H%M')}"
        self.output_dir = OUTPUT_ROOT / self.exp_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2)

        self.writer = SummaryWriter(TENSORBOARD_ROOT / self.exp_name)

        self.model = MambaVisionSegmentor(
            in_ch=3,
            num_classes=NUM_CLASSES,
            embed_dims=[64, 128, 256, 512]
        ).to(self.device)

        print(f"[Model] Params: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")

        class_weights = torch.tensor([1.0, args.pos_weight]).to(self.device)
        self.criterion = OHEMFocalLoss(
            alpha=0.25, 
            gamma=2.0, 
            weight=class_weights,
            ohem_ratio=args.ohem_ratio
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        self.warmup_epochs = args.warmup_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs - self.warmup_epochs
        )

        self.best_miou = 0.0
        self.best_f1 = 0.0

    def warmup_lr(self, epoch, batch_idx, num_batches):
        if epoch < self.warmup_epochs:
            warmup_factor = (epoch * num_batches + batch_idx) / (self.warmup_epochs * num_batches)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.args.lr * warmup_factor

    def train_epoch(self, epoch, dataloader):
        self.model.train()
        metrics = Metrics(NUM_CLASSES)
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            if epoch < self.warmup_epochs:
                self.warmup_lr(epoch, batch_idx, len(dataloader))

            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            targets = masks.cpu().numpy()
            metrics.update(preds, targets)

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            step = epoch * len(dataloader) + batch_idx
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), step)

        if epoch >= self.warmup_epochs:
            self.scheduler.step()

        result = metrics.compute()
        result['loss'] = total_loss / len(dataloader)
        return result

    @torch.no_grad()
    def validate(self, epoch, dataloader):
        self.model.eval()
        metrics = Metrics(NUM_CLASSES)
        total_loss = 0.0

        for batch in tqdm(dataloader, desc="Validate"):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            targets = masks.cpu().numpy()
            metrics.update(preds, targets)

        result = metrics.compute()
        result['loss'] = total_loss / len(dataloader)
        return result

    def log(self, epoch, train_m, val_m):
        for split, m in [('Train', train_m), ('Val', val_m)]:
            for k, v in m.items():
                self.writer.add_scalar(f'{split}/{k}', v, epoch)

        print(f"\n[Epoch {epoch}] Train Loss: {train_m['loss']:.4f}, Val Loss: {val_m['loss']:.4f}")
        print(f"[Epoch {epoch}] Val mIoU: {val_m['mIoU']:.4f}, F1: {val_m['F1-Score']:.4f}")
        print(f"[Epoch {epoch}] Recall: {val_m['Recall']:.4f}, Precision: {val_m['Precision']:.4f}")
        print(f"[Epoch {epoch}] IoU BG: {val_m['IoU_BG']:.4f}, FG: {val_m['IoU_FG']:.4f}")

    def save(self, epoch, val_m, is_best=False):
        ckpt = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': val_m,
            'args': vars(self.args)
        }

        torch.save(ckpt, self.output_dir / 'last.pth')
        if is_best:
            torch.save(ckpt, self.output_dir / 'best.pth')
            print(f"[Save] Best model saved (mIoU: {val_m['mIoU']:.4f})")

    def fit(self, train_loader, val_loader):
        print(f"[Training] {self.args.epochs} epochs")

        for epoch in range(1, self.args.epochs + 1):
            train_m = self.train_epoch(epoch, train_loader)
            val_m = self.validate(epoch, val_loader)

            self.log(epoch, train_m, val_m)

            is_best = val_m['mIoU'] > self.best_miou
            if is_best:
                self.best_miou = val_m['mIoU']
                self.best_f1 = val_m['F1-Score']

            self.save(epoch, val_m, is_best)

            if epoch > 10 and val_m['F1-Score'] < 0.01:
                print("[Warning] Training seems failed (F1 too low)")

        print(f"[Done] Best mIoU: {self.best_miou:.4f}, F1: {self.best_f1:.4f}")
        self.writer.close()

# ==================== 主函数 ====================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--regions', type=str, default='Asia1',
                       help='训练区域：Asia1, Asia2, Asia1,Asia2, All')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=128)

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)

    parser.add_argument('--pos_weight', type=float, default=10.0,
                       help='正类权重，用于处理类别不平衡')
    parser.add_argument('--ohem_ratio', type=float, default=0.7,
                       help='OHEM保留的困难样本比例')

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_ratio', type=float, default=0.2)

    return parser.parse_args()

def get_regions(s):
    """解析区域参数"""
    available = ['Asia1', 'Asia2']
    if s == 'All':
        return available
    regions = [r.strip() for r in s.split(',')]
    valid_regions = [r for r in regions if r in available]
    if not valid_regions:
        print(f"[Warning] No valid regions found in '{s}', using Asia1")
        return ['Asia1']
    return valid_regions

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    check_hardware()

    args = parse_args()
    set_seed(args.seed)

    regions = get_regions(args.regions)
    print(f"[Config] Training regions: {regions}")
    print(f"[Config] Data root: {TRAIN_ROOT}")

    # 验证路径存在性
    for region in regions:
        raw_dir = TRAIN_ROOT / region / "raw"
        mask_dir = TRAIN_ROOT / region / "mask_label"
        print(f"[Check] {region}/raw exists: {raw_dir.exists()}")
        print(f"[Check] {region}/mask_label exists: {mask_dir.exists()}")

    # 创建数据集
    dataset = LandsatDataset(regions, args.patch_size, args.stride, True)

    if len(dataset) == 0:
        print("[Error] No data loaded! Please check data paths.")
        return

    # 划分训练/验证集
    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    print(f"[Data] Train: {train_size}, Val: {val_size}")

    # 创建加权采样器
    print("[Sampler] Calculating sample weights...")
    weights = []
    for i in tqdm(range(len(dataset)), desc="Checking labels"):
        is_pos = dataset.check_positive(i)
        weights.append(args.pos_weight if is_pos else 1.0)

    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 训练
    trainer = Trainer(args)
    trainer.fit(train_loader, val_loader)

    # Git同步
    sync_to_git(f"Train complete: regions={args.regions}, mIoU={trainer.best_miou:.4f}")

if __name__ == '__main__':
    main()