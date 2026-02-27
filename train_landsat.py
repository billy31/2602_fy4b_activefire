#!/usr/bin/env python3
"""
train_landsat.py - DeepLabV3+ for Fire Detection
GitHub: https://github.com/billy31/2602_fy4b_activefire
"""

import os
import sys
import argparse
import logging
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, '/root/codes/fire0226/MambaVision')
from mambavision import create_model

import rasterio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = '/root/autodl-tmp/training'
DEFAULT_PRETRAIN_DIR = '/root/autodl-tmp/pretrained'
DEFAULT_OUTPUT_DIR = '/root/autodl-tmp/training/output'
DEFAULT_TENSORBOARD_DIR = '/root/tf-logs'


# ============================================================================
# ASPP Module (DeepLabV3+)
# ============================================================================

class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        size = x.shape[-2:]
        x = self.gap(x)
        x = self.relu(self.bn(self.conv(x)))
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, dilations=[6, 12, 18]):
        super().__init__()
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_3x3s = nn.ModuleList([ASPPConv(in_channels, out_channels, d) for d in dilations])
        self.pooling = ASPPPooling(in_channels, out_channels)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (2 + len(dilations)), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        outputs = [self.conv_1x1(x)]
        outputs.extend([conv(x) for conv in self.conv_3x3s])
        outputs.append(self.pooling(x))
        x = torch.cat(outputs, dim=1)
        return self.project(x)


# ============================================================================
# DeepLabV3+ Decoder
# ============================================================================

class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, encoder_dim, num_classes, low_level_dim=48):
        super().__init__()
        self.aspp = ASPP(encoder_dim, 256, dilations=[6, 12, 18])
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(encoder_dim, low_level_dim, 1, bias=False),
            nn.BatchNorm2d(low_level_dim),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(256 + low_level_dim, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x_encoder, input_shape):
        x = self.aspp(x_encoder)
        x = F.interpolate(x, size=(x_encoder.size(2), x_encoder.size(3)), mode='bilinear', align_corners=False)
        low_level = self.low_level_conv(x_encoder)
        x = torch.cat([x, low_level], dim=1)
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x


# ============================================================================
# Model
# ============================================================================

class MambaVisionDeepLab(nn.Module):
    def __init__(self, model_name='mamba_vision_S', num_classes=2, input_channels=3, pretrained=True, pretrained_path=None):
        super().__init__()
        self.backbone = create_model(model_name, pretrained=False, num_classes=num_classes)
        if input_channels != 3:
            self._modify_input(input_channels)
        if pretrained and pretrained_path:
            self._load_pretrained(pretrained_path)
        dims = {'mamba_vision_T': 640, 'mamba_vision_S': 768, 'mamba_vision_B': 1024, 'mamba_vision_L': 1568}
        encoder_dim = dims.get(model_name, 768)
        self.decoder = DeepLabV3PlusDecoder(encoder_dim, num_classes)
        self.num_classes = num_classes
    
    def _modify_input(self, input_channels):
        if hasattr(self.backbone, 'patch_embed') and hasattr(self.backbone.patch_embed, 'conv_down'):
            conv = self.backbone.patch_embed.conv_down[0]
            new_conv = nn.Conv2d(input_channels, conv.out_channels, kernel_size=conv.kernel_size, 
                               stride=conv.stride, padding=conv.padding, bias=conv.bias is not None)
            with torch.no_grad():
                n = conv.weight.size(1)
                repeat = (input_channels + n - 1) // n
                w = conv.weight.repeat(1, repeat, 1, 1)
                new_conv.weight.copy_(w[:, :input_channels, :, :])
                if conv.bias is not None:
                    new_conv.bias.copy_(conv.bias)
            self.backbone.patch_embed.conv_down[0] = new_conv
    
    def _load_pretrained(self, path):
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt.get('model', ckpt))
        state = {k: v for k, v in state.items() if not k.startswith('head.')}
        self.backbone.load_state_dict(state, strict=False)
        logger.info(f"Loaded pretrained from {path}")
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.backbone.patch_embed(x)
        for level in self.backbone.levels:
            x = level(x)
        x = self.backbone.norm(x)
        x = self.decoder(x, (H, W))
        return x


# ============================================================================
# Loss
# ============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        probs = F.softmax(pred, dim=1)[:, 1, :, :]
        target = (target == 1).float()
        intersection = (probs * target).sum()
        union = probs.sum() + target.sum()
        return 1 - (2. * intersection + self.smooth) / (union + self.smooth)


class TverskyLoss(nn.Module):
    """
    Tversky Loss - 可调整假阴性(FN)和假阳性(FP)的权重
    alpha: FN权重 (漏检)
    beta: FP权重 (误检)
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        probs = F.softmax(pred, dim=1)[:, 1, :, :]
        targets_fg = (target == 1).float()
        
        tp = (probs * targets_fg).sum()
        fp = (probs * (1 - targets_fg)).sum()
        fn = ((1 - probs) * targets_fg).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1 - tversky


class FocalLoss(nn.Module):
    """
    Focal Loss - 处理类别不平衡，让模型关注难分类样本
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class OHEMLoss(nn.Module):
    """
    Online Hard Example Mining Loss
    只使用最困难的样本进行反向传播
    """
    def __init__(self, ratio=0.7):
        super().__init__()
        self.ratio = ratio
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        # 选择损失最高的样本
        num_hard = int(ce_loss.numel() * self.ratio)
        hard_loss, _ = torch.topk(ce_loss.view(-1), num_hard)
        return hard_loss.mean()


class RecallFocusedLoss(nn.Module):
    """
    针对低Recall优化的损失函数
    使用高alpha惩罚假阴性（漏检）
    """
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.alpha = alpha  # 高alpha关注假阴性
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        probs = F.softmax(pred, dim=1)[:, 1, :, :]
        targets_fg = (target == 1).float()
        
        tp = (probs * targets_fg).sum()
        fp = (probs * (1 - targets_fg)).sum()
        fn = ((1 - probs) * targets_fg).sum()
        
        # 高alpha更惩罚fn（漏检）
        recall_loss = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1 - recall_loss


class FireLoss(nn.Module):
    """
    火点专用组合损失 - 平衡Precision和Recall
    
    针对当前问题：Recall高(91%)但Precision极低(0.69%)
    策略：
    - 基础CE：稳定梯度
    - Dice Loss：直接优化IoU
    - Tversky Loss：可调整FP/FN平衡
    - 动态权重：根据训练阶段调整
    """
    def __init__(self, fg_weight=500.0, tversky_beta=0.3, 
                 use_recall_focus=False, recall_alpha=0.7):
        super().__init__()
        self.fg_weight = fg_weight
        self.use_recall_focus = use_recall_focus
        self.tversky_beta = tversky_beta
        
        self.dice = DiceLoss()
        # tversky_beta控制FP惩罚力度
        # beta=0.3：更关注Recall（当前需要）
        # beta=0.5：平衡
        # beta=0.7：更关注Precision
        self.tversky = TverskyLoss(alpha=1.0-tversky_beta, beta=tversky_beta)
        
        if use_recall_focus:
            self.recall_loss = RecallFocusedLoss(alpha=recall_alpha, beta=0.3)
    
    def forward(self, pred, target, epoch=0):
        # 带权重的CE
        weight = torch.tensor([1.0, self.fg_weight], device=pred.device)
        ce = F.cross_entropy(pred, target, weight=weight)
        
        # Dice和Tversky
        dice = self.dice(pred, target)
        tversky = self.tversky(pred, target)
        
        # 基础组合
        total = ce + dice + 2.0 * tversky
        
        # 可选：添加Recall-focused loss
        if self.use_recall_focus:
            recall = self.recall_loss(pred, target)
            total = total + recall
        
        return total


# ============================================================================
# Dataset
# ============================================================================

class FireDataset(Dataset):
    def __init__(self, data_dir, region, bands=[7,6,2], mode='train', split=0.8, seed=42, min_fg_pixels=5):
        self.raw_dir = os.path.join(data_dir, region, 'raw')
        self.label_dir = os.path.join(data_dir, region, 'mask_label')
        self.bands = bands
        self.mode = mode
        
        samples = self._scan_samples()
        self.samples = self._filter_fire(samples, min_fg_pixels)
        
        np.random.seed(seed)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(len(indices) * split)
        
        if mode == 'train':
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
        
        logger.info(f"[{mode}] {len(self.indices)} fire patches")
    
    def _scan_samples(self):
        samples = []
        for f in os.listdir(self.label_dir):
            if '_voting_' in f and f.endswith('.tif'):
                raw_f = f.replace('_voting_', '_').replace('.tif', '.tif')
                raw_path = os.path.join(self.raw_dir, raw_f)
                label_path = os.path.join(self.label_dir, f)
                if os.path.exists(raw_path):
                    samples.append({'raw': raw_path, 'label': label_path})
        return samples
    
    def _filter_fire(self, samples, min_fg):
        filtered = []
        for s in samples:
            try:
                with rasterio.open(s['label']) as src:
                    label = src.read(1)
                fg = (label == 1).sum()
                if fg >= min_fg:
                    s['fg_count'] = int(fg)
                    s['fg_ratio'] = fg / label.size
                    filtered.append(s)
            except:
                pass
        logger.info(f"Filtered: {len(filtered)}/{len(samples)} have >= {min_fg} fire pixels")
        return filtered
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        s = self.samples[self.indices[idx]]
        
        with rasterio.open(s['raw']) as src:
            bands = self.bands if max(self.bands) <= src.count else list(range(1, src.count+1))
            image = src.read(bands)
        
        with rasterio.open(s['label']) as src:
            label = src.read(1)
        
        image = image.astype(np.float32)
        for i in range(image.shape[0]):
            b = image[i]
            if b.max() > b.min():
                image[i] = (b - b.min()) / (b.max() - b.min())
        
        if self.mode == 'train':
            image, label = self._augment(image, label)
        
        return torch.from_numpy(image).float(), torch.from_numpy(label.astype(np.int64))
    
    def _augment(self, img, lbl):
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=2).copy()
            lbl = np.flip(lbl, axis=1).copy()
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=1).copy()
            lbl = np.flip(lbl, axis=0).copy()
        return img, lbl


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device, scaler, use_amp, epoch=0):
    model.train()
    total_loss = 0
    tp = fp = fn = 0
    
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels, epoch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels, epoch)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()
        
        if i % 20 == 0:
            p = tp / (tp + fp + 1e-8) * 100
            r = tp / (tp + fn + 1e-8) * 100
            logger.info(f'  Batch {i}/{len(loader)} Loss: {loss.item():.4f} P:{p:.1f}% R:{r:.1f}%')
    
    avg_loss = total_loss / len(loader)
    precision = tp / (tp + fp + 1e-8) * 100
    recall = tp / (tp + fn + 1e-8) * 100
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8) * 100
    logger.info(f'Train - Loss: {avg_loss:.4f} IoU:{iou:.2f}% P:{precision:.2f}% R:{recall:.2f}% F1:{f1:.2f}%')
    return avg_loss, iou, precision, recall, f1


@torch.no_grad()
def validate(model, loader, criterion, device, epoch=0):
    model.eval()
    total_loss = 0
    tp = fp = fn = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels, epoch)
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()
    
    avg_loss = total_loss / len(loader)
    precision = tp / (tp + fp + 1e-8) * 100
    recall = tp / (tp + fn + 1e-8) * 100
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8) * 100
    logger.info(f'Val - Loss: {avg_loss:.4f} IoU:{iou:.2f}% P:{precision:.2f}% R:{recall:.2f}% F1:{f1:.2f}%')
    return avg_loss, iou, precision, recall, f1


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('region', type=str)
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--tensorboard-dir', type=str, default=DEFAULT_TENSORBOARD_DIR)
    parser.add_argument('--bands', type=int, nargs='+', default=[7, 6, 2])
    parser.add_argument('--model', type=str, default='mamba_vision_S')
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--fg-weight', type=float, default=500.0, 
                       help='Foreground class weight for CE loss (default: 500)')
    parser.add_argument('--tversky-beta', type=float, default=0.3,
                       help='Tversky beta (FP penalty), lower=focus on recall (default: 0.3)')
    parser.add_argument('--use-amp', action='store_true', default=True)
    parser.add_argument('--tensorboard', action='store_true', default=True)
    
    # 早停参数
    parser.add_argument('--early-stop-patience', type=int, default=20,
                       help='Early stopping patience epochs (default: 20)')
    parser.add_argument('--early-stop-min-delta', type=float, default=0.5,
                       help='Minimum improvement for early stopping (default: 0.5%)')
    parser.add_argument('--early-stop-min-epochs', type=int, default=30,
                       help='Minimum epochs before early stopping (default: 30)')
    
    args = parser.parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    if args.output_dir is None:
        args.output_dir = os.path.join(DEFAULT_OUTPUT_DIR, args.region)
    os.makedirs(args.output_dir, exist_ok=True)
    
    writer = SummaryWriter(os.path.join(args.tensorboard_dir, f"fire_{datetime.now().strftime('%Y%m%d_%H%M%S')}")) if args.tensorboard else None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    
    train_ds = FireDataset(args.data_dir, args.region, args.bands, 'train')
    val_ds = FireDataset(args.data_dir, args.region, args.bands, 'val')
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    pretrained_path = os.path.join(DEFAULT_PRETRAIN_DIR, 'mambavision_small_1k.pth') if args.pretrained else None
    model = MambaVisionDeepLab(args.model, 2, len(args.bands), args.pretrained, pretrained_path).to(device)
    logger.info(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    
    criterion = FireLoss(fg_weight=args.fg_weight, tversky_beta=args.tversky_beta)
    logger.info(f'Using FireLoss: fg_weight={args.fg_weight}, tversky_beta={args.tversky_beta}')
    
    backbone_params, decoder_params = [], []
    for name, p in model.named_parameters():
        (backbone_params if 'backbone' in name else decoder_params).append(p)
    
    optimizer = AdamW([{'params': backbone_params, 'lr': args.lr * 0.1}, {'params': decoder_params, 'lr': args.lr}], weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler() if args.use_amp else None
    
    # 早停设置
    best_iou = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    early_stop_triggered = False
    
    logger.info(f'Early stopping: patience={args.early_stop_patience}, '
                f'min_delta={args.early_stop_min_delta}%, min_epochs={args.early_stop_min_epochs}')
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f'\nEpoch {epoch}/{args.epochs}')
        logger.info('-' * 60)
        
        # 训练
        train_loss, train_iou, train_p, train_r, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, args.use_amp, epoch)
        
        # 验证
        val_loss, val_iou, val_p, val_r, val_f1 = validate(
            model, val_loader, criterion, device, epoch)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # TensorBoard记录 - 训练指标
        if writer:
            # 损失
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            
            # 验证指标
            writer.add_scalar('Metrics/Val_IoU', val_iou, epoch)
            writer.add_scalar('Metrics/Val_Precision', val_p, epoch)
            writer.add_scalar('Metrics/Val_Recall', val_r, epoch)
            writer.add_scalar('Metrics/Val_F1', val_f1, epoch)
            
            # 训练指标
            writer.add_scalar('Metrics/Train_IoU', train_iou, epoch)
            writer.add_scalar('Metrics/Train_Precision', train_p, epoch)
            writer.add_scalar('Metrics/Train_Recall', train_r, epoch)
            writer.add_scalar('Metrics/Train_F1', train_f1, epoch)
            
            # 学习率
            writer.add_scalar('Train/lr', current_lr, epoch)
            
            # 记录Precision-Recall平衡
            writer.add_scalar('Balance/Val_P-R-Diff', abs(val_p - val_r), epoch)
            writer.add_scalar('Balance/Train_P-R-Diff', abs(train_p - train_r), epoch)
        
        # 保存最佳模型
        if val_iou > best_iou + args.early_stop_min_delta:
            best_iou = val_iou
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'iou': val_iou, 
                       'precision': val_p, 'recall': val_r, 'args': vars(args)}, 
                      os.path.join(args.output_dir, 'best_model.pth'))
            logger.info(f'✓ Saved best model (IoU: {best_iou:.2f}%, P:{val_p:.2f}%, R:{val_r:.2f}%)')
        else:
            epochs_no_improve += 1
            logger.info(f'  No improvement for {epochs_no_improve} epochs (best: {best_iou:.2f}% @ epoch {best_epoch})')
        
        # 早停检查
        if epoch >= args.early_stop_min_epochs and epochs_no_improve >= args.early_stop_patience:
            logger.warning(f'\n🛑 Early stopping triggered!')
            logger.warning(f'   No improvement for {epochs_no_improve} epochs')
            logger.warning(f'   Best IoU: {best_iou:.2f}% at epoch {best_epoch}')
            early_stop_triggered = True
            break
    
    if not early_stop_triggered:
        logger.info(f'\n✅ Training completed (all {args.epochs} epochs)')
    
    logger.info(f'🏆 Best result: IoU {best_iou:.2f}% (Precision {val_p:.2f}%, Recall {val_r:.2f}%) at epoch {best_epoch}')
    
    if writer:
        writer.close()


if __name__ == '__main__':
    main()
