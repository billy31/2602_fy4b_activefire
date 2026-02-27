#!/usr/bin/env python3
"""
train_landsat.py - 火点检测优化版
GitHub: https://github.com/billy31/2602_fy4b_activefire

优化重点：
1. 简化损失函数 - 纯Dice Loss
2. 数据过滤 - min_fg_pixels=50
3. F1早停 - 3epoch无提升即停
4. 冻结backbone训练策略
5. 学习率warmup
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
from torch.optim.lr_scheduler import LambdaLR
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
# 简化版损失函数 - 纯Dice Loss（对不平衡数据最鲁棒）
# ============================================================================

class DiceLoss(nn.Module):
    """Dice Loss - 使用sigmoid，适用于单通道输出"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        # pred: [B, 1, H, W], target: [B, H, W]
        probs = torch.sigmoid(pred).squeeze(1)  # [B, H, W]
        target_fg = (target == 1).float()
        
        intersection = (probs * target_fg).sum()
        union = probs.sum() + target_fg.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class FocalDiceLoss(nn.Module):
    """Focal + Dice 组合 - 适用于单通道输出"""
    def __init__(self, dice_weight=1.0, focal_weight=0.5, gamma=2.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.gamma = gamma
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        # Dice
        dice = self.dice(pred, target)
        
        # Focal (使用BCE)
        bce = F.binary_cross_entropy_with_logits(pred.squeeze(1), target.float(), reduction='none')
        probs = torch.sigmoid(pred).squeeze(1)
        pt = probs * target.float() + (1 - probs) * (1 - target.float())
        focal = ((1 - pt) ** self.gamma * bce).mean()
        
        return self.dice_weight * dice + self.focal_weight * focal


# ============================================================================
# 模型
# ============================================================================

class SimpleDecoder(nn.Module):
    """简化解码器 - 更易训练"""
    def __init__(self, encoder_dim, num_classes):
        super().__init__()
        
        # 渐进上采样
        self.dec1 = nn.Sequential(
            nn.Conv2d(encoder_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.final = nn.Conv2d(64, num_classes, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
    
    def forward(self, x, input_shape):
        x = self.dec1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        x = self.dec2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        x = self.dec3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        x = self.final(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        return x


class FireDetectionModel(nn.Module):
    def __init__(self, model_name='mamba_vision_S', num_classes=2, 
                 input_channels=3, pretrained=True, pretrained_path=None):
        super().__init__()
        
        # 骨干
        self.backbone = create_model(model_name, pretrained=False, num_classes=num_classes)
        
        # 修改输入
        if input_channels != 3:
            self._modify_input(input_channels)
        
        # 加载预训练
        if pretrained and pretrained_path:
            self._load_pretrained(pretrained_path)
        
        # 简化解码器
        dims = {'mamba_vision_T': 640, 'mamba_vision_S': 768, 
                'mamba_vision_B': 1024, 'mamba_vision_L': 1568}
        encoder_dim = dims.get(model_name, 768)
        
        self.decoder = SimpleDecoder(encoder_dim, num_classes)
        self.num_classes = num_classes
    
    def _modify_input(self, input_channels):
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
    
    def _load_pretrained(self, path):
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt.get('model', ckpt))
        state = {k: v for k, v in state.items() if not k.startswith('head.')}
        self.backbone.load_state_dict(state, strict=False)
        logger.info(f"Loaded pretrained: {path}")
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.backbone.patch_embed(x)
        for level in self.backbone.levels:
            x = level(x)
        x = self.backbone.norm(x)
        x = self.decoder(x, (H, W))
        return x
    
    def freeze_backbone(self):
        """冻结backbone，只训练decoder"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen, only training decoder")
    
    def unfreeze_backbone(self):
        """解冻backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")


# ============================================================================
# 数据集 - 提高min_fg_pixels过滤噪声
# ============================================================================

class FireDataset(Dataset):
    def __init__(self, data_dir, region, bands=[7,6,2], mode='train', 
                 split=0.8, seed=42, min_fg_pixels=50):  # 提高到50过滤噪声
        self.raw_dir = os.path.join(data_dir, region, 'raw')
        self.label_dir = os.path.join(data_dir, region, 'mask_label')
        self.bands = bands
        self.mode = mode
        
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
        
        logger.info(f"[{mode}] {len(self.indices)} patches (min_fg={min_fg_pixels})")
    
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
        
        # 检查数据有效性
        if np.all(image == 0) or np.all(label == 0):
            logger.warning(f"Zero data detected in sample {idx}")
        
        # 归一化
        image = image.astype(np.float32)
        for i in range(image.shape[0]):
            b = image[i]
            if b.max() > b.min():
                image[i] = (b - b.min()) / (b.max() - b.min())
            else:
                image[i] = b / (b.max() + 1e-8) if b.max() > 0 else b
        
        if self.mode == 'train':
            image, label = self._augment(image, label)
        
        return torch.from_numpy(image).float(), torch.from_numpy(label.astype(np.int64))
    
    def _augment(self, img, lbl):
        # 随机水平翻转
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=2).copy()
            lbl = np.flip(lbl, axis=1).copy()
        # 随机垂直翻转
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=1).copy()
            lbl = np.flip(lbl, axis=0).copy()
        # 随机旋转90度（新增）
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)  # 旋转90, 180, 或 270度
            img = np.rot90(img, k, axes=(1, 2)).copy()
            lbl = np.rot90(lbl, k).copy()
        return img, lbl


# ============================================================================
# 训练
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device, scaler, use_amp, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    tp = fp = fn = 0
    
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            # 梯度裁剪 - 防止梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            # 梯度裁剪 - 防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()
        
        if i % 10 == 0:
            p = tp / (tp + fp + 1e-8) * 100
            r = tp / (tp + fn + 1e-8) * 100
            f1 = 2 * p * r / (p + r + 1e-8)
            logger.info(f'  [{i}/{len(loader)}] Loss: {loss.item():.4f} P:{p:.1f}% R:{r:.1f}% F1:{f1:.1f}%')
    
    avg_loss = total_loss / len(loader)
    precision = tp / (tp + fp + 1e-8) * 100
    recall = tp / (tp + fn + 1e-8) * 100
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    logger.info(f'Train - Loss: {avg_loss:.4f} P:{precision:.2f}% R:{recall:.2f}% F1:{f1:.2f}%')
    return avg_loss, f1


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    tp = fp = fn = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
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
    
    logger.info(f'Val - Loss: {avg_loss:.4f} F1:{f1:.2f}% IoU:{iou:.2f}% P:{precision:.2f}% R:{recall:.2f}%')
    return avg_loss, f1, iou, precision, recall


# ============================================================================
# Warmup调度器
# ============================================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Warmup阶段线性增加
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine退火
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


# ============================================================================
# 可视化
# ============================================================================

def visualize_predictions(model, loader, device, num_samples=4, save_dir='./visualizations'):
    """可视化预测结果"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    count = 0
    with torch.no_grad():
        for images, labels in loader:
            if count >= num_samples:
                break
            
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)[:, 1, :, :]
            preds = outputs.argmax(dim=1)
            
            # 保存前几个样本
            for i in range(min(2, images.size(0))):
                if count >= num_samples:
                    break
                
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                
                # 输入图像 (RGB合成)
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
                axes[3].set_title('Prediction')
                axes[3].axis('off')
                
                plt.tight_layout()
                plt.savefig(f'{save_dir}/sample_{count}.png', dpi=150)
                plt.close()
                
                count += 1
    
    logger.info(f"Saved {count} visualizations to {save_dir}")


# ============================================================================
# Git自动提交功能
# ============================================================================

def git_commit_auto(message):
    """自动提交代码变更到Git"""
    try:
        import subprocess
        
        # 检查是否有变更
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, cwd='/root/codes/fire0226/selfCodes')
        
        if result.stdout.strip():
            # 有变更，执行提交
            subprocess.run(['git', 'add', '-A'], cwd='/root/codes/fire0226/selfCodes', check=True)
            subprocess.run(['git', 'commit', '-m', message], cwd='/root/codes/fire0226/selfCodes', check=True)
            
            # 尝试推送
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
    parser = argparse.ArgumentParser(description='Fire Detection Training')
    
    # 基本参数
    parser.add_argument('region', type=str, help='Region name (e.g., Asia1)')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--tensorboard-dir', type=str, default=DEFAULT_TENSORBOARD_DIR)
    
    # 数据参数
    parser.add_argument('--bands', type=int, nargs='+', default=[7, 6, 2])
    parser.add_argument('--min-fg-pixels', type=int, default=50, 
                       help='Min fire pixels to filter noise (default: 50)')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='mamba_vision_S')
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--freeze-backbone-epochs', type=int, default=10,
                       help='Freeze backbone for N epochs (default: 10)')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                       help='Max gradient norm for clipping (default: 1.0)')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='Warmup epochs, 5% of total (default: 5)')
    
    # 早停参数 - 基于F1
    parser.add_argument('--early-stop-patience', type=int, default=3,
                       help='Early stop patience based on F1 (default: 3)')
    parser.add_argument('--early-stop-min-f1', type=float, default=40.0,
                       help='Minimum F1 to consider successful (default: 40%)')
    
    # 其他
    parser.add_argument('--use-amp', action='store_true', default=True)
    parser.add_argument('--tensorboard', action='store_true', default=True)
    parser.add_argument('--visualize', action='store_true', default=False)
    
    args = parser.parse_args()
    
    # 训练前自动提交
    git_commit_auto(f"Pre-train: Start training {args.region} with lr={args.lr}, bs={args.batch_size}")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    if args.output_dir is None:
        args.output_dir = os.path.join(DEFAULT_OUTPUT_DIR, args.region)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # TensorBoard
    writer = None
    if args.tensorboard:
        exp_name = f"fire_{args.region}_{datetime.now().strftime('%m%d_%H%M')}"
        tb_dir = os.path.join(args.tensorboard_dir, exp_name)
        writer = SummaryWriter(tb_dir)
        logger.info(f'TensorBoard: {tb_dir}')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    
    # 数据集
    train_ds = FireDataset(args.data_dir, args.region, args.bands, 'train', 
                          min_fg_pixels=args.min_fg_pixels)
    val_ds = FireDataset(args.data_dir, args.region, args.bands, 'val',
                        min_fg_pixels=args.min_fg_pixels)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    
    # 模型
    pretrained_path = os.path.join(DEFAULT_PRETRAIN_DIR, 'mambavision_small_1k.pth') if args.pretrained else None
    model = FireDetectionModel(args.model, 1, len(args.bands), args.pretrained, pretrained_path).to(device)
    logger.info(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    
    # 损失 - 简化版Dice
    criterion = DiceLoss()
    logger.info('Using Dice Loss')
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Warmup + Cosine调度
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs, args.lr)
    scaler = GradScaler() if args.use_amp else None
    
    # 训练状态
    best_f1 = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    
    for epoch in range(1, args.epochs + 1):
        current_lr = scheduler.step(epoch - 1)
        logger.info(f'\nEpoch {epoch}/{args.epochs} (lr={current_lr:.2e})')
        logger.info('-' * 60)
        
        # 阶段性解冻backbone - 更灵活的策略
        if epoch == 1 and args.freeze_backbone_epochs > 0:
            model.freeze_backbone()
        elif epoch == args.freeze_backbone_epochs + 1:
            model.unfreeze_backbone()
            # 解冻后使用较小的学习率
            optimizer = AdamW([
                {'params': model.backbone.parameters(), 'lr': args.lr * 0.1},
                {'params': model.decoder.parameters(), 'lr': args.lr}
            ], weight_decay=args.weight_decay)
            logger.info('Optimizer reinitialized with layer-wise lr')
        
        # 训练
        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device, scaler, args.use_amp, args.max_grad_norm)
        
        # 验证
        val_loss, val_f1, val_iou, val_p, val_r = validate(model, val_loader, criterion, device)
        
        # TensorBoard
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/F1', val_f1, epoch)
            writer.add_scalar('Metrics/IoU', val_iou, epoch)
            writer.add_scalar('Metrics/Precision', val_p, epoch)
            writer.add_scalar('Metrics/Recall', val_r, epoch)
            writer.add_scalar('Train/lr', current_lr, epoch)
        
        # 保存最佳模型（基于F1）
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'f1': val_f1, 'iou': val_iou, 'p': val_p, 'r': val_r,
                'args': vars(args)
            }, os.path.join(args.output_dir, 'best_model.pth'))
            logger.info(f'✓ Saved best model (F1: {best_f1:.2f}%)')
        else:
            epochs_no_improve += 1
            logger.info(f'  No F1 improvement for {epochs_no_improve} epochs')
        
        # 早停 - 3epoch无F1提升即停
        if epochs_no_improve >= args.early_stop_patience:
            logger.warning(f'\n🛑 Early stopping! No F1 improvement for {args.early_stop_patience} epochs')
            logger.warning(f'   Best F1: {best_f1:.2f}% at epoch {best_epoch}')
            
            if best_f1 < args.early_stop_min_f1:
                logger.warning(f'   Warning: Best F1 {best_f1:.2f}% < {args.early_stop_min_f1}% target')
            break
    
    # 可视化
    if args.visualize:
        visualize_predictions(model, val_loader, device)
    
    logger.info(f'\n🏆 Best: F1 {best_f1:.2f}% (P:{val_p:.1f}%, R:{val_r:.1f}%) @ epoch {best_epoch}')
    
    # 训练后自动提交
    git_commit_auto(f"Post-train: {args.region} best F1={best_f1:.2f}% @ epoch {best_epoch}")
    
    if writer:
        writer.close()


if __name__ == '__main__':
    main()
