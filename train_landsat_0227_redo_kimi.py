"""
Update 0227_redo_kimi: 
- 初始版本：针对Landsat-8 (7/6/2波段) 跨卫星迁移学习
- 支持多区域训练 (Asia1, Asia2, All)
- 处理类别不平衡问题（正例数据增强 + Focal Loss + 加权采样）
- 修改MambaVision输入层适配3通道遥感数据
- 集成TensorBoard和自动Git同步
- 优化指标：mIoU, F1-Score, Recall, Precision
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
import rasterio
from rasterio.windows import Window
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from tqdm import tqdm
import git

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
DATA_ROOT = Path("/root/autodl-tmp")
TRAIN_ROOT = DATA_ROOT / "training"
META_ROOT = DATA_ROOT / "meta"
OUTPUT_ROOT = DATA_ROOT / "output"
TENSORBOARD_ROOT = Path("/root/tf-logs")
PRETRAINED_ROOT = Path("/root/autodl-tmp/pretrained")

# Landsat-8 7/6/2波段对应索引 (B7=SWIR2, B6=SWIR1, B2=Blue)
# 注意：GDAL波段索引从1开始
LANDSAT_BANDS = [7, 6, 2]  # 对应文件中的波段位置

# 类别定义（二分类问题）
NUM_CLASSES = 2
CLASS_NAMES = ['Background', 'Target']

# 设备检测
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[Hardware] GPU: {GPU_NAME}, Memory: {GPU_MEMORY:.2f} GB")
else:
    DEVICE = torch.device("cpu")
    print("[Hardware] Warning: No GPU detected, using CPU")

# ==================== Git同步工具 ====================
def sync_to_git(comment: str):
    """自动同步代码到Git仓库"""
    try:
        repo = git.Repo(search_parent_directories=True)
        repo.git.add('--all')
        repo.index.commit(f"[{datetime.datetime.now().strftime('%m%d')}] {comment}")
        origin = repo.remote(name='origin')
        origin.push()
        print(f"[Git] Successfully synced: {comment}")
    except Exception as e:
        print(f"[Git] Sync warning: {e}")

# ==================== 数据增强（针对正例增强） ====================
class RemoteSensingAugmentation:
    """针对遥感数据的增强策略，重点增强正例样本"""
    
    def __init__(self, is_training=True, positive_enhance=False):
        self.is_training = is_training
        self.positive_enhance = positive_enhance  # 是否进行正例增强
        
    def __call__(self, image, mask):
        if not self.is_training:
            return image, mask
            
        # 基础增强
        # 随机水平翻转
        if random.random() > 0.5:
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()
            
        # 随机垂直翻转
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()
            
        # 随机旋转 (90, 180, 270)
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            image = np.rot90(image, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k).copy()
        
        # 正例增强策略
        if self.positive_enhance:
            # 亮度调整（只增加亮度，模拟不同光照条件）
            if random.random() > 0.5:
                factor = random.uniform(1.0, 1.3)
                image = np.clip(image * factor, 0, 1)
                
            # 对比度调整
            if random.random() > 0.5:
                mean = image.mean()
                factor = random.uniform(0.9, 1.2)
                image = np.clip((image - mean) * factor + mean, 0, 1)
                
            # 添加噪声（模拟传感器噪声）
            if random.random() > 0.7:
                noise = np.random.normal(0, 0.01, image.shape)
                image = np.clip(image + noise, 0, 1)
                
        return image, mask

# ==================== 数据集类 ====================
class LandsatDataset(Dataset):
    """Landsat-8数据集加载器"""
    
    def __init__(self, 
                 regions: List[str],
                 patch_size: int = 256,
                 stride: int = 128,
                 is_training: bool = True,
                 positive_ratio: float = 0.5):  # 正例采样比例
        self.regions = regions
        self.patch_size = patch_size
        self.stride = stride
        self.is_training = is_training
        self.positive_ratio = positive_ratio
        
        self.samples = []
        self.augment = RemoteSensingAugmentation(
            is_training=is_training, 
            positive_enhance=True
        )
        
        self._load_data()
        
    def _load_data(self):
        """加载数据路径并构建样本列表"""
        print(f"[Data] Loading regions: {self.regions}")
        
        for region in self.regions:
            region_path = TRAIN_ROOT / region
            raw_dir = region_path / "raw"
            mask_dir = region_path / "mask_label"
            
            if not raw_dir.exists():
                print(f"[Warning] Region {region} raw dir not found: {raw_dir}")
                continue
                
            # 获取所有原始影像
            raw_files = list(raw_dir.glob("*.tif"))
            print(f"[Data] {region}: Found {len(raw_files)} raw images")
            
            for raw_file in raw_files:
                # 构建对应的mask文件名
                # 训练：LC08_L1TP_xxx_voting_p00141.tif
                # 标签：LC08_L1TP_xxx_p00141.tif
                mask_name = raw_file.name.replace("_voting_", "_")
                mask_file = mask_dir / mask_name
                
                if mask_file.exists():
                    # 解析影像获取有效区域
                    self._parse_image_patches(raw_file, mask_file, region)
                else:
                    print(f"[Warning] Mask not found: {mask_file}")
        
        print(f"[Data] Total samples loaded: {len(self.samples)}")
        
        # 分析正负样本比例
        if self.samples:
            pos_count = sum([s['has_positive'] for s in self.samples])
            print(f"[Data] Positive samples: {pos_count}/{len(self.samples)} ({pos_count/len(self.samples)*100:.2f}%)")
    
    def _parse_image_patches(self, raw_file: Path, mask_file: Path, region: str):
        """解析影像并生成patch位置"""
        try:
            with rasterio.open(raw_file) as src:
                height, width = src.height, src.width
                
            # 计算patch位置
            for y in range(0, height - self.patch_size + 1, self.stride):
                for x in range(0, width - self.patch_size + 1, self.stride):
                    self.samples.append({
                        'raw': raw_file,
                        'mask': mask_file,
                        'region': region,
                        'x': x,
                        'y': y,
                        'has_positive': None  # 延迟加载
                    })
        except Exception as e:
            print(f"[Error] Failed to parse {raw_file}: {e}")
    
    def _check_positive(self, idx: int) -> bool:
        """检查patch是否包含正例"""
        sample = self.samples[idx]
        if sample['has_positive'] is not None:
            return sample['has_positive']
            
        try:
            with rasterio.open(sample['mask']) as src:
                window = Window(sample['x'], sample['y'], self.patch_size, self.patch_size)
                mask = src.read(1, window=window)
                is_positive = (mask > 0).sum() > 0
                self.samples[idx]['has_positive'] = is_positive
                return is_positive
        except:
            return False
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 读取影像数据
        with rasterio.open(sample['raw']) as src:
            window = Window(sample['x'], sample['y'], self.patch_size, self.patch_size)
            # 读取指定波段 (7,6,2 -> 索引6,5,1)
            bands = []
            for b in LANDSAT_BANDS:
                band_data = src.read(b, window=window)
                bands.append(band_data)
            image = np.stack(bands, axis=0).astype(np.float32)
            
            # 归一化到0-1
            # Landsat-8反射率范围通常是0-10000或0-1
            if image.max() > 1.0:
                image = image / 10000.0
            image = np.clip(image, 0, 1)
        
        # 读取标签
        with rasterio.open(sample['mask']) as src:
            window = Window(sample['x'], sample['y'], self.patch_size, self.patch_size)
            mask = src.read(1, window=window)
            # 二值化
            mask = (mask > 0).astype(np.int64)
        
        # 数据增强
        image, mask = self.augment(image, mask)
        
        # 转换为Tensor
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        
        return {
            'image': image,
            'mask': mask,
            'region': sample['region'],
            'has_positive': (mask > 0).any().item()
        }

# ==================== 类别不平衡采样器 ====================
def create_weighted_sampler(dataset: LandsatDataset, positive_weight: float = 3.0):
    """创建加权采样器以平衡正负样本"""
    print("[Sampler] Calculating sample weights...")
    weights = []
    
    for i in tqdm(range(len(dataset)), desc="Analyzing class distribution"):
        is_pos = dataset._check_positive(i)
        if is_pos:
            weights.append(positive_weight)
        else:
            weights.append(1.0)
    
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler

# ==================== Focal Loss（处理类别不平衡） ====================
class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ==================== MambaVision修改版（适配3通道输入） ====================
class PatchEmbed(nn.Module):
    """修改的Patch Embedding层，支持任意输入通道数"""
    
    def __init__(self, in_channels=3, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = self.proj(x)  # B, C, H, W -> B, embed_dim, H//4, W//4
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, H*W, C
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

class MambaVisionEncoder(nn.Module):
    """简化的MambaVision编码器（适配遥感数据）"""
    
    def __init__(self, in_channels=3, embed_dims=[96, 192, 384, 768], num_heads=[3, 6, 12, 24]):
        super().__init__()
        
        # Stage 1: Patch Embedding + CNN Block
        self.patch_embed = PatchEmbed(in_channels, embed_dims[0], patch_size=4)
        self.stage1 = nn.Sequential(
            nn.Conv2d(embed_dims[0], embed_dims[0], 3, padding=1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU(),
            nn.Conv2d(embed_dims[0], embed_dims[0], 3, padding=1),
            nn.BatchNorm2d(embed_dims[0]),
        )
        self.down1 = nn.Conv2d(embed_dims[0], embed_dims[1], 3, stride=2, padding=1)
        
        # Stage 2: CNN Block
        self.stage2 = nn.Sequential(
            nn.Conv2d(embed_dims[1], embed_dims[1], 3, padding=1),
            nn.BatchNorm2d(embed_dims[1]),
            nn.GELU(),
            nn.Conv2d(embed_dims[1], embed_dims[1], 3, padding=1),
            nn.BatchNorm2d(embed_dims[1]),
        )
        self.down2 = nn.Conv2d(embed_dims[1], embed_dims[2], 3, stride=2, padding=1)
        
        # Stage 3: Mamba-like Block (简化版)
        self.stage3 = nn.Sequential(
            nn.Conv2d(embed_dims[2], embed_dims[2], 3, padding=1, groups=embed_dims[2]),
            nn.BatchNorm2d(embed_dims[2]),
            nn.GELU(),
            nn.Conv2d(embed_dims[2], embed_dims[2], 1),
            nn.BatchNorm2d(embed_dims[2]),
        )
        self.down3 = nn.Conv2d(embed_dims[2], embed_dims[3], 3, stride=2, padding=1)
        
        # Stage 4: Transformer-like Block (简化版)
        self.stage4 = nn.TransformerEncoderLayer(
            d_model=embed_dims[3], 
            nhead=num_heads[3],
            dim_feedforward=embed_dims[3]*4,
            batch_first=True
        )
        
    def forward(self, x):
        # Stage 1
        x = self.patch_embed(x)
        x = x + self.stage1(x)
        x1 = x
        
        # Stage 2
        x = self.down1(x)
        x = x + self.stage2(x)
        x2 = x
        
        # Stage 3
        x = self.down2(x)
        x = x + self.stage3(x)
        x3 = x
        
        # Stage 4
        x = self.down3(x)
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # B, H*W, C
        x = self.stage4(x)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x4 = x
        
        return [x1, x2, x3, x4]

class MambaVisionSegmentor(nn.Module):
    """MambaVision分割网络（U-Net风格解码器）"""
    
    def __init__(self, in_channels=3, num_classes=2, embed_dims=[96, 192, 384, 768]):
        super().__init__()
        
        self.encoder = MambaVisionEncoder(in_channels, embed_dims)
        
        # 解码器
        self.up3 = nn.ConvTranspose2d(embed_dims[3], embed_dims[2], 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(embed_dims[3], embed_dims[2], 3, padding=1),
            nn.BatchNorm2d(embed_dims[2]),
            nn.GELU(),
        )
        
        self.up2 = nn.ConvTranspose2d(embed_dims[2], embed_dims[1], 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(embed_dims[2], embed_dims[1], 3, padding=1),
            nn.BatchNorm2d(embed_dims[1]),
            nn.GELU(),
        )
        
        self.up1 = nn.ConvTranspose2d(embed_dims[1], embed_dims[0], 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(embed_dims[1], embed_dims[0], 3, padding=1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU(),
        )
        
        # 最终上采样到原始分辨率
        self.final_up = nn.ConvTranspose2d(embed_dims[0], embed_dims[0], 4, stride=4)
        self.seg_head = nn.Conv2d(embed_dims[0], num_classes, 1)
        
    def forward(self, x):
        # 编码器
        skips = self.encoder(x)
        x1, x2, x3, x4 = skips
        
        # 解码器（带跳跃连接）
        d3 = self.up3(x4)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)
        
        # 上采样到原始分辨率
        out = self.final_up(d1)
        out = self.seg_head(out)
        
        return out

# ==================== 指标计算 ====================
class MetricsCalculator:
    """计算分割指标：mIoU, F1-Score, Recall, Precision"""
    
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.total_pixels = 0
        self.class_correct = [0] * self.num_classes
        self.class_total = [0] * self.num_classes
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.all_preds = []
        self.all_targets = []
    
    def update(self, preds: np.ndarray, targets: np.ndarray):
        """更新指标"""
        preds_flat = preds.flatten()
        targets_flat = targets.flatten()
        
        # 混淆矩阵
        for t, p in zip(targets_flat, preds_flat):
            self.confusion_matrix[t, p] += 1
            
        self.all_preds.extend(preds_flat)
        self.all_targets.extend(targets_flat)
    
    def compute(self) -> Dict[str, float]:
        """计算最终指标"""
        # IoU per class
        ious = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            iou = tp / (tp + fp + fn + 1e-8)
            ious.append(iou)
        
        miou = np.mean(ious)
        
        # F1, Recall, Precision (针对正类，即class=1)
        if len(self.all_preds) > 0:
            precision = precision_score(self.all_targets, self.all_preds, pos_label=1, zero_division=0)
            recall = recall_score(self.all_targets, self.all_preds, pos_label=1, zero_division=0)
            f1 = f1_score(self.all_targets, self.all_preds, pos_label=1, zero_division=0)
        else:
            precision = recall = f1 = 0.0
        
        return {
            'mIoU': miou,
            'F1-Score': f1,
            'Recall': recall,
            'Precision': precision,
            'IoU_Background': ious[0],
            'IoU_Target': ious[1]
        }

# ==================== 训练引擎 ====================
class Trainer:
    """训练管理器"""
    
    def __init__(self, args):
        self.args = args
        self.device = DEVICE
        
        # 创建输出目录
        self.exp_name = f"landsat_mamba_{datetime.datetime.now().strftime('%m%d_%H%M')}"
        self.output_dir = OUTPUT_ROOT / self.exp_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(TENSORBOARD_ROOT / self.exp_name)
        
        # 构建模型
        self.model = MambaVisionSegmentor(
            in_channels=3,
            num_classes=NUM_CLASSES,
            embed_dims=[96, 192, 384, 768]
        ).to(self.device)
        
        print(f"[Model] Parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")
        
        # 损失函数（类别不平衡处理）
        # 计算类别权重
        class_weights = torch.tensor([1.0, args.pos_weight]).to(self.device)
        if args.use_focal:
            self.criterion = FocalLoss(alpha=0.25, gamma=2.0, weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # 学习率调度
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # 指标计算器
        self.train_metrics = MetricsCalculator(NUM_CLASSES)
        self.val_metrics = MetricsCalculator(NUM_CLASSES)
        
        # 最佳模型跟踪
        self.best_miou = 0.0
        self.best_f1 = 0.0
        
    def train_epoch(self, epoch: int, dataloader: DataLoader):
        """训练一个epoch"""
        self.model.train()
        self.train_metrics.reset()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 更新指标
            epoch_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            targets = masks.cpu().numpy()
            self.train_metrics.update(preds, targets)
            
            # 更新进度条
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # 记录到TensorBoard（每10步）
            global_step = epoch * len(dataloader) + batch_idx
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Train/Loss_step', loss.item(), global_step)
        
        # 计算epoch指标
        metrics = self.train_metrics.compute()
        metrics['loss'] = epoch_loss / len(dataloader)
        
        return metrics
    
    @torch.no_grad()
    def validate(self, epoch: int, dataloader: DataLoader):
        """验证"""
        self.model.eval()
        self.val_metrics.reset()
        epoch_loss = 0.0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch} [Val]"):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            epoch_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            targets = masks.cpu().numpy()
            self.val_metrics.update(preds, targets)
        
        metrics = self.val_metrics.compute()
        metrics['loss'] = epoch_loss / len(dataloader)
        
        return metrics
    
    def log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """记录指标到TensorBoard和终端"""
        # TensorBoard
        for split, metrics in [('Train', train_metrics), ('Val', val_metrics)]:
            for key, value in metrics.items():
                self.writer.add_scalar(f'{split}/{key}', value, epoch)
        
        # 终端输出
        print(f"\\n[Epoch {epoch}] Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        print(f"[Epoch {epoch}] Val mIoU: {val_metrics['mIoU']:.4f}, F1: {val_metrics['F1-Score']:.4f}")
        print(f"[Epoch {epoch}] Val Recall: {val_metrics['Recall']:.4f}, Precision: {val_metrics['Precision']:.4f}")
        print(f"[Epoch {epoch}] IoU - Background: {val_metrics['IoU_Background']:.4f}, Target: {val_metrics['IoU_Target']:.4f}")
    
    def save_checkpoint(self, epoch: int, val_metrics: Dict, is_best: bool = False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_metrics': val_metrics,
            'args': vars(self.args)
        }
        
        # 保存最新模型
        torch.save(checkpoint, self.output_dir / 'last.pth')
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best.pth')
            print(f"[Checkpoint] Saved best model (mIoU: {val_metrics['mIoU']:.4f}, F1: {val_metrics['F1-Score']:.4f})")
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        """完整训练流程"""
        print(f"[Training] Starting training for {self.args.epochs} epochs")
        
        for epoch in range(1, self.args.epochs + 1):
            # 训练
            train_metrics = self.train_epoch(epoch, train_loader)
            
            # 验证
            val_metrics = self.validate(epoch, val_loader)
            
            # 学习率调整
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/LR', current_lr, epoch)
            
            # 记录指标
            self.log_metrics(epoch, train_metrics, val_metrics)
            
            # 保存最佳模型（优先看mIoU，其次F1）
            is_best = False
            if val_metrics['mIoU'] > self.best_miou:
                self.best_miou = val_metrics['mIoU']
                is_best = True
            elif abs(val_metrics['mIoU'] - self.best_miou) < 0.001 and val_metrics['F1-Score'] > self.best_f1:
                self.best_f1 = val_metrics['F1-Score']
                is_best = True
            
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # 早停检查（如果F1连续5个epoch低于0.01，可能是训练失败）
            if epoch > 5 and val_metrics['F1-Score'] < 0.01:
                print(f"[Warning] F1-Score too low ({val_metrics['F1-Score']:.4f}), possible training failure")
        
        print(f"[Training] Finished! Best mIoU: {self.best_miou:.4f}")
        self.writer.close()

# ==================== 主函数 ====================
def parse_args():
    parser = argparse.ArgumentParser(description='Landsat-8 MambaVision Training')
    
    # 数据参数
    parser.add_argument('--regions', type=str, default='Asia1',
                       help='训练区域，支持: Asia1, Asia2, Asia1,Asia2, All')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=128)
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # 类别不平衡处理
    parser.add_argument('--pos_weight', type=float, default=10.0,
                       help='正类权重')
    parser.add_argument('--use_focal', action='store_true', default=True,
                       help='使用Focal Loss')
    parser.add_argument('--pos_enhance', action='store_true', default=True,
                       help='正例数据增强')
    parser.add_argument('--sample_pos_ratio', type=float, default=0.5,
                       help='采样中正例的目标比例')
    
    # 其他
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    
    return parser.parse_args()

def get_regions(region_str: str) -> List[str]:
    """解析区域参数"""
    available_regions = ['Asia1', 'Asia2']  # 可扩展
    
    if region_str == 'All':
        return available_regions
    else:
        regions = [r.strip() for r in region_str.split(',')]
        # 验证区域存在性
        for r in regions:
            if r not in available_regions:
                print(f"[Warning] Unknown region: {r}, available: {available_regions}")
        return [r for r in regions if r in available_regions]

def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 解析区域
    regions = get_regions(args.regions)
    if not regions:
        print("[Error] No valid regions specified!")
        return
    print(f"[Config] Training regions: {regions}")
    
    # 创建数据集
    print("[Data] Loading training dataset...")
    full_dataset = LandsatDataset(
        regions=regions,
        patch_size=args.patch_size,
        stride=args.stride,
        is_training=True,
        positive_ratio=args.sample_pos_ratio
    )
    
    if len(full_dataset) == 0:
        print("[Error] No data loaded!")
        return
    
    # 划分训练集和验证集
    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"[Data] Train: {train_size}, Val: {val_size}")
    
    # 创建采样器（处理类别不平衡）
    train_sampler = create_weighted_sampler(full_dataset, positive_weight=args.pos_weight)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建训练器
    trainer = Trainer(args)
    
    # 开始训练
    trainer.fit(train_loader, val_loader)
    
    # Git同步
    sync_to_git(f"Training completed: regions={args.regions}, best_mIoU={trainer.best_miou:.4f}")

if __name__ == '__main__':
    main()
