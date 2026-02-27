#!/usr/bin/env python3
"""
深度诊断脚本 - 检查训练问题的根本原因
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import rasterio

# 设置路径
sys.path.insert(0, '/root/codes/fire0226/MambaVision')
sys.path.insert(0, '/root/codes/fire0226/selfCodes')

from train_landsat import FireDataset, FireDetectionModel

# 辅助函数
def create_model(num_classes=1, input_channels=3, pretrained=False):
    """创建模型"""
    model = FireDetectionModel(
        model_name='mamba_vision_S',
        num_classes=num_classes,
        input_channels=input_channels,
        pretrained=False
    )
    return model

# 配置
DEFAULT_DATA_DIR = '/root/autodl-tmp/training'
REGION = 'Asia1'
BANDS = [7, 6, 2]
MIN_FG_PIXELS = 50

def check_raw_data():
    """1. 检查原始数据"""
    print("\n" + "="*60)
    print("1. 检查原始数据质量")
    print("="*60)
    
    region_dir = os.path.join(DEFAULT_DATA_DIR, REGION)
    raw_dir = os.path.join(region_dir, 'raw')
    mask_dir = os.path.join(region_dir, 'mask_label')
    
    # 找几个样本
    import glob
    mask_files = sorted(glob.glob(os.path.join(mask_dir, '*_voting_*.tif')))[:3]
    
    for mask_path in mask_files:
        # 正确的文件名转换: _voting_ -> _
        mask_name = os.path.basename(mask_path)
        raw_name = mask_name.replace('_voting_', '_')
        raw_path = os.path.join(raw_dir, raw_name)
        
        print(f"\n标签: {mask_name}")
        print(f"图像: {raw_name}")
        
        # 读取标签
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            print(f"  标签 - shape: {mask.shape}, dtype: {mask.dtype}")
            print(f"         unique values: {np.unique(mask)}")
            print(f"         sum: {mask.sum()}, max: {mask.max()}")
        
        # 读取图像
        if os.path.exists(raw_path):
            with rasterio.open(raw_path) as src:
                img = src.read()
                print(f"  图像 - shape: {img.shape}, dtype: {img.dtype}")
                print(f"         bands {BANDS} stats:")
                for i, b in enumerate(BANDS):
                    band = img[b-1]  # 1-indexed
                    print(f"           band {b}: min={band.min():.2f}, max={band.max():.2f}, mean={band.mean():.2f}, std={band.std():.2f}")
        else:
            print(f"  ⚠️  找不到对应图像: {raw_path}")

def check_dataset_loading():
    """2. 检查数据集加载"""
    print("\n" + "="*60)
    print("2. 检查数据集加载")
    print("="*60)
    
    dataset = FireDataset(
        data_dir=DEFAULT_DATA_DIR,
        region=REGION,
        bands=BANDS,
        mode='train',
        min_fg_pixels=MIN_FG_PIXELS
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("⚠️  数据集为空！")
        return None
    
    # 加载几个样本
    for i in range(min(3, len(dataset))):
        img, lbl = dataset[i]
        print(f"\nSample {i}:")
        print(f"  img shape: {img.shape}, dtype: {img.dtype}")
        print(f"  img range: [{img.min():.4f}, {img.max():.4f}]")
        print(f"  lbl shape: {lbl.shape}, dtype: {lbl.dtype}")
        print(f"  lbl unique: {torch.unique(lbl).tolist()}")
        print(f"  lbl sum: {lbl.sum().item()}, fg ratio: {lbl.sum().item()/lbl.numel()*100:.4f}%")
    
    return dataset

def check_normalization():
    """3. 检查归一化对数据的影响"""
    print("\n" + "="*60)
    print("3. 检查归一化影响")
    print("="*60)
    
    region_dir = os.path.join(DEFAULT_DATA_DIR, REGION)
    raw_dir = os.path.join(region_dir, 'raw')
    mask_dir = os.path.join(region_dir, 'mask_label')
    
    import glob
    mask_files = sorted(glob.glob(os.path.join(mask_dir, '*_voting_*.tif')))
    
    all_pixels = []
    for mask_path in mask_files[:20]:  # 采样20个
        raw_name = os.path.basename(mask_path).replace('_voting_', '_')
        raw_path = os.path.join(raw_dir, raw_name)
        
        if os.path.exists(raw_path):
            with rasterio.open(raw_path) as src:
                img = src.read()
                # 只收集 Bands 7, 6, 2
                for b in BANDS:
                    all_pixels.extend(img[b-1].flatten())
    
    if len(all_pixels) == 0:
        print("⚠️  未能采样到像素数据")
        return
    
    all_pixels = np.array(all_pixels)
    print(f"采样像素数: {len(all_pixels)}")
    print(f"原始值 - min: {all_pixels.min():.2f}, max: {all_pixels.max():.2f}")
    print(f"         mean: {all_pixels.mean():.2f}, std: {all_pixels.std():.2f}")
    print(f"         1%: {np.percentile(all_pixels, 1):.2f}, 99%: {np.percentile(all_pixels, 99):.2f}")
    
    # 模拟全局归一化
    normalized = (all_pixels - all_pixels.mean()) / (all_pixels.std() + 1e-8)
    print(f"\n全局归一化后 - min: {normalized.min():.4f}, max: {normalized.max():.4f}")
    print(f"                mean: {normalized.mean():.4f}, std: {normalized.std():.4f}")
    
    # 检查当前使用的per-patch min-max归一化的影响
    print(f"\n当前使用per-patch min-max归一化:")
    print(f"  - 会破坏band间相对关系")
    print(f"  - 如果patch内无变化，会产生噪声")

def visualize_samples(dataset, save_dir='/tmp/diagnose'):
    """4. 可视化输入/标签样本"""
    print("\n" + "="*60)
    print("4. 可视化样本")
    print("="*60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(min(5, len(dataset))):
        img, lbl = dataset[i]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 图像 (RGB组合)
        img_np = img.numpy()
        rgb = np.stack([img_np[0], img_np[1], img_np[2]], axis=-1)
        # 归一化到0-1用于显示
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        
        axes[0].imshow(rgb)
        axes[0].set_title(f'Input (B{BANDS})\nRange: [{img.min():.2f}, {img.max():.2f}]')
        axes[0].axis('off')
        
        # 标签
        lbl_np = lbl.numpy()
        axes[1].imshow(lbl_np, cmap='Reds', vmin=0, vmax=1)
        axes[1].set_title(f'Label\nFG: {lbl.sum()}/{lbl.numel()} ({lbl.sum()/lbl.numel()*100:.4f}%)')
        axes[1].axis('off')
        
        # 叠加
        axes[2].imshow(rgb)
        axes[2].imshow(lbl_np, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'sample_{i}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

def check_model_output():
    """5. 检查模型输出"""
    print("\n" + "="*60)
    print("5. 检查模型输出")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 创建模型
    model = create_model(num_classes=1, input_channels=len(BANDS), pretrained=False)
    model = model.to(device)
    model.eval()
    
    # 测试输入
    x = torch.randn(2, len(BANDS), 256, 256).to(device)
    
    with torch.no_grad():
        out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
    print(f"Output mean: {out.mean().item():.4f}, std: {out.std().item():.4f}")
    
    # 检查预测概率
    prob = torch.sigmoid(out)
    print(f"\nSigmoid后 - range: [{prob.min().item():.4f}, {prob.max().item():.4f}]")
    print(f"            mean: {prob.mean().item():.4f}")
    
    # 检查是否有梯度
    model.train()
    out = model(x)
    loss = out.mean()
    loss.backward()
    
    has_grad = False
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            grad_norms.append((name, grad_norm))
    
    if has_grad:
        print(f"\n✅ 梯度正常传播")
        grad_norms.sort(key=lambda x: x[1], reverse=True)
        print(f"Top 5 gradient norms:")
        for name, norm in grad_norms[:5]:
            print(f"  {name}: {norm:.6f}")
    else:
        print(f"\n⚠️  没有梯度！")
    
    return model

def check_single_overfit():
    """6. 单张过拟合测试"""
    print("\n" + "="*60)
    print("6. 单张过拟合测试")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据集
    dataset = FireDataset(
        data_dir=DEFAULT_DATA_DIR,
        region=REGION,
        bands=BANDS,
        mode='train',
        min_fg_pixels=MIN_FG_PIXELS
    )
    
    if len(dataset) == 0:
        print("⚠️  数据集为空，跳过过拟合测试")
        return
    
    # 取第一个样本
    img, lbl, meta = dataset[0]
    img = img.unsqueeze(0).to(device)  # Add batch dim
    lbl = lbl.unsqueeze(0).to(device)
    
    print(f"样本: {meta}")
    print(f"  FG ratio: {lbl.sum().item()/lbl.numel()*100:.4f}%")
    
    # 创建小模型
    model = create_model(num_classes=1, input_channels=len(BANDS), pretrained=False)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50.0]).to(device))
    
    model.train()
    losses = []
    
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, lbl)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            with torch.no_grad():
                prob = torch.sigmoid(out)
                pred = (prob > 0.5).float()
                
                tp = (pred * lbl).sum().item()
                fp = (pred * (1-lbl)).sum().item()
                fn = ((1-pred) * lbl).sum().item()
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                print(f"  Epoch {epoch}: Loss={loss.item():.4f}, P={precision:.2%}, R={recall:.2%}, F1={f1:.2%}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Single Sample Overfitting')
    plt.grid(True)
    plt.savefig('/tmp/diagnose/overfit_curve.png', dpi=150)
    plt.close()
    print(f"  Saved: /tmp/diagnose/overfit_curve.png")
    
    if losses[-1] < 0.1:
        print("\n✅ 模型可以过拟合单张样本 - 模型容量足够")
    else:
        print("\n⚠️  无法过拟合单张样本 - 可能有严重问题！")

def check_pretrained_weights():
    """7. 检查预训练权重加载"""
    print("\n" + "="*60)
    print("7. 检查预训练权重")
    print("="*60)
    
    pretrained_path = '/root/autodl-tmp/pretrained/mambavision_small_1k.pth'
    
    if not os.path.exists(pretrained_path):
        print(f"⚠️  预训练权重不存在: {pretrained_path}")
        return
    
    ckpt = torch.load(pretrained_path, map_location='cpu')
    print(f"Checkpoint keys: {list(ckpt.keys())[:10]}")
    
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    
    print(f"\n总参数数量: {len(state_dict)}")
    
    # 检查一些层的统计
    sample_keys = list(state_dict.keys())[:5]
    print(f"\nSample layers:")
    for k in sample_keys:
        v = state_dict[k]
        print(f"  {k}: shape={v.shape}, range=[{v.min():.4f}, {v.max():.4f}]")

def main():
    print("\n" + "="*60)
    print("训练问题深度诊断")
    print("="*60)
    
    # 1. 检查原始数据
    check_raw_data()
    
    # 2. 检查数据集加载
    dataset = check_dataset_loading()
    
    if dataset is None:
        print("\n⚠️  数据集为空，停止诊断")
        return
    
    # 3. 检查归一化
    check_normalization()
    
    # 4. 可视化样本
    visualize_samples(dataset)
    
    # 5. 检查模型输出
    check_model_output()
    
    # 6. 单张过拟合测试
    check_single_overfit()
    
    # 7. 检查预训练权重
    check_pretrained_weights()
    
    print("\n" + "="*60)
    print("诊断完成，检查 /tmp/diagnose/ 目录的可视化结果")
    print("="*60)

if __name__ == '__main__':
    main()
