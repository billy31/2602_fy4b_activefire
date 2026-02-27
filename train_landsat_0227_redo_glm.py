#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2026/02/27
# @Author   : AI Assistant
# @File     : train_landsat_0227_redo_glm.py
# @Desc     : MambaVision training script.
#             FIX: Robust file matching for naming inconsistencies (voting tag).
"""
本次更新内容 (v3):
1. 【核心修复】重写文件匹配逻辑。
	- 兼容 "Raw含voting, Label不含" 和 "Raw与Label同名" 两种情况。
	- 采用文件ID提取匹配，忽略 `_voting` 等细微差异。
2. 【调试增强】打印 Label 文件夹内容，便于排查路径问题。
3. 【环境提醒】增加 MambaVision 安装提示。
"""
import os
import sys
import glob
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
import random
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import rasterio
from PIL import Image
import albumentations as A
# ---------------------------
# 1. Environment Check & Import
# ---------------------------
try:
	# Try importing official MambaVision
	from mamba_vision import MambaVision
	MAMBA_AVAILABLE = True
	print("[Env] MambaVision module loaded successfully.")
except ImportError:
	MAMBA_AVAILABLE = False
	print("\n" + "!"*60)
	print("[Warning] 'mamba_vision' module not found.")
	print("Action: Please install it using: pip install mamba-vision (or git clone)")
	print("Currently using a Dummy Model for structure verification.")
	print("!"*60 + "\n")
	class MambaVision(nn.Module):
		def __init__(self, in_chans=3, **kwargs):
			super().__init__()
			self.stem = nn.Conv2d(in_chans, 64, kernel_size=7, stride=4, padding=3)
			self.num_features = 64
		def forward(self, x): return self.stem(x)
# ---------------------------
# 2. Git Sync
# ---------------------------
def git_sync(commit_msg="Update training script"):
	try:
		subprocess.run(["git", "add", "."], check=True, capture_output=True)
		subprocess.run(["git", "commit", "-m", commit_msg], check=True, capture_output=True)
		subprocess.run(["git", "push"], check=True, capture_output=True)
		print(f"[Git] Synced: {commit_msg}")
	except Exception:
		pass
# ---------------------------
# 3. Data Handling
# ---------------------------
class LandsatDataset(Dataset):
	def __init__(self, root_dir, regions, mode='train', transform=None):
		self.root_dir = root_dir
		self.transform = transform
		self.samples = []
		# 1. Path Detection
		potential_training_path = os.path.join(root_dir, 'training')
		if os.path.exists(potential_training_path):
			base_data_path = potential_training_path
		else:
			base_data_path = root_dir
		print(f"[Data] Base path: {base_data_path}")
		# 2. Region List
		if regions == 'All':
			region_list = [d for d in os.listdir(base_data_path) 
							if os.path.isdir(os.path.join(base_data_path, d)) and d.startswith('Asia')]
		else:
			region_list = [r.strip() for r in regions.split(',')]
		# 3. Scan Files
		for region in region_list:
			raw_dir = os.path.join(base_data_path, region, 'raw')
			label_dir = os.path.join(base_data_path, region, 'mask_label')
			if not os.path.exists(raw_dir) or not os.path.exists(label_dir):
				print(f"[Skip] Directory missing for region {region}")
				continue
			# Get all files
			raw_files = glob.glob(os.path.join(raw_dir, '*.tif'))
			label_files = glob.glob(os.path.join(label_dir, '*.tif'))
			# Debug: Print sample filenames from both sides
			if len(raw_files) > 0 and len(label_files) > 0:
				print(f"\n[Debug] Region: {region}")
				print(f"  Sample Raw:   {os.path.basename(raw_files[0])}")
				print(f"  Sample Label: {os.path.basename(label_files[0])}")
			# Build a map for labels for fast lookup
			# Key: processed_id (e.g., LC08_xxx_p00141), Value: full_path
			label_map = {}
			for p in label_files:
				fname = os.path.basename(p)
				# Extract the core ID: e.g., p00141
				# Assumption: ID is usually at the end like _p00141.tif
				match = re.search(r'(p\d+)', fname)
				if match:
					pid = match.group(1)
					label_map[pid] = p
			# Match Raw to Label
			for raw_path in raw_files:
				raw_name = os.path.basename(raw_path)
				# Extract ID from Raw
				match = re.search(r'(p\d+)', raw_name)
				if match:
					pid = match.group(1)
					if pid in label_map:
						label_path = label_map[pid]
						self.samples.append((raw_path, label_path))
		print(f"\n[Data] Total matched pairs: {len(self.samples)}")
		if len(self.samples) == 0:
			print("[Error] No pairs found. Please check if 'pXXXXX' IDs match between raw and label.")
	def __len__(self):
		return len(self.samples)
	def __getitem__(self, idx):
		raw_path, label_path = self.samples[idx]
		try:
			# Read Raw (Bands 7, 6, 2)
			with rasterio.open(raw_path) as src:
				if src.count >= 7:
					img = src.read([7, 6, 2])
				else:
					img = src.read([1, 2, 3]) # Fallback
				img = np.transpose(img, (1, 2, 0)).astype(np.float32)
				img = np.clip(img, 0, 10000) / 10000.0
			# Read Label
			with rasterio.open(label_path) as src:
				label = src.read(1).astype(np.float32)
			label = (label > 0).astype(np.float32)
			# Augmentation
			if self.transform:
				aug = self.transform(image=img, mask=label)
				img, label = aug['image'], aug['mask']
			# To Tensor
			img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
			label_tensor = torch.from_numpy(label).unsqueeze(0).long()
			return img_tensor, label_tensor, 1.0
		except Exception as e:
			print(f"Error loading: {raw_path} - {e}")
			return torch.zeros(3, 256, 256), torch.zeros(1, 256, 256).long(), 0.0
def get_transforms():
	return A.Compose([
		A.RandomRotate90(p=0.5),
		A.HorizontalFlip(p=0.5),
		A.VerticalFlip(p=0.5),
		A.RandomBrightnessContrast(p=0.3),
	])
# ---------------------------
# 4. Model
# ---------------------------
class MambaVisionSeg(nn.Module):
	def __init__(self, input_channels=3, num_classes=2):
		super().__init__()
		if MAMBA_AVAILABLE:
			self.backbone = MambaVision(in_chans=input_channels)
			feat_dim = getattr(self.backbone, 'num_features', 96)
		else:
			# Dummy backbone
			self.backbone = nn.Sequential(
				nn.Conv2d(3, 64, 7, stride=2, padding=3),
				nn.ReLU(),
				nn.Conv2d(64, 128, 3, stride=2, padding=1)
			)
			feat_dim = 128
		self.head = nn.Sequential(
			nn.Upsample(scale_factor=4, mode='bilinear'),
			nn.Conv2d(feat_dim, 128, 3, padding=1),
			nn.ReLU(),
			nn.Conv2d(128, num_classes, 1)
		)
	def forward(self, x):
		feat = self.backbone(x)
		if isinstance(feat, (tuple, list)): feat = feat[-1]
		return self.head(feat)
# ---------------------------
# 5. Metrics & Loss
# ---------------------------
def calc_metrics(pred, target, eps=1e-6):
	pred = torch.softmax(pred, dim=1)
	pred_class = pred.argmax(dim=1)
	target = target.squeeze(1)
	ious = []
	for cls in range(2):
		inter = ((pred_class == cls) & (target == cls)).sum().float()
		union = ((pred_class == cls) | (target == cls)).sum().float()
		ious.append((inter + eps) / (union + eps))
	miou = np.mean([i.item() for i in ious])
	tp = ((pred_class == 1) & (target == 1)).sum().float()
	fp = ((pred_class == 1) & (target == 0)).sum().float()
	fn = ((pred_class == 0) & (target == 1)).sum().float()
	prec = (tp + eps) / (tp + fp + eps)
	rec = (tp + eps) / (tp + fn + eps)
	f1 = 2 * (prec * rec) / (prec + rec + eps)
	return miou, f1.item(), rec.item(), prec.item()
class CombinedLoss(nn.Module):
	def __init__(self, weight):
		super().__init__()
		self.ce = nn.CrossEntropyLoss(weight=weight)
	def forward(self, pred, target):
		loss_ce = self.ce(pred, target.squeeze(1).long())
		# Dice Loss
		pred_prob = torch.softmax(pred, dim=1)[:, 1, ...]
		target_f = target.squeeze(1).float()
		inter = (pred_prob * target_f).sum()
		dice = 1 - (2. * inter + 1) / (pred_prob.sum() + target_f.sum() + 1)
		return loss_ce + dice
# ---------------------------
# 6. Main
# ---------------------------
def main(args):
	git_sync(f"Run: {args.regions}")
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# Data
	dataset = LandsatDataset(args.data_root, args.regions, transform=get_transforms())
	if len(dataset) == 0:
		print("[Fatal] No data found. Check filenames in Debug output above.")
		return
	loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
	# Model
	model = MambaVisionSeg().to(device)
	# Optimizer
	weights = torch.tensor([0.5, 5.0]).to(device) # Handle imbalance
	criterion = CombinedLoss(weights)
	optimizer = optim.AdamW(model.parameters(), lr=args.lr)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
	writer = SummaryWriter(args.log_dir)
	# Train
	for epoch in range(args.epochs):
		model.train()
		pbar = tqdm(loader, desc=f"Ep {epoch+1}")
		for imgs, labels, _ in pbar:
			imgs, labels = imgs.to(device), labels.to(device)
			optimizer.zero_grad()
			with torch.cuda.amp.autocast():
				out = model(imgs)
				loss = criterion(out, labels)
			loss.backward()
			optimizer.step()
			with torch.no_grad():
				miou, f1, _, _ = calc_metrics(out, labels)
			pbar.set_postfix({'loss': f"{loss.item():.3f}", 'mIoU': f"{miou:.3f}", 'F1': f"{f1:.3f}"})
		scheduler.step()
		torch.save(model.state_dict(), os.path.join(args.output_dir, 'last.pth'))
	writer.close()
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--data-root', type=str, default='/root/autodl-tmp/')
	parser.add_argument('--regions', type=str, default='Asia1')
	parser.add_argument('--batch-size', type=int, default=4)
	parser.add_argument('--epochs', type=int, default=50)
	parser.add_argument('--lr', type=float, default=2e-4)
	parser.add_argument('--log-dir', type=str, default='/root/tf-logs/')
	parser.add_argument('--output-dir', type=str, default='/root/autodl-tmp/output/')
	args = parser.parse_args()
	os.makedirs(args.output_dir, exist_ok=True)
	main(args)