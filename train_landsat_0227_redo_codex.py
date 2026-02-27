
#!/usr/bin/env python3
"""
train_landsat_0227_redo_codex.py

Landsat-8 fire detection training with MambaVision backbone.
- Single source domain training (Landsat-8) with input-layer adaptation.
- Robust region selection: Asia1 | Asia1,Asia2 | ALL
- Strong positive augmentation + imbalance-aware sampling
- Metrics: mIoU, F1, Recall, Precision (focus on mIoU/F1)
- TensorBoard on by default
- Auto git sync each run (best-effort)

Data layout (per datadescription):
  {data_dir}/{Region}/raw
  {data_dir}/{Region}/mask_label

Labels: 1 = fire, 0 = background (binarized with label > 0).
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
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter

import rasterio

try:
    import cv2  # optional, used for resize/affine
except Exception:
    cv2 = None
    from PIL import Image


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = "/root/autodl-tmp/training"
DEFAULT_PRETRAIN_DIR = "/root/autodl-tmp/pretrained"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/training/output"
DEFAULT_TENSORBOARD_DIR = "/root/tf-logs"


# -----------------------------------------------------------------------------
# MambaVision import
# -----------------------------------------------------------------------------

def import_mambavision():
    try:
        from mambavision import create_model  # type: ignore
        return create_model
    except Exception:
        mv_path = os.environ.get("MAMBAVISION_PATH", "/root/codes/fire0226/MambaVision")
        if os.path.isdir(mv_path):
            sys.path.insert(0, mv_path)
            from mambavision import create_model  # type: ignore
            return create_model
        raise RuntimeError("Cannot import mambavision. Set MAMBAVISION_PATH or install package.")


create_model = import_mambavision()


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def git_auto_commit(message: str, cwd: Optional[str] = None):
    try:
        import subprocess

        repo = cwd or os.path.dirname(os.path.abspath(__file__))
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=repo)
        if result.returncode != 0:
            logger.warning("Not a git repository or git error")
            return
        if result.stdout.strip():
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
            commit_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {message}"
            subprocess.run(["git", "commit", "-m", commit_msg], cwd=repo, check=True)
            push_result = subprocess.run(["git", "push"], capture_output=True, text=True, cwd=repo)
            if push_result.returncode == 0:
                logger.info("Git synced: %s", message[:60])
            else:
                logger.warning("Git commit OK but push failed")
        else:
            logger.info("No code changes to commit")
    except Exception as e:
        logger.warning("Git auto-commit failed: %s", e)


def resize_band(band: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    if cv2 is not None:
        return cv2.resize(band, size, interpolation=cv2.INTER_LINEAR)
    img = Image.fromarray(band)
    return np.array(img.resize(size, resample=Image.BILINEAR))


def resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    if cv2 is not None:
        return cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    img = Image.fromarray(mask.astype(np.uint8))
    return np.array(img.resize(size, resample=Image.NEAREST))


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(pred)
        intersection = (probs * target).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return (1 - dice).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        probs = torch.sigmoid(pred)
        pt = torch.where(target > 0, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class CombinedLoss(nn.Module):
    def __init__(self, pos_weight: float = 5.0, dice_weight: float = 1.0, focal_weight: float = 0.5):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        if self.pos_weight.device != pred.device:
            self.pos_weight = self.pos_weight.to(pred.device)
        bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=self.pos_weight)
        total = bce
        if self.dice_weight > 0:
            total += self.dice_weight * self.dice(pred, target)
        if self.focal_weight > 0:
            total += self.focal_weight * self.focal(pred, target)
        return total

# -----------------------------------------------------------------------------
# Decoder (FPN)
# -----------------------------------------------------------------------------

class FPNDecoder(nn.Module):
    def __init__(self, encoder_dims: List[int], num_classes: int = 1):
        super().__init__()
        self.lateral4 = nn.Conv2d(encoder_dims[3], 256, 1)
        self.lateral3 = nn.Conv2d(encoder_dims[2], 256, 1)
        self.lateral2 = nn.Conv2d(encoder_dims[1], 256, 1)
        self.lateral1 = nn.Conv2d(encoder_dims[0], 256, 1)
        self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth1 = nn.Conv2d(256, 256, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, num_classes, 1),
        )

    def forward(self, features: List[torch.Tensor], input_shape: Tuple[int, int]) -> torch.Tensor:
        f1, f2, f3, f4 = features
        p4 = self.lateral4(f4)
        p3 = self.smooth3(self.lateral3(f3) + self.upsample(p4))
        p2 = self.smooth2(self.lateral2(f2) + self.upsample(p3))
        p1 = self.smooth1(self.lateral1(f1) + self.upsample(p2))
        out = self.seg_head(p1)
        out = F.interpolate(out, size=input_shape, mode="bilinear", align_corners=False)
        return out


def extract_features(backbone: nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
    features = []
    x = backbone.patch_embed(x)
    features.append(x)
    for i, level in enumerate(backbone.levels):
        x = level(x)
        if i < 3:
            features.append(x)
    x = backbone.norm(x)
    features.append(x)
    return features[:4]


class FireDetectionModel(nn.Module):
    def __init__(self, model_name: str, input_channels: int, pretrained_path: Optional[str]):
        super().__init__()
        self.backbone = create_model(model_name, pretrained=False, num_classes=1)
        if input_channels != 3:
            self._modify_input(input_channels)
        if pretrained_path:
            self._load_pretrained(pretrained_path)

        dims_map = {
            "mamba_vision_T": [96, 192, 384, 384],
            "mamba_vision_S": [96, 192, 384, 768],
            "mamba_vision_B": [128, 256, 512, 1024],
            "mamba_vision_L": [128, 256, 512, 1568],
        }
        self.encoder_dims = dims_map.get(model_name, [96, 192, 384, 768])
        self.decoder = FPNDecoder(self.encoder_dims, num_classes=1)

    def _modify_input(self, input_channels: int):
        if hasattr(self.backbone, "patch_embed") and hasattr(self.backbone.patch_embed, "conv_down"):
            conv = self.backbone.patch_embed.conv_down[0]
            new_conv = nn.Conv2d(
                input_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                bias=conv.bias is not None,
            )
            with torch.no_grad():
                n = conv.weight.size(1)
                repeat = (input_channels + n - 1) // n
                w = conv.weight.repeat(1, repeat, 1, 1)
                new_conv.weight.copy_(w[:, :input_channels, :, :])
                if conv.bias is not None:
                    new_conv.bias.copy_(conv.bias)
            self.backbone.patch_embed.conv_down[0] = new_conv

    def _load_pretrained(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))
        state = {k: v for k, v in state.items() if not k.startswith("head.")}
        self.backbone.load_state_dict(state, strict=False)
        logger.info("Loaded pretrained: %s", path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2:]
        features = extract_features(self.backbone, x)
        return self.decoder(features, (H, W))

    def get_param_groups(self, lr: float, backbone_lr_scale: float) -> List[Dict]:
        return [
            {"params": self.backbone.parameters(), "lr": lr * backbone_lr_scale, "name": "backbone"},
            {"params": self.decoder.parameters(), "lr": lr, "name": "decoder"},
        ]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class FireDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        region: str,
        bands: List[int],
        mode: str = "train",
        split: float = 0.8,
        seed: int = 42,
        min_fg_pixels: int = 5,
        neg_per_pos: float = 1.0,
        crop_to_fire: bool = True,
        crop_scale_min: float = 0.6,
        crop_scale_max: float = 1.0,
    ):
        self.raw_dir = os.path.join(data_dir, region, "raw")
        self.label_dir = os.path.join(data_dir, region, "mask_label")
        self.bands = bands
        self.mode = mode
        self.min_fg_pixels = min_fg_pixels
        self.neg_per_pos = max(0.0, float(neg_per_pos))
        self.rng = np.random.default_rng(seed)
        self.crop_to_fire = crop_to_fire
        self.crop_scale_min = float(crop_scale_min)
        self.crop_scale_max = float(crop_scale_max)

        samples = self._scan_samples()
        self.samples = self._filter_samples(samples)

        self.rng = np.random.default_rng(seed)
        indices = self.rng.permutation(len(self.samples))
        split_idx = int(len(indices) * split)
        if mode == "train":
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]

        logger.info("[%s] %s: %d patches", mode, region, len(self.indices))

    @staticmethod
    def _binarize_label(label: np.ndarray) -> np.ndarray:
        return (label > 0).astype(np.uint8)

    def _scan_samples(self) -> List[Dict]:
        samples = []
        if not os.path.exists(self.label_dir):
            return samples
        for f in os.listdir(self.label_dir):
            if not f.endswith(".tif"):
                continue
            label_path = os.path.join(self.label_dir, f)
            if "_voting_" in f:
                raw_f = f.replace("_voting_", "_")
            else:
                raw_f = f.replace(".tif", "_voting_.tif")
            raw_path = os.path.join(self.raw_dir, raw_f)
            if os.path.exists(raw_path):
                samples.append({"raw": raw_path, "label": label_path})
        return samples

    def _filter_samples(self, samples: List[Dict]) -> List[Dict]:
        pos_samples, neg_samples = [], []
        for s in samples:
            try:
                with rasterio.open(s["label"]) as src:
                    label = src.read(1)
                label = self._binarize_label(label)
                fg = int((label > 0).sum())
                s["fg_count"] = fg
                if fg >= self.min_fg_pixels:
                    pos_samples.append(s)
                else:
                    neg_samples.append(s)
            except Exception:
                pass
        if len(pos_samples) == 0:
            kept = neg_samples
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
        logger.info("Filtered: pos=%d, neg_kept=%d", len(pos_samples), len(kept) - len(pos_samples))
        return kept

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[self.indices[idx]]
        with rasterio.open(s["raw"]) as src:
            bands = self.bands if max(self.bands) <= src.count else list(range(1, src.count + 1))
            image = src.read(bands).astype(np.float32)
        with rasterio.open(s["label"]) as src:
            label = src.read(1)
        label = self._binarize_label(label)

        image = self._normalize(image)
        if self.mode == "train":
            image, label = self._augment(image, label)
        return torch.from_numpy(image).float(), torch.from_numpy(label.astype(np.int64))

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        for i in range(image.shape[0]):
            p1 = np.percentile(image[i], 1)
            p99 = np.percentile(image[i], 99)
            band = np.clip(image[i], p1, p99)
            image[i] = (band - p1) / (p99 - p1 + 1e-8)
        return np.clip(image, 0, 1)

    def _augment(self, img: np.ndarray, lbl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        C, H, W = img.shape
        has_fire = (lbl > 0).any()

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

        if has_fire:
            if self.crop_to_fire and np.random.rand() < 0.6:
                img, lbl = self._crop_to_fire(img, lbl)
            if np.random.rand() < 0.3:
                img, lbl = self._fire_copy_paste(img, lbl)
            if np.random.rand() < 0.4:
                fire_mask = lbl > 0
                intensity = 1.1 + 0.4 * np.random.rand()
                for c in range(min(3, C)):
                    img_c = img[c].copy()
                    img_c[fire_mask] = np.clip(img_c[fire_mask] * intensity, 0, 1)
                    img[c] = img_c

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
        idx = self.rng.integers(0, len(ys))
        cy, cx = int(ys[idx]), int(xs[idx])
        scale = self.rng.uniform(self.crop_scale_min, self.crop_scale_max)
        ch = max(16, int(H * scale))
        cw = max(16, int(W * scale))
        y1 = int(np.clip(cy - ch // 2, 0, H - ch))
        x1 = int(np.clip(cx - cw // 2, 0, W - cw))
        y2, x2 = y1 + ch, x1 + cw
        img_crop = img[:, y1:y2, x1:x2]
        lbl_crop = lbl[y1:y2, x1:x2]
        img_resized = np.stack([resize_band(img_crop[c], (W, H)) for c in range(C)], axis=0)
        lbl_resized = resize_mask(lbl_crop, (W, H))
        return img_resized, lbl_resized

    def _fire_copy_paste(self, img: np.ndarray, lbl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        C, H, W = img.shape
        fire_mask = lbl > 0
        if not fire_mask.any():
            return img, lbl
        result_img, result_lbl = img.copy(), lbl.copy()
        for _ in range(self.rng.integers(1, 3)):
            y_offset = int(self.rng.integers(-H // 4, H // 4))
            x_offset = int(self.rng.integers(-W // 4, W // 4))
            if cv2 is None:
                continue
            M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
            shifted_mask = cv2.warpAffine(fire_mask.astype(np.uint8), M, (W, H)) > 0
            overlap = (result_lbl > 0) & shifted_mask
            if overlap.sum() < shifted_mask.sum() * 0.3:
                for c in range(C):
                    shifted_band = cv2.warpAffine(img[c], M, (W, H))
                    result_img[c] = np.where(shifted_mask, shifted_band, result_img[c])
                result_lbl = np.maximum(result_lbl, shifted_mask.astype(np.int64))
        return result_img, result_lbl

    def get_sample_weights(self) -> List[float]:
        # Weights must align with self.indices (train/val split)
        weights = []
        for idx in self.indices:
            s = self.samples[int(idx)]
            weights.append(1.0 if s.get("fg_count", 0) > 0 else 0.25)
        return weights

# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def compute_metrics(tp: int, fp: int, fn: int) -> Tuple[float, float, float, float]:
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    miou = tp / (tp + fp + fn + 1e-8)
    return miou * 100, f1 * 100, recall * 100, precision * 100


@torch.no_grad()

def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
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
    thresholds = np.linspace(0.05, 0.95, 19).tolist()
    best_f1, best_thresh = 0.0, 0.5
    best_tp = best_fp = best_fn = 0
    for thresh in thresholds:
        preds = (all_probs > thresh).long()
        gt = (all_targets > 0).long()
        tp = ((preds == 1) & (gt == 1)).sum().item()
        fp = ((preds == 1) & (gt == 0)).sum().item()
        fn = ((preds == 0) & (gt == 1)).sum().item()
        _, f1, _, _ = compute_metrics(tp, fp, fn)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_tp, best_fp, best_fn = tp, fp, fn

    miou, f1, recall, precision = compute_metrics(best_tp, best_fp, best_fn)
    avg_loss = total_loss / max(len(loader), 1)
    pos_ratio = all_targets.float().mean().item() * 100
    logger.info(
        "Val - Loss:%.4f mIoU:%.2f%% F1:%.2f%% R:%.2f%% P:%.2f%% Best@%.2f (pos=%.4f%%)",
        avg_loss,
        miou,
        f1,
        recall,
        precision,
        best_thresh,
        pos_ratio,
    )
    return avg_loss, miou, f1, recall, precision, best_thresh


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler],
    use_amp: bool,
    max_grad_norm: float,
):
    model.train()
    total_loss = 0.0
    tp = fp = fn = 0
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        with autocast("cuda", enabled=use_amp):
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

        total_loss += loss.item()
        with torch.no_grad():
            probs = torch.sigmoid(outputs).squeeze(1)
            preds = (probs > 0.5).long()
            gt = (labels > 0).long()
            tp += ((preds == 1) & (gt == 1)).sum().item()
            fp += ((preds == 1) & (gt == 0)).sum().item()
            fn += ((preds == 0) & (gt == 1)).sum().item()
        if i % 20 == 0:
            miou, f1, recall, precision = compute_metrics(tp, fp, fn)
            logger.info(
                "  [%d/%d] Loss:%.4f mIoU:%.2f%% F1:%.2f%% R:%.2f%% P:%.2f%%",
                i,
                len(loader),
                loss.item(),
                miou,
                f1,
                recall,
                precision,
            )

    miou, f1, recall, precision = compute_metrics(tp, fp, fn)
    avg_loss = total_loss / max(len(loader), 1)
    logger.info(
        "Train - Loss:%.4f mIoU:%.2f%% F1:%.2f%% R:%.2f%% P:%.2f%%",
        avg_loss,
        miou,
        f1,
        recall,
        precision,
    )
    return avg_loss, miou, f1, recall, precision


# -----------------------------------------------------------------------------
# Regions
# -----------------------------------------------------------------------------

def parse_regions_arg(regions_arg: str, data_dir: str) -> List[str]:
    if regions_arg.upper() == "ALL":
        regions = []
        for d in sorted(os.listdir(data_dir)):
            dir_path = os.path.join(data_dir, d)
            if os.path.isdir(dir_path) and d not in ["output", "meta", ".git"] and not d.startswith("."):
                raw_dir = os.path.join(dir_path, "raw")
                label_dir = os.path.join(dir_path, "mask_label")
                if os.path.exists(raw_dir) and os.path.exists(label_dir):
                    regions.append(d)
        logger.info("Auto-detected regions: %s", regions)
        return regions
    return [r.strip() for r in regions_arg.split(",") if r.strip()]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Landsat Fire Training (MambaVision)")
    parser.add_argument("regions", type=str, help="Asia1 | Asia1,Asia2 | ALL")

    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--tensorboard-dir", type=str, default=DEFAULT_TENSORBOARD_DIR)

    parser.add_argument("--bands", type=int, nargs="+", default=[7, 6, 2])
    parser.add_argument("--min-fg-pixels", type=int, default=5)
    parser.add_argument("--neg-per-pos", type=float, default=1.0)
    parser.add_argument("--crop-to-fire", action="store_true", default=True)
    parser.add_argument("--crop-scale-min", type=float, default=0.6)
    parser.add_argument("--crop-scale-max", type=float, default=1.0)

    parser.add_argument("--model", type=str, default="mamba_vision_S",
                        choices=["mamba_vision_T", "mamba_vision_S", "mamba_vision_B", "mamba_vision_L"])
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--pretrained-path", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--use-amp", action="store_true", default=True)

    parser.add_argument("--pos-weight", type=float, default=5.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--focal-weight", type=float, default=0.5)

    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--no-tensorboard", action="store_true", default=False)

    args = parser.parse_args()

    regions = parse_regions_arg(args.regions, args.data_dir)
    if len(regions) == 0:
        raise ValueError("No valid regions found")

    torch.manual_seed(42)
    np.random.seed(42)

    if args.output_dir is None:
        region_name = regions[0] if len(regions) == 1 else f"multi{len(regions)}"
        args.output_dir = os.path.join(DEFAULT_OUTPUT_DIR, f"{region_name}_mambavision")
    os.makedirs(args.output_dir, exist_ok=True)

    git_auto_commit(f"Start training: {regions}, bands={args.bands}, model={args.model}")

    writer = None
    if not args.no_tensorboard:
        exp_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_landsat_mamba"
        tb_dir = os.path.join(args.tensorboard_dir, exp_name)
        writer = SummaryWriter(tb_dir)
        logger.info("TensorBoard: %s", tb_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Regions: %s", regions)
    logger.info("Bands: %s", args.bands)

    train_datasets, val_datasets = [], []
    for region in regions:
        try:
            train_ds = FireDataset(
                args.data_dir, region, args.bands, "train",
                min_fg_pixels=args.min_fg_pixels, neg_per_pos=args.neg_per_pos,
                crop_to_fire=args.crop_to_fire,
                crop_scale_min=args.crop_scale_min, crop_scale_max=args.crop_scale_max,
            )
            val_ds = FireDataset(
                args.data_dir, region, args.bands, "val",
                min_fg_pixels=args.min_fg_pixels, neg_per_pos=args.neg_per_pos,
                crop_to_fire=False,
                crop_scale_min=args.crop_scale_min, crop_scale_max=args.crop_scale_max,
            )
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
        except Exception as e:
            logger.warning("Skip region %s: %s", region, e)

    if len(train_datasets) == 0:
        raise ValueError("No valid datasets")

    train_ds = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_ds = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]

    weights = None
    if hasattr(train_ds, "get_sample_weights"):
        weights = train_ds.get_sample_weights()
    elif isinstance(train_ds, ConcatDataset):
        all_weights = []
        for ds in train_ds.datasets:
            if hasattr(ds, "get_sample_weights"):
                all_weights.extend(ds.get_sample_weights())
        if len(all_weights) == len(train_ds):
            weights = all_weights
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True) if weights else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None),
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    pretrained_path = args.pretrained_path or (
        os.path.join(DEFAULT_PRETRAIN_DIR, "mambavision_small_1k.pth") if args.pretrained else None
    )
    model = FireDetectionModel(args.model, len(args.bands), pretrained_path).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Params: %.2fM", total_params / 1e6)

    criterion = CombinedLoss(pos_weight=args.pos_weight, dice_weight=args.dice_weight, focal_weight=args.focal_weight)
    param_groups = model.get_param_groups(args.lr, args.backbone_lr_scale)
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=1e-6)
    scaler = GradScaler() if args.use_amp else None

    best_f1 = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            lr = args.lr * epoch / args.warmup_epochs
            for pg in optimizer.param_groups:
                if pg.get("name") == "backbone":
                    pg["lr"] = lr * args.backbone_lr_scale
                else:
                    pg["lr"] = lr
            logger.info("Epoch %d/%d [Warmup] lr=%.2e", epoch, args.epochs, lr)
        else:
            scheduler.step()
            logger.info("Epoch %d/%d lr=%.2e", epoch, args.epochs, optimizer.param_groups[0]["lr"])

        logger.info("-" * 60)
        train_loss, train_miou, train_f1, train_r, train_p = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, args.use_amp, args.max_grad_norm
        )
        val_loss, val_miou, val_f1, val_r, val_p, best_thresh = validate(
            model, val_loader, criterion, device
        )

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Metrics/mIoU", val_miou, epoch)
            writer.add_scalar("Metrics/F1", val_f1, epoch)
            writer.add_scalar("Metrics/Recall", val_r, epoch)
            writer.add_scalar("Metrics/Precision", val_p, epoch)
            writer.add_scalar("Threshold/best", best_thresh, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "f1": val_f1, "miou": val_miou},
                os.path.join(args.output_dir, "best_model.pth"),
            )
            logger.info("Saved best model (F1: %.2f%% @ %d, mIoU: %.2f%%)", best_f1, best_epoch, val_miou)
        else:
            epochs_no_improve += 1
            logger.info("No improvement for %d epochs", epochs_no_improve)

        if epochs_no_improve >= args.early_stop_patience:
            logger.warning("Early stopping! Best F1: %.2f%% @ epoch %d", best_f1, best_epoch)
            break

    git_auto_commit(f"Complete: best F1={best_f1:.2f}% @ epoch {best_epoch}, regions={regions}")
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
