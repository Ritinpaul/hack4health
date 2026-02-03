"""
Multi-Task Learning for Dental Caries Segmentation & Classification
Single file for Google Colab - GPU Optimized for 4GB VRAM

Author: Auto-generated
Usage: Upload this file to Colab and run all cells
"""

# ==============================================================================
# GPU Optimization Settings
# ==============================================================================

GPU_MEMORY_OPTIMIZED = True  # Set True for 4GB GPUs like RTX 3050 Ti

if GPU_MEMORY_OPTIMIZED:
    ENCODER = 'efficientnet-b2'  # Smaller encoder (b2 instead of b4)
    BATCH_SIZE = 4               # Reduced batch size
    GRADIENT_ACCUM_STEPS = 2     # Accumulate gradients over 2 steps (effective batch=8)
    IMAGE_SIZE = 224             # Slightly smaller images
else:
    ENCODER = 'efficientnet-b4'
    BATCH_SIZE = 8
    GRADIENT_ACCUM_STEPS = 1
    IMAGE_SIZE = 256

print(f"GPU Optimization: {'ENABLED' if GPU_MEMORY_OPTIMIZED else 'DISABLED'}")
print(f"Encoder: {ENCODER}, Batch: {BATCH_SIZE}, Accum: {GRADIENT_ACCUM_STEPS}, Img: {IMAGE_SIZE}")

# ==============================================================================
# Installation (Run once)
# ==============================================================================

# !pip install -q torch torchvision
# !pip install -q segmentation-models-pytorch albumentations opencv-python-headless
# !pip install -q scikit-learn pandas tqdm

# ==============================================================================
# Imports
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm.auto import tqdm
import os
from glob import glob

# ==============================================================================
# Dataset
# ==============================================================================

class MTLDataset(Dataset):
    def __init__(self, images_paths, masks_paths, labels, transform=None):
        self.images_paths = images_paths
        self.masks_paths = masks_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.images_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.masks_paths[idx] and os.path.exists(self.masks_paths[idx]):
            mask = cv2.imread(self.masks_paths[idx], cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 127, 1.0, cv2.THRESH_BINARY)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        mask = mask.astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)

        return image, mask, self.labels[idx]

# ==============================================================================
# Augmentation
# ==============================================================================

def get_train_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# ==============================================================================
# Model
# ==============================================================================

class MTLModel(nn.Module):
    def __init__(self, encoder_name='efficientnet-b2'):
        super(MTLModel, self).__init__()
        
        self.segmentation_model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None,
            decoder_attention_type='scse'
        )
        
        encoder_channels = self.segmentation_model.encoder.out_channels[-1]
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(encoder_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        seg_output = self.segmentation_model(x)
        encoder_features = self.segmentation_model.encoder(x)
        cls_features = encoder_features[-1]
        cls_pooled = self.gap(cls_features)
        cls_output = self.classification_head(cls_pooled)
        return seg_output, cls_output

# ==============================================================================
# Loss Functions
# ==============================================================================

class MTLLoss(nn.Module):
    def __init__(self, seg_weight=0.7, cls_weight=0.3):
        super(MTLLoss, self).__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight

    def dice_loss(self, pred, target):
        pred = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2. * intersection + 1) / (pred.sum() + target.sum() + 1)

    def forward(self, seg_out, cls_out, seg_tgt, cls_tgt):
        seg_loss = self.dice_loss(seg_out, seg_tgt)
        cls_loss = F.binary_cross_entropy_with_logits(cls_out, cls_tgt.view(-1, 1).float())
        total = self.seg_weight * seg_loss + self.cls_weight * cls_loss
        return total, seg_loss, cls_loss

# ==============================================================================
# Training
# ==============================================================================

def train(model, train_loader, val_loader, device, epochs=50, lr=3e-4):
    criterion = MTLLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)
    scaler = GradScaler()
    
    best_dice = 0.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for batch_idx, (imgs, masks, labels) in enumerate(pbar):
            imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
            
            with autocast():
                seg_out, cls_out = model(imgs)
                loss, seg_loss, cls_loss = criterion(seg_out, cls_out, masks, labels)
                loss = loss / GRADIENT_ACCUM_STEPS  # Scale loss for accumulation
            
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % GRADIENT_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * GRADIENT_ACCUM_STEPS
            pbar.set_postfix({'Loss': f'{train_loss/(batch_idx+1):.4f}'})
        
        # Validation
        model.eval()
        val_dice = 0
        with torch.no_grad():
            for imgs, masks, labels in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                seg_out, _ = model(imgs)
                pred = torch.sigmoid(seg_out)
                dice = (2 * (pred * masks).sum()) / (pred.sum() + masks.sum() + 1)
                val_dice += dice.item()
        
        val_dice /= len(val_loader)
        scheduler.step()
        
        print(f"Epoch {epoch}: Val Dice = {val_dice:.4f}")
        
        # Save best
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), 'mtl_best.pth')
            print(f"✓ New best! Dice: {best_dice:.4f}")
        
        # Save every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'mtl_epoch_{epoch}.pth')
    
    return model

# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == '__main__':
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # TODO: Upload your dataset or mount Google Drive
    # For Colab: !unzip /content/drive/MyDrive/dental_dataset.zip
    
    ROOT_DIR = '.'  # Change to your data path
    
    # Load data paths
    caries_imgs = [f for f in glob(f'{ROOT_DIR}/Carries/*.png') if 'mask' not in f]
    normal_imgs = [f for f in glob(f'{ROOT_DIR}/Normal/*.png') if 'mask' not in f]
    
    data = []
    for img in caries_imgs:
        mask = img.replace('.png', '-mask.png')
        if os.path.exists(mask):
            data.append((img, mask, 1))
    
    for img in normal_imgs:
        data.append((img, None, 0))
    
    imgs, masks, labels = zip(*data)
    
    # Split
    train_imgs, val_imgs, train_masks, val_masks, train_labels, val_labels = train_test_split(
        imgs, masks, labels, train_size=0.8, stratify=labels, random_state=42
    )
    
    # Datasets
    train_ds = MTLDataset(train_imgs, train_masks, train_labels, get_train_transform(IMAGE_SIZE))
    val_ds = MTLDataset(val_imgs, val_masks, val_labels, get_val_transform(IMAGE_SIZE))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model
    model = MTLModel(encoder_name=ENCODER).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    model = train(model, train_loader, val_loader, device, epochs=50, lr=3e-4)
    
    print("✓ Training complete!")
