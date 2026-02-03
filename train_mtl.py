import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
import argparse
from tqdm import tqdm
import os
import time

from src.mtl_dataset import get_mtl_dataloaders
from src.mtl_model import create_mtl_model, count_parameters
from src.mtl_losses import create_mtl_loss
from src.metrics import dice_coefficient, jaccard_index
from src.utils import AverageMeter, save_checkpoint

def calculate_classification_metrics(outputs, targets):
    """Calculate classification accuracy."""
    preds = torch.sigmoid(outputs) > 0.5
    targets = targets.view(-1, 1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total

def train_epoch(model, loader, optimizer, criterion, scaler, device, epoch, total_epochs, grad_accum_steps=1):
    model.train()
    
    total_loss_meter = AverageMeter()
    seg_loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    cls_acc_meter = AverageMeter()
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", ncols=120)
    
    optimizer.zero_grad()
    
    for batch_idx, (images, masks, labels) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        
        with autocast():
            seg_output, cls_output = model(images)
            total_loss, seg_loss, cls_loss = criterion(seg_output, cls_output, masks, labels)
            total_loss = total_loss / grad_accum_steps  # Scale loss for accumulation
        
        scaler.scale(total_loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Calculate metrics
        with torch.no_grad():
            seg_preds = torch.sigmoid(seg_output)
            dice = dice_coefficient(seg_preds, masks).item()
            cls_acc = calculate_classification_metrics(cls_output, labels)
        
        # Update meters
        batch_size = images.size(0)
        total_loss_meter.update(total_loss.item() * grad_accum_steps, batch_size)
        seg_loss_meter.update(seg_loss.item(), batch_size)
        cls_loss_meter.update(cls_loss.item(), batch_size)
        dice_meter.update(dice, batch_size)
        cls_acc_meter.update(cls_acc, batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{total_loss_meter.avg:.4f}',
            'Dice': f'{dice_meter.avg:.4f}',
            'Acc': f'{cls_acc_meter.avg:.2%}'
        })
    
    return {
        'total_loss': total_loss_meter.avg,
        'seg_loss': seg_loss_meter.avg,
        'cls_loss': cls_loss_meter.avg,
        'dice': dice_meter.avg,
        'cls_acc': cls_acc_meter.avg
    }

def validate(model, loader, criterion, device, epoch, total_epochs):
    model.eval()
    
    total_loss_meter = AverageMeter()
    seg_loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    cls_acc_meter = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Val]  ", ncols=120)
        
        for images, masks, labels in pbar:
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            seg_output, cls_output = model(images)
            total_loss, seg_loss, cls_loss = criterion(seg_output, cls_output, masks, labels)
            
            seg_preds = torch.sigmoid(seg_output)
            dice = dice_coefficient(seg_preds, masks).item()
            iou = jaccard_index(seg_preds, masks).item()
            cls_acc = calculate_classification_metrics(cls_output, labels)
            
            batch_size = images.size(0)
            total_loss_meter.update(total_loss.item(), batch_size)
            seg_loss_meter.update(seg_loss.item(), batch_size)
            cls_loss_meter.update(cls_loss.item(), batch_size)
            dice_meter.update(dice, batch_size)
            iou_meter.update(iou, batch_size)
            cls_acc_meter.update(cls_acc, batch_size)
            
            pbar.set_postfix({
                'Loss': f'{total_loss_meter.avg:.4f}',
                'Dice': f'{dice_meter.avg:.4f}',
                'Acc': f'{cls_acc_meter.avg:.2%}'
            })
    
    return {
        'total_loss': total_loss_meter.avg,
        'seg_loss': seg_loss_meter.avg,
        'cls_loss': cls_loss_meter.avg,
        'dice': dice_meter.avg,
        'iou': iou_meter.avg,
        'cls_acc': cls_acc_meter.avg
    }

def print_epoch_summary(epoch, total_epochs, train_metrics, val_metrics):
    """Print formatted epoch summary."""
    print("=" * 100)
    print(f"Epoch {epoch}/{total_epochs} Summary")
    print("-" * 100)
    print("TRAINING:")
    print(f"  Total Loss: {train_metrics['total_loss']:.4f} | "
          f"Seg Loss: {train_metrics['seg_loss']:.4f} | "
          f"Cls Loss: {train_metrics['cls_loss']:.4f}")
    print(f"  Dice: {train_metrics['dice']:.4f} | "
          f"Cls Acc: {train_metrics['cls_acc']:.2%}")
    
    print("\nVALIDATION:")
    print(f"  Total Loss: {val_metrics['total_loss']:.4f} | "
          f"Seg Loss: {val_metrics['seg_loss']:.4f} | "
          f"Cls Loss: {val_metrics['cls_loss']:.4f}")
    print(f"  Dice: {val_metrics['dice']:.4f} | "
          f"IoU: {val_metrics['iou']:.4f} | "
          f"Cls Acc: {val_metrics['cls_acc']:.2%}")

def main():
    parser = argparse.ArgumentParser(description='Train Multi-Task Dental Model')
    parser.add_argument('--root', type=str, default='.', help='Root directory')
    parser.add_argument('--encoder', type=str, default='efficientnet-b4', help='Encoder backbone')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--workers', type=int, default=0, help='Data loader workers')
    parser.add_argument('--patience', type=int, default=25, help='Early stopping patience')
    parser.add_argument('--seg-weight', type=float, default=0.7, help='Segmentation loss weight')
    parser.add_argument('--cls-weight', type=float, default=0.3, help='Classification loss weight')
    parser.add_argument('--save-every', type=int, default=10, help='Save checkpoint every N epochs')
    
    # GPU Optimization for 4GB VRAM (RTX 3050 Ti)
    parser.add_argument('--gpu-opt', action='store_true', help='Enable GPU memory optimization for 4GB VRAM')
    parser.add_argument('--grad-accum', type=int, default=1, help='Gradient accumulation steps')
    
    args = parser.parse_args()
    
    # Auto-adjust for GPU optimization
    if args.gpu_opt:
        print("GPU Optimization ENABLED for 4GB VRAM")
        if args.encoder == 'efficientnet-b4':
            args.encoder = 'efficientnet-b2'
            print(f"  → Encoder downgraded: efficientnet-b4 → efficientnet-b2")
        if args.batch_size > 4:
            args.batch_size = 4
            print(f"  → Batch size reduced to 4")
        if args.grad_accum == 1:
            args.grad_accum = 2
            print(f"  → Gradient accumulation: 2 steps (effective batch = 8)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    train_loader, val_loader = get_mtl_dataloaders(
        args.root, batch_size=args.batch_size, num_workers=args.workers
    )
    
    # Model
    model = create_mtl_model(encoder_name=args.encoder)
    model = model.to(device)
    print(f"Model: UNet++ with {args.encoder} encoder")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Loss, Optimizer, Scheduler
    criterion = create_mtl_loss(seg_weight=args.seg_weight, cls_weight=args.cls_weight)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = GradScaler()
    
    # Training
    best_dice = 0.0
    patience_counter = 0
    
    os.makedirs('checkpoints', exist_ok=True)
    
    print("\n" + "=" * 100)
    print("Starting Multi-Task Training")
    print("=" * 100 + "\n")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch, args.epochs, args.grad_accum
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, epoch, args.epochs
        )
        
        scheduler.step()
        
        # Print summary
        print_epoch_summary(epoch, args.epochs, train_metrics, val_metrics)
        
        # Save checkpoint every N epochs
        if epoch % args.save_every == 0:
            checkpoint_path = f'checkpoints/mtl_model_epoch_{epoch}.pth'
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'dice': val_metrics['dice'],
                'iou': val_metrics['iou'],
                'cls_acc': val_metrics['cls_acc']
            }, filename=checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            patience_counter = 0
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_dice': best_dice,
                'iou': val_metrics['iou'],
                'cls_acc': val_metrics['cls_acc']
            }, filename='checkpoints/mtl_model_best.pth')
            print(f"✓ New best model! Dice: {best_dice:.4f}")
        else:
            patience_counter += 1
        
        print("=" * 100 + "\n")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping after {epoch} epochs. Best Dice: {best_dice:.4f}")
            break
    
    print(f"\nTraining complete! Best Dice: {best_dice:.4f}")

if __name__ == '__main__':
    main()
