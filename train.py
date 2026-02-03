import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
import argparse
from tqdm import tqdm
import os

from src.data_loader import get_dataloaders
from src.model import create_model
from src.losses import CombinedLoss
from src.metrics import dice_coefficient, jaccard_index, pixel_accuracy
from src.utils import AverageMeter, save_checkpoint

def train_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        with torch.no_grad():
            preds = torch.sigmoid(outputs)
            dice = dice_coefficient(preds, masks)
        
        loss_meter.update(loss.item(), images.size(0))
        dice_meter.update(dice.item(), images.size(0))
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", dice=f"{dice_meter.avg:.4f}")
    
    return loss_meter.avg, dice_meter.avg

def validate(model, loader, criterion, device, epoch):
    model.eval()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            preds = torch.sigmoid(outputs)
            dice = dice_coefficient(preds, masks)
            iou = jaccard_index(preds, masks)
            acc = pixel_accuracy(preds, masks)
            
            loss_meter.update(loss.item(), images.size(0))
            dice_meter.update(dice.item(), images.size(0))
            iou_meter.update(iou.item(), images.size(0))
            acc_meter.update(acc.item(), images.size(0))
            
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", dice=f"{dice_meter.avg:.4f}")
    
    return loss_meter.avg, dice_meter.avg, iou_meter.avg, acc_meter.avg

def main():
    parser = argparse.ArgumentParser(description='Train Dental Segmentation Model')
    parser.add_argument('--root', type=str, default='.', help='Root directory')
    parser.add_argument('--arch', type=str, default='UnetPlusPlus', help='Model architecture')
    parser.add_argument('--encoder', type=str, default='efficientnet-b4', help='Encoder backbone')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--workers', type=int, default=0, help='Data loader workers')
    parser.add_argument('--patience', type=int, default=25, help='Early stopping patience')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    train_loader, val_loader, _ = get_dataloaders(
        args.root, batch_size=args.batch_size, num_workers=args.workers
    )
    
    # Model
    model = create_model(arch=args.arch, encoder_name=args.encoder)
    model = model.to(device)
    print(f"Model: {args.arch} with {args.encoder} encoder")
    
    # Loss, Optimizer, Scheduler
    criterion = CombinedLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = GradScaler()
    
    # Training
    best_dice = 0.0
    patience_counter = 0
    
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch
        )
        
        val_loss, val_dice, val_iou, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )
        
        scheduler.step()
        
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}, Acc: {val_acc:.4f}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_dice': best_dice,
                'optimizer': optimizer.state_dict(),
            }, filename=f'checkpoints/dental_model_best.pth')
            print(f"  ✓ New best model saved! Dice: {best_dice:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping after {epoch} epochs. Best Dice: {best_dice:.4f}")
            break
    
    print(f"\nTraining complete! Best Dice: {best_dice:.4f}")

if __name__ == '__main__':
    main()
