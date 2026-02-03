import argparse
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

from src.data_loader import get_dataloaders
from src.model import create_model
from src.losses import get_loss_function
from src.utils import AverageMeter, save_checkpoint
from src.metrics import dice_coefficient, jaccard_index, pixel_accuracy

def train(model, loader, criterion, optimizer, scheduler, device, scaler, epoch):
    model.train()
    loss_meter = AverageMeter()
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None:
             scheduler.step()
        
        loss_meter.update(loss.item(), images.size(0))
        pbar.set_postfix(
            loss=f"{loss_meter.avg:.4f}", 
            lr=f"{optimizer.param_groups[0]['lr']:.6f}"
        )
        
    return loss_meter.avg

def validate(model, loader, criterion, device, epoch):
    model.eval()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)
    with torch.no_grad():
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
            
            pbar.set_postfix(
                loss=f"{loss_meter.avg:.4f}", 
                dice=f"{dice_meter.avg:.4f}",
                iou=f"{iou_meter.avg:.4f}",
                acc=f"{acc_meter.avg:.4f}"
            )
            
    return loss_meter.avg, dice_meter.avg, iou_meter.avg, acc_meter.avg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='.', help='Root directory')
    parser.add_argument('--arch', type=str, default='Unet', help='Model architecture')
    parser.add_argument('--encoder', type=str, default='resnet34', help='Encoder backbone')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--name', type=str, default='dental_model', help='Experiment name')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Starting Training: {args.name}")
    print(f"Device: {args.device} | Epochs: {args.epochs} | Batch: {args.batch_size} | Workers: {args.workers}")
    print(f"Model: {args.arch} ({args.encoder})")
    print(f"{'='*60}\n")
    
    # Data
    train_loader, val_loader, _ = get_dataloaders(args.root, batch_size=args.batch_size, num_workers=args.workers)
    
    # Model
    model = create_model(arch=args.arch, encoder_name=args.encoder)
    model.to(args.device)
    
    # Loss & Optimizer
    criterion = get_loss_function('BCEDice')
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.3
    )
    
    # Tools
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=os.path.join('logs', args.name))
    
    best_score = 0.0
    
    print(f"{'Epoch':^6} | {'Train Loss':^10} | {'Val Loss':^10} | {'Dice':^8} | {'IoU':^8} | {'Acc':^8} | {'Time'}")
    print("-" * 75)
    
    import time
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train(model, train_loader, criterion, optimizer, scheduler, args.device, scaler, epoch)
        
        # Validate
        val_loss, val_dice, val_iou, val_acc = validate(model, val_loader, criterion, args.device, epoch)
        
        epoch_time = time.time() - epoch_start
        
        # Log
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metric/Dice', val_dice, epoch)
        writer.add_scalar('Metric/IoU', val_iou, epoch)
        writer.add_scalar('Metric/Accuracy', val_acc, epoch)
        
        # Print Table Row
        print(f"{epoch:^6} | {train_loss:^10.4f} | {val_loss:^10.4f} | {val_dice:^8.4f} | {val_iou:^8.4f} | {val_acc:^8.4f} | {epoch_time:.0f}s")
        
        # Checkpoint
        is_best = val_dice > best_score
        if is_best:
            best_score = val_dice
            
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_score': best_score,
            'optimizer': optimizer.state_dict(),
        }, is_best, name=args.name)
        
    total_time = time.time() - start_time
    print(f"\nTraining Complete. Total time: {total_time/60:.1f} min")
    print(f"Best Dice Score: {best_score:.4f}")
    writer.close()

if __name__ == '__main__':
    main()
