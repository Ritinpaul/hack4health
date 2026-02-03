import torch
import argparse
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

from src.data_loader import get_dataloaders
from src.model import create_model
from src.metrics import dice_coefficient, jaccard_index, hausdorff_distance, sensitivity, specificity, pixel_accuracy
from src.utils import AverageMeter

def evaluate(model, loader, device):
    model.eval()
    
    metrics = {
        'Dice': [],
        'IoU': [],
        'Accuracy': [],
        'Hausdorff': [],
        'Sensitivity': [],
        'Specificity': []
    }
    
    print("Starting evaluation...")
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            
            logits = model(images)
            preds = torch.sigmoid(logits)
            
            for i in range(images.size(0)):
                pred_i = preds[i]
                mask_i = masks[i]
                
                d = dice_coefficient(pred_i, mask_i).item()
                iou = jaccard_index(pred_i, mask_i).item()
                acc = pixel_accuracy(pred_i, mask_i).item()
                sens = sensitivity(pred_i, mask_i).item()
                spec = specificity(pred_i, mask_i).item()
                hd = hausdorff_distance(pred_i, mask_i)
                
                metrics['Dice'].append(d)
                metrics['IoU'].append(iou)
                metrics['Accuracy'].append(acc)
                metrics['Hausdorff'].append(hd)
                metrics['Sensitivity'].append(sens)
                metrics['Specificity'].append(spec)
    
    # Create DataFrame
    df = pd.DataFrame(metrics)
    
    # Calculate Summary
    summary = df.mean().to_frame(name='Mean')
    summary['Std'] = df.std()
    
    return df, summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--root', type=str, default='.', help='Root directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=str, default='evaluation_report.csv', help='Output CSV file')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loader workers')
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location=args.device)
    
    # Create Model (assuming architecture is standard Unet-resnet34 as per plan, 
    # ideally config should be saved in checkpoint, but hardcoding for now as per train.py defaults)
    model = create_model(arch='Unet', encoder_name='resnet34')
    model.load_state_dict(checkpoint['state_dict'])
    model.to(args.device)
    
    # Get Test Loader
    _, _, test_loader = get_dataloaders(args.root, batch_size=1, num_workers=args.workers) # Batch size 1 for accurate per-sample analysis
    
    # Evaluate
    df, summary = evaluate(model, test_loader, args.device)
    
    print("\n" + "="*40)
    print("Evaluation Results (Test Set)")
    print("="*40)
    print(summary)
    print("="*40)
    
    df.to_csv(args.output, index=False)
    print(f"\nDetailed report saved to {args.output}")

if __name__ == '__main__':
    main()
