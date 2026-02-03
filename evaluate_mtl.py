import torch
import argparse
import pandas as pd
from tqdm import tqdm

from src.mtl_dataset import get_mtl_dataloaders
from src.mtl_model import create_mtl_model
from src.mtl_metrics import MTLEvaluator

def evaluate_model(model, loader, device):
    """Evaluate model on dataset."""
    model.eval()
    evaluator = MTLEvaluator(device)
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating", ncols=100)
        for images, masks, labels in pbar:
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            seg_output, cls_output = model(images)
            evaluator.update(seg_output, cls_output, masks, labels)
    
    results = evaluator.compute()
    return results, evaluator

def main():
    parser = argparse.ArgumentParser(description='Evaluate Multi-Task Dental Model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--root', type=str, default='.', help='Root directory')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--workers', type=int, default=0, help='Data loader workers')
    parser.add_argument('--output', type=str, default='evaluation_results.csv', help='Output CSV file')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    _, val_loader = get_mtl_dataloaders(
        args.root, batch_size=args.batch_size, num_workers=args.workers
    )
    
    # Load model
    model = create_mtl_model()
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    print(f"Loaded model from: {args.model}")
    
    # Evaluate
    results, evaluator = evaluate_model(model, val_loader, device)
    
    # Print results
    evaluator.print_results(results)
    
    # Save to CSV
    seg_metrics = results['segmentation']
    cls_metrics = results['classification']
    
    data = {
        'Metric': [
            'Dice Score', 'IoU', 'Pixel Accuracy', 'Sensitivity', 'Specificity',
            'Cls Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'
        ],
        'Value': [
            seg_metrics['dice'],
            seg_metrics['iou'],
            seg_metrics['pixel_acc'],
            seg_metrics['sensitivity'],
            seg_metrics['specificity'],
            cls_metrics['accuracy'],
            cls_metrics['precision'],
            cls_metrics['recall'],
            cls_metrics['f1'],
            cls_metrics['auc']
        ]
    }
    
    if 'hausdorff' in seg_metrics:
        data['Metric'].append('Hausdorff Distance')
        data['Value'].append(seg_metrics['hausdorff'])
    
    df = pd.DataFrame(data)
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")

if __name__ == '__main__':
    main()
