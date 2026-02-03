import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# ============================================================================
# Segmentation Metrics (already in src/metrics.py, importing here)
# ============================================================================

from src.metrics import (
    dice_coefficient,
    jaccard_index,
    pixel_accuracy,
    hausdorff_distance,
    sensitivity,
    specificity
)

# ============================================================================
# Classification Metrics
# ============================================================================

def calculate_classification_metrics(outputs, targets, threshold=0.5):
    """
    Calculate comprehensive classification metrics.
    
    Args:
        outputs: Model outputs (logits), shape (N, 1) or (N,)
        targets: Ground truth labels (0 or 1), shape (N,)
        threshold: Classification threshold (default: 0.5)
    
    Returns:
        dict with accuracy, precision, recall, f1, auc
    """
    # Convert to numpy
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu()
        probs = torch.sigmoid(outputs).numpy().flatten()
    else:
        probs = outputs.flatten()
    
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy().flatten()
    else:
        targets = targets.flatten()
    
    # Binary predictions
    preds = (probs > threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(targets, preds),
        'precision': precision_score(targets, preds, zero_division=0),
        'recall': recall_score(targets, preds, zero_division=0),
        'f1': f1_score(targets, preds, zero_division=0),
    }
    
    # AUC (only if both classes present)
    if len(np.unique(targets)) > 1:
        metrics['auc'] = roc_auc_score(targets, probs)
    else:
        metrics['auc'] = 0.0
    
    return metrics

def calculate_confusion_matrix(outputs, targets, threshold=0.5):
    """Calculate confusion matrix."""
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu()
        probs = torch.sigmoid(outputs).numpy().flatten()
    else:
        probs = outputs.flatten()
    
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy().flatten()
    else:
        targets = targets.flatten()
    
    preds = (probs > threshold).astype(int)
    cm = confusion_matrix(targets, preds)
    
    return cm

# ============================================================================
# Multi-Task Evaluation
# ============================================================================

class MTLEvaluator:
    """Comprehensive multi-task evaluator."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset all accumulators."""
        # Segmentation metrics
        self.dice_scores = []
        self.iou_scores = []
        self.pixel_accs = []
        self.sensitivities = []
        self.specificities = []
        self.hausdorff_dists = []
        
        # Classification metrics
        self.cls_outputs = []
        self.cls_targets = []
    
    def update(self, seg_output, cls_output, seg_target, cls_target):
        """
        Update metrics with a batch.
        
        Args:
            seg_output: Segmentation predictions (B, 1, H, W)
            cls_output: Classification predictions (B, 1)
            seg_target: Segmentation ground truth (B, 1, H, W)
            cls_target: Classification labels (B,)
        """
        seg_preds = torch.sigmoid(seg_output)
        
        # Calculate segmentation metrics per sample
        batch_size = seg_output.size(0)
        for i in range(batch_size):
            pred_i = seg_preds[i]
            target_i = seg_target[i]
            
            self.dice_scores.append(dice_coefficient(pred_i, target_i).item())
            self.iou_scores.append(jaccard_index(pred_i, target_i).item())
            self.pixel_accs.append(pixel_accuracy(pred_i, target_i).item())
            self.sensitivities.append(sensitivity(pred_i, target_i).item())
            self.specificities.append(specificity(pred_i, target_i).item())
            
            # Hausdorff (skip if causes issues)
            try:
                hd = hausdorff_distance(pred_i, target_i)
                self.hausdorff_dists.append(hd)
            except:
                pass
        
        # Accumulate classification outputs
        self.cls_outputs.append(cls_output.detach().cpu())
        self.cls_targets.append(cls_target.detach().cpu())
    
    def compute(self):
        """Compute final metrics."""
        # Segmentation metrics
        seg_metrics = {
            'dice': np.mean(self.dice_scores),
            'iou': np.mean(self.iou_scores),
            'pixel_acc': np.mean(self.pixel_accs),
            'sensitivity': np.mean(self.sensitivities),
            'specificity': np.mean(self.specificities),
        }
        
        if len(self.hausdorff_dists) > 0:
            seg_metrics['hausdorff'] = np.mean(self.hausdorff_dists)
        
        # Classification metrics
        all_cls_outputs = torch.cat(self.cls_outputs, dim=0)
        all_cls_targets = torch.cat(self.cls_targets, dim=0)
        
        cls_metrics = calculate_classification_metrics(all_cls_outputs, all_cls_targets)
        cm = calculate_confusion_matrix(all_cls_outputs, all_cls_targets)
        
        cls_metrics['confusion_matrix'] = cm
        
        return {
            'segmentation': seg_metrics,
            'classification': cls_metrics
        }
    
    def print_results(self, results):
        """Print formatted results."""
        print("\n" + "=" * 80)
        print("SEGMENTATION METRICS")
        print("-" * 80)
        seg = results['segmentation']
        print(f"  Dice Score:      {seg['dice']:.4f}")
        print(f"  IoU (Jaccard):   {seg['iou']:.4f}")
        print(f"  Pixel Accuracy:  {seg['pixel_acc']:.4f}")
        print(f"  Sensitivity:     {seg['sensitivity']:.4f}")
        print(f"  Specificity:     {seg['specificity']:.4f}")
        if 'hausdorff' in seg:
            print(f"  Hausdorff Dist:  {seg['hausdorff']:.2f} pixels")
        
        print("\n" + "=" * 80)
        print("CLASSIFICATION METRICS")
        print("-" * 80)
        cls = results['classification']
        print(f"  Accuracy:        {cls['accuracy']:.4f} ({cls['accuracy']*100:.2f}%)")
        print(f"  Precision:       {cls['precision']:.4f}")
        print(f"  Recall:          {cls['recall']:.4f}")
        print(f"  F1-Score:        {cls['f1']:.4f}")
        print(f"  AUC-ROC:         {cls['auc']:.4f}")
        
        print("\n  Confusion Matrix:")
        cm = cls['confusion_matrix']
        print(f"    TN: {cm[0,0]:4d}  |  FP: {cm[0,1]:4d}")
        print(f"    FN: {cm[1,0]:4d}  |  TP: {cm[1,1]:4d}")
        print("=" * 80 + "\n")
