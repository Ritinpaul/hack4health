import torch
import numpy as np

def dice_coefficient(pred, target, smooth=1.0):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def jaccard_index(pred, target, smooth=1.0):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    total = pred_flat.sum() + target_flat.sum()
    union = total - intersection
    return (intersection + smooth) / (union + smooth)

def pixel_accuracy(pred, target):
    pred_mask = (pred > 0.5).float()
    correct = (pred_mask == target).sum()
    total = torch.numel(target)
    return correct / total

def hausdorff_distance(pred, target):
    """Computes Hausdorff Distance (requires binary masks)"""
    from scipy.spatial.distance import directed_hausdorff
    
    pred_np = (pred.squeeze().cpu().numpy() > 0.5).astype(bool)
    target_np = (target.squeeze().cpu().numpy() > 0.5).astype(bool)
    
    # If empty, return inf or max dist
    if not np.any(pred_np) or not np.any(target_np):
        return 0.0 # simplified for stability, or handle as error
        
    coordinates_pred = np.argwhere(pred_np)
    coordinates_target = np.argwhere(target_np)
    
    d1 = directed_hausdorff(coordinates_pred, coordinates_target)[0]
    d2 = directed_hausdorff(coordinates_target, coordinates_pred)[0]
    
    return max(d1, d2)

def sensitivity(pred, target):
    """Computes Sensitivity (Recall)"""
    pred_mask = (pred > 0.5).float()
    
    tp = (pred_mask * target).sum()
    fn = target.sum() - tp
    
    return (tp + 1e-6) / (tp + fn + 1e-6)

def specificity(pred, target):
    """Computes Specificity (True Negative Rate)"""
    pred_mask = (pred > 0.5).float()
    
    tn = ((1 - pred_mask) * (1 - target)).sum()
    fp = (1 - target).sum() - tn
    
    return (tn + 1e-6) / (tn + fp + 1e-6)
