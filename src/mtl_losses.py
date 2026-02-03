import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Segmentation Losses
# ============================================================================

class DiceLoss(nn.Module):
    """Dice Loss for segmentation."""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class TverskyLoss(nn.Module):
    """Tversky Loss - handles class imbalance by weighting FP and FN differently."""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # FP weight
        self.beta = beta    # FN weight (higher = penalize missing caries more)
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        tp = (inputs * targets).sum()
        fp = ((1 - targets) * inputs).sum()
        fn = (targets * (1 - inputs)).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky

class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss - adds focal term to focus on hard examples."""
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1.0):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        tp = (inputs * targets).sum()
        fp = ((1 - targets) * inputs).sum()
        fn = (targets * (1 - inputs)).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma
        return focal_tversky

class SegmentationLoss(nn.Module):
    """Combined segmentation loss: FocalTversky + Dice."""
    def __init__(self, focal_weight=0.5, dice_weight=0.5):
        super(SegmentationLoss, self).__init__()
        self.focal_tversky = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        focal_loss = self.focal_tversky(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss

# ============================================================================
# Classification Loss
# ============================================================================

class ClassificationLoss(nn.Module):
    """Binary Cross-Entropy with Logits for classification."""
    def __init__(self):
        super(ClassificationLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # inputs: (B, 1), targets: (B,) -> reshape targets to (B, 1)
        targets = targets.view(-1, 1).float()
        return self.bce(inputs, targets)

# ============================================================================
# Multi-Task Loss
# ============================================================================

class MTLLoss(nn.Module):
    """
    Multi-Task Learning Loss combining segmentation and classification.
    
    Args:
        seg_weight: Weight for segmentation loss (default: 0.7)
        cls_weight: Weight for classification loss (default: 0.3)
    """
    def __init__(self, seg_weight=0.7, cls_weight=0.3):
        super(MTLLoss, self).__init__()
        self.seg_loss_fn = SegmentationLoss()
        self.cls_loss_fn = ClassificationLoss()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight

    def forward(self, seg_output, cls_output, seg_target, cls_target):
        """
        Args:
            seg_output: Segmentation predictions (B, 1, H, W)
            cls_output: Classification predictions (B, 1)
            seg_target: Segmentation ground truth (B, 1, H, W)
            cls_target: Classification labels (B,)
        
        Returns:
            total_loss, seg_loss, cls_loss (for logging)
        """
        seg_loss = self.seg_loss_fn(seg_output, seg_target)
        cls_loss = self.cls_loss_fn(cls_output, cls_target)
        
        total_loss = self.seg_weight * seg_loss + self.cls_weight * cls_loss
        
        return total_loss, seg_loss, cls_loss

# ============================================================================
# Factory Function
# ============================================================================

def create_mtl_loss(seg_weight=0.7, cls_weight=0.3):
    """Create multi-task loss with specified weights."""
    return MTLLoss(seg_weight=seg_weight, cls_weight=cls_weight)
