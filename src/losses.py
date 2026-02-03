import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)  
        
        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets):
        loss_bce = self.bce_loss(inputs, targets)
        loss_dice = self.dice_loss(inputs, targets)
        
        return self.bce_weight * loss_bce + (1 - self.bce_weight) * loss_dice

def get_loss_function(name='BCEDice'):
    if name == 'Dice':
        return DiceLoss()
    elif name == 'BCE':
        return nn.BCEWithLogitsLoss()
    elif name == 'BCEDice':
        return BCEDiceLoss()
    else:
        raise ValueError(f"Unknown loss function: {name}")
