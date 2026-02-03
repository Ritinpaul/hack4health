import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class DentalModel(nn.Module):
    def __init__(self, arch='Unet', encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1):
        super(DentalModel, self).__init__()
        
        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None
        )
    
    def forward(self, x):
        return self.model(x)

def create_model(arch='Unet', encoder_name='resnet34', weights='imagenet'):
    return DentalModel(arch=arch, encoder_name=encoder_name, encoder_weights=weights)
