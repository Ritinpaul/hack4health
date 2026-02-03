import torch.nn as nn
import segmentation_models_pytorch as smp

class DentalModel(nn.Module):
    def __init__(self, arch='UnetPlusPlus', encoder_name='efficientnet-b4', 
                 encoder_weights='imagenet', in_channels=3, classes=1):
        super(DentalModel, self).__init__()
        
        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
            decoder_attention_type='scse'  # Spatial and Channel Squeeze-Excitation
        )
    
    def forward(self, x):
        return self.model(x)

def create_model(arch='UnetPlusPlus', encoder_name='efficientnet-b4', 
                 encoder_weights='imagenet', in_channels=3, classes=1):
    return DentalModel(arch, encoder_name, encoder_weights, in_channels, classes)
