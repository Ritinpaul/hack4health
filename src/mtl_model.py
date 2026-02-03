import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class MTLModel(nn.Module):
    """
    Multi-Task Learning Model for Dental Caries Segmentation & Classification
    
    Architecture:
        - Shared Encoder: EfficientNet-B4 (ImageNet pretrained)
        - Segmentation Head: UNet++ Decoder
        - Classification Head: Global Average Pooling + FC Layers
    
    Outputs:
        seg_output: Segmentation logits (B, 1, H, W)
        cls_output: Classification logits (B, 1)
    """
    def __init__(self, 
                 encoder_name='efficientnet-b4',
                 encoder_weights='imagenet',
                 in_channels=3,
                 seg_classes=1,
                 cls_classes=1):
        super(MTLModel, self).__init__()
        
        # Segmentation branch (UNet++ with shared encoder)
        self.segmentation_model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=seg_classes,
            activation=None,  # We'll apply sigmoid later
            decoder_attention_type='scse'  # Spatial and Channel Squeeze-Excitation
        )
        
        # Get encoder output channels for classification head
        # EfficientNet-B4 encoder output: 1792 channels
        encoder_channels = self.segmentation_model.encoder.out_channels[-1]
        
        # Classification head
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(encoder_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, cls_classes)  # Binary classification
        )
    
    def forward(self, x):
        # Get segmentation output (full forward pass through UNet++)
        seg_output = self.segmentation_model(x)
        
        # For classification, extract encoder features separately
        encoder_features = self.segmentation_model.encoder(x)
        
        # Classification branch (using deepest encoder features)
        cls_features = encoder_features[-1]  # Deepest feature map
        cls_pooled = self.gap(cls_features)
        cls_output = self.classification_head(cls_pooled)
        
        return seg_output, cls_output

def create_mtl_model(encoder_name='efficientnet-b4', 
                     encoder_weights='imagenet',
                     in_channels=3,
                     seg_classes=1,
                     cls_classes=1):
    """
    Factory function to create MTL model.
    
    Returns:
        MTLModel instance
    """
    model = MTLModel(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        seg_classes=seg_classes,
        cls_classes=cls_classes
    )
    return model

# Model Summary Helper
def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    # Test model creation
    model = create_mtl_model()
    print(f"MTL Model created successfully!")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 256, 256)
    seg_out, cls_out = model(dummy_input)
    print(f"Segmentation output shape: {seg_out.shape}")  # (2, 1, 256, 256)
    print(f"Classification output shape: {cls_out.shape}")  # (2, 1)
