import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_training_augmentation(image_size=(256, 256)):
    train_transform = [
        A.Resize(height=image_size[0], width=image_size[1]),
        
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, 
                           border_mode=0, p=0.7),
        A.ElasticTransform(alpha=120, sigma=6, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        
        # X-ray specific enhancements
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1),
            A.GaussianBlur(blur_limit=(3, 5), p=1),
            A.MedianBlur(blur_limit=5, p=1),
        ], p=0.4),
        
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
            A.RandomGamma(gamma_limit=(80, 120), p=1),
        ], p=0.5),
        
        # Normalize and convert
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]

    return A.Compose(train_transform)

def get_validation_augmentation(image_size=(256, 256)):
    test_transform = [
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]

    return A.Compose(test_transform)
