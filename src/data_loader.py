import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from glob import glob
from .augmentations import get_training_augmentation, get_validation_augmentation

class DentalDataset(Dataset):
    def __init__(self, images_paths, masks_paths, transform=None):
        self.images_paths = images_paths
        self.masks_paths = masks_paths
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        mask_path = self.masks_paths[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 1.0, cv2.THRESH_BINARY)
        mask = mask.astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)

        return image, mask

def get_datasets(root_dir, image_size=(256, 256), split_ratio=(0.8, 0.2), seed=42):
    caries_images = glob(os.path.join(root_dir, 'Carries', '*.png'))
    caries_images = [f for f in caries_images if 'mask' not in f]
    
    normal_images = glob(os.path.join(root_dir, 'Normal', '*.png'))
    normal_images = [f for f in normal_images if 'mask' not in f]

    all_images = caries_images + normal_images
    all_masks = []
    valid_images = []
    
    for img_path in all_images:
        base, ext = os.path.splitext(img_path)
        mask_path = f"{base}-mask{ext}"
        
        if os.path.exists(mask_path):
            valid_images.append(img_path)
            all_masks.append(mask_path)
        else:
            print(f"Warning: Mask not found for {img_path}, skipping.")

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        valid_images, all_masks, train_size=split_ratio[0], random_state=seed, shuffle=True
    )

    train_ds = DentalDataset(
        train_imgs, train_masks, 
        transform=get_training_augmentation(image_size)
    )
    
    val_ds = DentalDataset(
        val_imgs, val_masks, 
        transform=get_validation_augmentation(image_size)
    )
    
    # Use val as test for compatibility
    test_ds = val_ds

    print(f"Dataset split completed:")
    print(f"Train: {len(train_ds)} images")
    print(f"Val: {len(val_ds)} images")
    
    return train_ds, val_ds, test_ds

def get_dataloaders(root_dir, batch_size=16, num_workers=4, image_size=(256, 256)):
    train_ds, val_ds, test_ds = get_datasets(root_dir, image_size)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader
