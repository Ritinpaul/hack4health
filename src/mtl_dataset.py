import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from glob import glob
from src.augmentations import get_training_augmentation, get_validation_augmentation

class MTLDataset(Dataset):
    """
    Multi-Task Learning Dataset for Dental Caries Segmentation & Classification
    
    Returns:
        image: RGB image tensor (3, H, W)
        mask: Binary segmentation mask tensor (1, H, W)
        label: Classification label (0=Normal, 1=Caries)
    """
    def __init__(self, images_paths, masks_paths, labels, transform=None):
        self.images_paths = images_paths
        self.masks_paths = masks_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        mask_path = self.masks_paths[idx]
        label = self.labels[idx]

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (or create zero mask for Normal images)
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 127, 1.0, cv2.THRESH_BINARY)
        else:
            # Normal images: create zero mask
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        mask = mask.astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)

        return image, mask, label

def get_mtl_datasets(root_dir, image_size=(256, 256), split_ratio=0.8, seed=42):
    """
    Prepares multi-task learning datasets with stratified split.
    
    Args:
        root_dir: Path to hackHealth folder
        image_size: Target image size
        split_ratio: Train split ratio (default: 0.8 for 80-20)
        seed: Random seed
    
    Returns:
        train_ds, val_ds: MTLDataset instances
    """
    
    # Collect Caries images (label=1)
    caries_images = glob(os.path.join(root_dir, 'Carries', '*.png'))
    caries_images = [f for f in caries_images if 'mask' not in f]
    
    caries_data = []
    for img_path in caries_images:
        base, ext = os.path.splitext(img_path)
        mask_path = f"{base}-mask{ext}"
        
        if os.path.exists(mask_path):
            caries_data.append({
                'image': img_path,
                'mask': mask_path,
                'label': 1  # Caries
            })
        else:
            print(f"Warning: Mask not found for {img_path}, skipping.")
    
    # Collect Normal images (label=0)
    normal_images = glob(os.path.join(root_dir, 'Normal', '*.png'))
    normal_images = [f for f in normal_images if 'mask' not in f]
    
    normal_data = []
    for img_path in normal_images:
        normal_data.append({
            'image': img_path,
            'mask': None,  # No mask for normal images
            'label': 0  # Normal
        })
    
    # Combine all data
    all_data = caries_data + normal_data
    
    # Extract lists
    all_images = [d['image'] for d in all_data]
    all_masks = [d['mask'] for d in all_data]
    all_labels = [d['label'] for d in all_data]
    
    # Stratified split
    train_imgs, val_imgs, train_masks, val_masks, train_labels, val_labels = train_test_split(
        all_images, all_masks, all_labels, 
        train_size=split_ratio, 
        random_state=seed, 
        shuffle=True,
        stratify=all_labels  # Stratified by class
    )
    
    # Create Datasets
    train_ds = MTLDataset(
        train_imgs, train_masks, train_labels,
        transform=get_training_augmentation(image_size)
    )
    
    val_ds = MTLDataset(
        val_imgs, val_masks, val_labels,
        transform=get_validation_augmentation(image_size)
    )
    
    print(f"MTL Dataset split completed:")
    print(f"  Train: {len(train_ds)} images ({sum(train_labels)} Caries + {len(train_labels)-sum(train_labels)} Normal)")
    print(f"  Val:   {len(val_ds)} images ({sum(val_labels)} Caries + {len(val_labels)-sum(val_labels)} Normal)")
    
    return train_ds, val_ds

def get_mtl_dataloaders(root_dir, batch_size=8, num_workers=0, image_size=(256, 256)):
    """
    Creates DataLoaders for multi-task learning.
    
    Returns:
        train_loader, val_loader
    """
    train_ds, val_ds = get_mtl_datasets(root_dir, image_size)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader
