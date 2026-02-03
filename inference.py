import argparse
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
import albumentations as A

from src.model import create_model

def get_inference_transform(image_size=(256, 256)):
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def visualize_prediction(image, mask, pred, save_path=None):
    overlay = image.copy()
    
    pred_uint8 = (pred * 255).astype(np.uint8)
    contours, _ = cv2.findContours(pred_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
    
    red_mask = np.zeros_like(image)
    red_mask[pred > 0.5] = [255, 0, 0]
    overlay = cv2.addWeighted(overlay, 1.0, red_mask, 0.3, 0)

    if mask is not None:
         mask_uint8 = (mask * 255).astype(np.uint8)
         gt_contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
         cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), 2)
    
    # Plotting
    cols = 3 if mask is not None else 2
    fig, axes = plt.subplots(1, cols, figsize=(cols*5, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    if mask is not None:
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
    axes[cols-1].imshow(overlay)
    axes[cols-1].set_title('Prediction (Red) vs GT (Green)')
    axes[cols-1].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def inference(model, image_path, device, transform, output_dir):
    # Load image
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    augmented = transform(image=original_image)
    input_tensor = augmented['image'].unsqueeze(0).to(device) # [1, 3, H, W]
    
    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        prob_map = torch.sigmoid(logits)
        pred_mask = (prob_map > 0.5).float().cpu().numpy().squeeze() # [H, W]
    
    # Resize prediction back to original size if needed (skipped for now as we resize visual too)
    # Actually, for visualization we usually verify on resized image or resize pred back.
    # Let's resize visualization components to 256x256 for consistency with input
    vis_image = cv2.resize(original_image, (256, 256))
    
    # Check for Ground Truth
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    mask_path = image_path.replace(name, f"{name}-mask") # Standard naming check
    
    gt_mask = None
    if os.path.exists(mask_path):
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.resize(gt_mask, (256, 256))
        gt_mask = (gt_mask > 127).astype(np.float32)

    # Save
    save_name = os.path.join(output_dir, f"pred_{base_name}")
    visualize_prediction(vis_image, gt_mask, pred_mask, save_path=save_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to image or directory')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location=args.device)
    model = create_model(arch='Unet', encoder_name='resnet34')
    model.load_state_dict(checkpoint['state_dict'])
    model.to(args.device)
    
    transform = get_inference_transform()
    
    if os.path.isdir(args.input):
        images = glob(os.path.join(args.input, '*.png'))
        # Filter out masks
        images = [f for f in images if 'mask' not in f]
        print(f"Found {len(images)} images in {args.input}")
    else:
        images = [args.input]
        
    print("Running inference...")
    for img_path in tqdm(images):
        inference(model, img_path, args.device, transform, args.output)
        
    print(f"Results saved to {args.output}")

if __name__ == '__main__':
    main()
