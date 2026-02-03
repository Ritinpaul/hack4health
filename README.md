# Dental Caries Segmentation

A deep learning-based system for automated dental caries detection and segmentation using semantic segmentation techniques.

## Overview

This project implements a U-Net based segmentation model with ResNet34 encoder to identify and segment caries regions in dental radiographs. The system is designed to assist dental professionals in early detection and diagnosis of dental caries.

## Features

- **Automated Segmentation**: Pixel-level caries detection using state-of-the-art deep learning
- **Data Augmentation**: Comprehensive augmentation pipeline for robust model training
- **Multiple Metrics**: Dice coefficient, IoU, pixel accuracy, sensitivity, and specificity
- **Visualization**: Interactive prediction visualization with ground truth comparison
- **Model Checkpointing**: Automatic saving of best performing models

## Project Structure

```
├── src/
│   ├── augmentations.py    # Data augmentation pipelines
│   ├── data_loader.py       # Dataset and dataloader utilities
│   ├── losses.py            # Loss functions (Dice, BCE+Dice)
│   ├── metrics.py           # Evaluation metrics
│   ├── model.py             # Model architecture
│   └── utils.py             # Helper utilities
├── train.py                 # Training script
├── evaluate.py              # Model evaluation
├── inference.py             # Inference and visualization
├── requirements.txt         # Project dependencies
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone git@github.com:Ritinpaul/hack4health.git
cd hack4health
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure

Organize your dataset as follows:
```
hackHealth/
├── Carries/
│   ├── image1.png
│   ├── image1-mask.png
│   └── ...
└── Normal/
    ├── image2.png
    ├── image2-mask.png
    └── ...
```

## Usage

### Training

Train the model with default parameters:
```bash
python train.py --root . --epochs 50 --batch-size 8
```

Available training arguments:
- `--root`: Root directory containing dataset (default: current directory)
- `--arch`: Model architecture (default: Unet)
- `--encoder`: Encoder backbone (default: resnet34)
- `--epochs`: Number of training epochs (default: 5)
- `--batch-size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 1e-4)
- `--workers`: Number of data loader workers (default: 0)

### Evaluation

Evaluate a trained model:
```bash
python evaluate.py --model checkpoints/dental_model_best.pth --root . --output results.csv
```

### Inference

Run inference on new images:
```bash
python inference.py --model checkpoints/dental_model_best.pth --input path/to/image.png --output predictions/
```

## Model Architecture

- **Encoder**: ResNet34 (ImageNet pretrained)
- **Decoder**: U-Net architecture
- **Loss Function**: Combined BCE and Dice Loss
- **Optimizer**: AdamW with OneCycleLR scheduler
- **Input Size**: 256×256 RGB images

## Performance Metrics

The model is evaluated using:
- Dice Coefficient
- Intersection over Union (IoU)
- Pixel Accuracy
- Sensitivity (Recall)
- Specificity
- Hausdorff Distance

## Requirements

- Python 3.8+
- PyTorch 1.10+
- segmentation-models-pytorch
- albumentations
- OpenCV
- NumPy
- scikit-learn
- pandas
- matplotlib

See `requirements.txt` for complete dependencies.

## Acknowledgments

- Built using [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- Data augmentation powered by [Albumentations](https://albumentations.ai/)

## License

This project is developed for educational and research purposes.
| **Dice Score** | 0.027 | Precision needs improvement (imbalanced data) |
| **Hausdorff** | 22.09 | Average boundary error distance |

*Note: Sensitivity is the critical metric for medical screening. 64% is a strong baseline for a short training run.*

## 📁 Project Structure

```
hackHealth/
├── checkpoints/       # Saved models (*.pth)
├── logs/              # TensorBoard logs
├── src/               # Source code
│   ├── data_loader.py # Dataset class & splitting
│   ├── model.py       # U-Net definition
│   ├── losses.py      # BCEDiceLoss
│   ├── metrics.py     # Evaluation metrics
│   └── augmentations.py # Albumentations pipeline
├── train.py           # Training script
├── evaluate.py        # Evaluation script
├── inference.py       # Batch inference script
├── streamlit_app.py   # Web demo app
└── requirements.txt   # Python dependencies
```
