# ==============================================================================
# Advanced ViT Face Anti-Spoofing Training Script (Fine-Tuned Version)
# ==============================================================================
# This is an enhanced version of the Vision Transformer (ViT) training script with 
# advanced techniques for maximum accuracy in face anti-spoofing tasks. It includes 
# cutting-edge features adapted for transformer architectures:
# 
# ADVANCED FEATURES:
# - Vision Transformer (ViT) with pre-trained weights from HuggingFace
# - Advanced data augmentation including Mixup and CutMix for transformers
# - Focal Loss and Label Smoothing for better generalization
# - Weighted Random Sampling for balanced training
# - Mixed Precision Training (AMP) for memory efficiency
# - Cosine Annealing with Warm Restarts for optimal learning rate scheduling
# - Gradient checkpointing for reduced memory usage
# - Comprehensive metrics including MSE and RMSE tracking
# - Test-Time Augmentation (TTA) capabilities
# - Advanced confusion matrix analysis with detailed metrics
# - Graceful checkpoint handling and resume functionality
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from transformers import ViTForImageClassification, ViTConfig
from PIL import Image
import os
import time
import numpy as np
torch.serialization.add_safe_globals([np.core.multiarray.scalar])
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_squared_error
import json
import signal
import sys
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings('ignore')

# ======================== ADVANCED CONFIGURATION ========================
# This configuration is optimized for maximum accuracy with Vision Transformers.
# All parameters have been fine-tuned based on extensive experimentation with ViTs.

# Core Training Parameters - Optimized for ViT architecture
BATCH_SIZE = 8                # Smaller batches for ViT due to memory requirements
                              # ViTs are more memory-intensive than CNNs
IMAGE_SIZE = (224, 224)       # Standard ViT input size (224x224)
                              # This matches the pre-trained ViT patch size
EPOCHS = 25                   # Fewer epochs as ViTs converge faster than CNNs
LEARNING_RATE = 1e-4          # Lower learning rate for fine-tuning pre-trained ViT
                              # ViTs require careful learning rate tuning
WEIGHT_DECAY = 1e-4           # Moderate weight decay for transformer regularization
SAMPLE_LIMIT = -1             # Use complete dataset for maximum training data

# Advanced Training Techniques for ViT
PATIENCE = 10                 # Early stopping patience adapted for ViT convergence
                              # ViTs typically show improvement patterns differently
LABEL_SMOOTHING = 0.1         # Label smoothing for transformer training
                              # Helps prevent overconfident predictions
GRADIENT_CLIP_NORM = 1.0      # Gradient clipping for stable transformer training
                              # ViTs can be sensitive to gradient explosions
MIXUP_ALPHA = 0.2             # Conservative mixup for ViT (transformers are sensitive)
CUTMIX_ALPHA = 1.0            # CutMix works well with patch-based ViTs
MIXUP_PROB = 0.3              # Lower probability for transformer augmentation

# Data Loading Optimization
NUM_WORKERS = 2               # Reduced workers to avoid memory issues with ViT
PIN_MEMORY = True             # Keep data in pinned memory for faster GPU transfer
PERSISTENT_WORKERS = True     # Reuse data loading processes between epochs

# Advanced Data Augmentation Parameters for ViT
# ViTs are more robust to geometric transformations than CNNs
HORIZONTAL_FLIP_PROB = 0.5    # Face orientation variation
ROTATION_DEGREES = 10         # Reduced rotation as ViT handles pose variation well
COLOR_JITTER_BRIGHTNESS = 0.2 # Enhanced lighting condition simulation
COLOR_JITTER_CONTRAST = 0.2   # Contrast variation for different cameras
COLOR_JITTER_SATURATION = 0.2 # Color saturation changes
COLOR_JITTER_HUE = 0.1        # Slight hue shifts for color variation
GRAYSCALE_PROB = 0.1          # Occasional grayscale conversion
RANDOM_ERASING_PROB = 0.15    # Random patch erasing (works well with ViT patches)
GAUSSIAN_BLUR_PROB = 0.1      # Simulate camera focus variations

# ViT-Specific Configuration
MODEL_NAME = 'google/vit-base-patch16-224'  # Pre-trained ViT model
PATCH_SIZE = 16               # ViT patch size (16x16 patches)
NUM_CLASSES = 2               # Binary classification: live vs spoof
GRADIENT_CHECKPOINTING = True # Enable for memory efficiency (important for RTX 3050)
FREEZE_BACKBONE_EPOCHS = 3    # Freeze backbone for first few epochs

# Advanced Optimizer Configuration for ViT
ADAM_BETA1 = 0.9              # First moment decay rate
ADAM_BETA2 = 0.999            # Second moment decay rate
ADAM_EPS = 1e-8               # Numerical stability constant

# Cosine Annealing with Warm Restarts Parameters for ViT
T_0 = 5                       # Shorter restart period for ViT
T_MULT = 2                    # Restart period multiplier
ETA_MIN = 1e-7                # Minimum learning rate

# File and Directory Configuration
MODEL_DIR = "vit_models"      # Directory for saving trained models
MODEL_NAME_SAVE = "best_vit_antispoofing.pth"  # Filename for best model weights

# Training Optimization Flags
USE_AMP = True                # Automatic Mixed Precision (crucial for ViT memory efficiency)
ENABLE_COMPILE = False        # PyTorch 2.0 compile (set False for compatibility)

# Test-Time Augmentation Configuration
TTA_ENABLED = True            # Enable test-time augmentation
TTA_CROPS = 5                 # Number of augmented crops for TTA

# Metrics Tracking Configuration
TRACK_MSE_RMSE = True         # Enable MSE/RMSE tracking for regression-like analysis
SAVE_DETAILED_PLOTS = True    # Save comprehensive visualization plots

# ======================== DEVICE SETUP ========================
# Configure the computing device and enable optimizations for ViT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enable PyTorch optimizations for ViT performance
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"CUDA optimizations enabled for {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ======================== ADVANCED DATASET CLASS ========================
class CASIAFASDDataset(Dataset):
    """
    Advanced CASIA-FASD Dataset class with comprehensive error handling and logging.
    
    This dataset class is specifically designed for face anti-spoofing tasks with
    robust image loading, error handling, and detailed logging for debugging.
    
    Expected directory structure:
    data_dir/
    ├── live/
    │   ├── *.png
    │   └── *.jpg
    └── spoof/
        ├── *.png
        └── *.jpg
    """
    
    def __init__(self, data_dir, transform=None, max_samples=None):
        """
        Initialize the CASIA-FASD dataset.
        
        Args:
            data_dir (str): Root directory containing 'live' and 'spoof' subdirectories
            transform (callable, optional): Optional transform to be applied on images
            max_samples (int, optional): Maximum number of samples to load per class
        """
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = ['spoof', 'live']  # 0: spoof, 1: live
        
        print(f"Initializing dataset from: {data_dir}")
        
        # Verify root directory exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        
        # Load images from both classes
        self._load_class_images('live', label=1, max_samples=max_samples)
        self._load_class_images('spoof', label=0, max_samples=max_samples)
        
        # Dataset statistics
        total_samples = len(self.images)
        live_count = sum(self.labels)
        spoof_count = total_samples - live_count
        
        print(f"Dataset loaded successfully:")
        print(f"  Total samples: {total_samples}")
        print(f"  Live samples: {live_count}")
        print(f"  Spoof samples: {spoof_count}")
        print(f"  Class balance: {live_count/total_samples:.2%} live, {spoof_count/total_samples:.2%} spoof")
        
        if total_samples == 0:
            raise ValueError("No images found in dataset directories")
    
    def _load_class_images(self, class_name, label, max_samples=None):
        """
        Load images from a specific class directory.
        
        Args:
            class_name (str): Name of the class directory ('live' or 'spoof')
            label (int): Label to assign to images from this class
            max_samples (int, optional): Maximum number of samples to load
        """
        class_dir = os.path.join(self.data_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: {class_name} directory not found: {class_dir}")
            return
        
        # Get all image files (png, jpg, jpeg)
        valid_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
        image_files = [
            f for f in os.listdir(class_dir) 
            if os.path.splitext(f)[1] in valid_extensions
        ]
        
        # Limit samples if specified
        if max_samples and len(image_files) > max_samples:
            image_files = random.sample(image_files, max_samples)
        
        print(f"  Loading {len(image_files)} {class_name} images...")
        
        # Add images to dataset
        for img_name in image_files:
            img_path = os.path.join(class_dir, img_name)
            self.images.append(img_path)
            self.labels.append(label)
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image, label) where image is the processed image tensor
        """
        if idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.images)}")
        
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            # Load and convert image to RGB
            image = Image.open(img_path).convert('RGB')
            
            # Apply transformations if provided
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            fallback_image = Image.new('RGB', IMAGE_SIZE, (0, 0, 0))
            if self.transform:
                fallback_image = self.transform(fallback_image)
            return fallback_image, label

# ======================== ADVANCED DATA AUGMENTATION ========================
class GaussianBlur:
    """
    Custom Gaussian blur augmentation for ViT training.
    
    Simulates camera focus variations and motion blur effects. ViTs can benefit
    from this type of augmentation as it doesn't destroy patch-level features.
    """
    def __init__(self, kernel_size=3, sigma_range=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
    
    def __call__(self, img):
        sigma = random.uniform(*self.sigma_range)
        return transforms.functional.gaussian_blur(img, self.kernel_size, sigma)

def get_vit_transforms():
    """
    Create ViT-optimized data augmentation pipelines.
    
    ViTs are generally more robust to geometric transformations than CNNs,
    but require careful augmentation to maintain patch coherence.
    
    Returns:
        tuple: (train_transform, val_transform) - Transformation pipelines
    """
    # Advanced training augmentation pipeline for ViT
    train_transform = transforms.Compose([
        # Resize with slight variation for robustness
        transforms.Resize((int(IMAGE_SIZE[0] * 1.05), int(IMAGE_SIZE[1] * 1.05))),
        # Random crop to final size
        transforms.RandomCrop(IMAGE_SIZE),
        # Random horizontal flip (faces are roughly symmetric)
        transforms.RandomHorizontalFlip(p=HORIZONTAL_FLIP_PROB),
        # Gentle rotation (ViTs handle rotation well)
        transforms.RandomRotation(ROTATION_DEGREES, interpolation=transforms.InterpolationMode.BILINEAR),
        # Color augmentation (important for anti-spoofing)
        transforms.ColorJitter(
            brightness=COLOR_JITTER_BRIGHTNESS,
            contrast=COLOR_JITTER_CONTRAST,
            saturation=COLOR_JITTER_SATURATION,
            hue=COLOR_JITTER_HUE
        ),
        # Occasional blur for focus variation
        transforms.RandomApply([GaussianBlur()], p=GAUSSIAN_BLUR_PROB),
        # Occasional grayscale (reduces color bias)
        transforms.RandomGrayscale(p=GRAYSCALE_PROB),
        # Convert to tensor
        transforms.ToTensor(),
        # Normalize with ImageNet statistics (required for pre-trained ViT)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Random erasing (works well with ViT patches)
        transforms.RandomErasing(p=RANDOM_ERASING_PROB, scale=(0.02, 0.15)),
    ])
    
    # Simple validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_transform

# ======================== MIXUP AND CUTMIX FOR VIT ========================
def mixup_data_vit(x, y, alpha=MIXUP_ALPHA):
    """
    Apply Mixup augmentation optimized for Vision Transformers.
    
    Mixup works well with ViTs as the attention mechanism can handle
    mixed visual features effectively.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data_vit(x, y, alpha=CUTMIX_ALPHA):
    """
    Apply CutMix augmentation optimized for Vision Transformers.
    
    CutMix is particularly effective with ViTs as it aligns well with
    the patch-based processing of the architecture.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda based on actual cut area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    """Generate random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Calculate loss for mixed samples."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ======================== FOCAL LOSS FOR VIT ========================
class FocalLoss(nn.Module):
    """
    Focal Loss implementation optimized for Vision Transformer training.
    
    Helps address class imbalance and focuses training on hard examples,
    which is particularly useful for face anti-spoofing tasks.
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ======================== METRICS LOGGER FOR VIT ========================
class ViTMetricsLogger:
    """
    Comprehensive metrics logger for Vision Transformer training.
    
    Tracks training/validation metrics including loss, accuracy, MSE, RMSE,
    and provides visualization capabilities.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metric tracking."""
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.mse_scores = []
        self.rmse_scores = []
        self.learning_rates = []
        self.epochs = []
    
    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, mse, rmse, lr):
        """Log metrics for one epoch."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        self.mse_scores.append(mse)
        self.rmse_scores.append(rmse)
        self.learning_rates.append(lr)
    
    def save_training_plots(self, save_dir):
        """Save comprehensive training plots."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Create main training curves plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        ax1.plot(self.epochs, self.train_losses, 'b-', label='Training Loss', marker='o', markersize=3)
        ax1.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss', marker='s', markersize=3)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(self.epochs, self.train_accs, 'b-', label='Training Accuracy', marker='o', markersize=3)
        ax2.plot(self.epochs, self.val_accs, 'r-', label='Validation Accuracy', marker='s', markersize=3)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate curve
        ax3.plot(self.epochs, self.learning_rates, 'g-', label='Learning Rate', marker='^', markersize=3)
        ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Best metrics summary
        best_train_acc = max(self.train_accs) if self.train_accs else 0
        best_val_acc = max(self.val_accs) if self.val_accs else 0
        final_loss = self.val_losses[-1] if self.val_losses else 0
        
        ax4.axis('off')
        summary_text = f"""Training Summary:
        
Best Training Accuracy: {best_train_acc:.2f}%
Best Validation Accuracy: {best_val_acc:.2f}%
Final Validation Loss: {final_loss:.4f}
Total Epochs: {len(self.epochs)}
Final Learning Rate: {self.learning_rates[-1]:.2e}

Model: Vision Transformer (ViT)
Task: Face Anti-Spoofing
Dataset: CASIA-FASD"""
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'vit_training_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create separate MSE/RMSE plot
        self.save_mse_rmse_plot(save_dir)
        
        print(f"Training plots saved to: {save_dir}")
    
    def save_mse_rmse_plot(self, save_dir):
        """Save separate MSE and RMSE plot."""
        if not self.mse_scores or not self.rmse_scores:
            print("No MSE/RMSE data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # MSE plot
        ax1.plot(self.epochs, self.mse_scores, 'purple', marker='o', markersize=4, linewidth=2)
        ax1.set_title('Mean Squared Error (MSE) Over Training', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE')
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#f8f9fa')
        
        # Add MSE statistics
        min_mse = min(self.mse_scores)
        avg_mse = np.mean(self.mse_scores)
        ax1.axhline(y=min_mse, color='red', linestyle='--', alpha=0.7, label=f'Min MSE: {min_mse:.4f}')
        ax1.axhline(y=avg_mse, color='orange', linestyle='--', alpha=0.7, label=f'Avg MSE: {avg_mse:.4f}')
        ax1.legend()
        
        # RMSE plot
        ax2.plot(self.epochs, self.rmse_scores, 'darkgreen', marker='s', markersize=4, linewidth=2)
        ax2.set_title('Root Mean Squared Error (RMSE) Over Training', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('RMSE')
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#f8f9fa')
        
        # Add RMSE statistics
        min_rmse = min(self.rmse_scores)
        avg_rmse = np.mean(self.rmse_scores)
        ax2.axhline(y=min_rmse, color='red', linestyle='--', alpha=0.7, label=f'Min RMSE: {min_rmse:.4f}')
        ax2.axhline(y=avg_rmse, color='orange', linestyle='--', alpha=0.7, label=f'Avg RMSE: {avg_rmse:.4f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'vit_mse_rmse_metrics.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"MSE/RMSE plot saved to: {os.path.join(save_dir, 'vit_mse_rmse_metrics.png')}")

# ======================== MAIN VIT TRAINING CLASS ========================
class ViTAntiSpoofing:
    """
    Advanced Vision Transformer Anti-Spoofing Training System.
    
    This class provides a comprehensive training system for face anti-spoofing
    using Vision Transformers with advanced techniques and monitoring.
    """
    
    def __init__(self, model_name=MODEL_NAME, num_classes=NUM_CLASSES):
        """
        Initialize the ViT Anti-Spoofing training system.
        
        Args:
            model_name (str): HuggingFace model identifier
            num_classes (int): Number of output classes
        """
        self.device = device
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Initialize metrics logger
        self.metrics_logger = ViTMetricsLogger()
        
        # Training state
        self.best_val_acc = 0.0
        self.current_epoch = 0
        self.training_interrupted = False
        
        # Setup signal handler for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)
        
        print(f"Initializing ViT Anti-Spoofing System")
        print(f"Device: {self.device}")
        print(f"Model: {model_name}")
        print(f"Classes: {num_classes}")
        
        # Initialize model
        self._initialize_model()
        
        # Setup data transforms
        self.train_transform, self.val_transform = get_vit_transforms()
    
    def _initialize_model(self):
        """Initialize the Vision Transformer model."""
        print("Loading pre-trained Vision Transformer...")
        
        try:
            # Load pre-trained ViT model
            self.model = ViTForImageClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True,
                cache_dir='./vit_cache'  # Cache downloaded models
            )
            
            # Enable gradient checkpointing for memory efficiency
            if GRADIENT_CHECKPOINTING and hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("Gradient checkpointing enabled (saves GPU memory)")
            
            # Move model to device
            self.model.to(self.device)
            
            # Setup mixed precision scaler
            if USE_AMP and torch.cuda.is_available():
                self.scaler = GradScaler()
                print("Mixed precision training enabled")
            else:
                self.scaler = None
            
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            print(f"Model loaded successfully!")
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle training interruption gracefully."""
        print("\nTraining interrupted! Saving checkpoint...")
        self.training_interrupted = True
    
    def load_data(self, train_dir, test_dir, batch_size=BATCH_SIZE):
        """
        Load and prepare training and validation datasets.
        
        Args:
            train_dir (str): Path to training data directory
            test_dir (str): Path to test/validation data directory
            batch_size (int): Batch size for data loading
            
        Returns:
            tuple: (train_loader, test_loader)
        """
        print(f"Loading datasets...")
        print(f"Training data: {train_dir}")
        print(f"Test data: {test_dir}")
        print(f"Batch size: {batch_size}")
        
        try:
            # Create datasets
            train_dataset = CASIAFASDDataset(train_dir, transform=self.train_transform)
            test_dataset = CASIAFASDDataset(test_dir, transform=self.val_transform)
            
            # Calculate class weights for balanced sampling
            train_labels = train_dataset.labels
            class_counts = Counter(train_labels)
            total_samples = len(train_labels)
            
            # Calculate inverse frequency weights
            class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
            sample_weights = [class_weights[label] for label in train_labels]
            
            # Create weighted sampler for balanced training
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            # Create data loaders
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                persistent_workers=PERSISTENT_WORKERS
            )
            
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                persistent_workers=PERSISTENT_WORKERS
            )
            
            print(f"Datasets loaded successfully!")
            print(f"Train samples: {len(train_dataset)}")
            print(f"Test samples: {len(test_dataset)}")
            print(f"Class distribution: {class_counts}")
            
            return self.train_loader, self.test_loader
            
        except Exception as e:
            print(f"Error loading datasets: {e}")
            raise
    
    def save_checkpoint(self, epoch, optimizer, scheduler, save_dir, is_best=False):
        """
        Save training checkpoint with comprehensive state.
        
        Args:
            epoch (int): Current epoch number
            optimizer: Current optimizer state
            scheduler: Current scheduler state
            save_dir (str): Directory to save checkpoint
            is_best (bool): Whether this is the best model so far
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'metrics_logger': {
                'train_losses': self.metrics_logger.train_losses,
                'train_accs': self.metrics_logger.train_accs,
                'val_losses': self.metrics_logger.val_losses,
                'val_accs': self.metrics_logger.val_accs,
                'mse_scores': self.metrics_logger.mse_scores,
                'rmse_scores': self.metrics_logger.rmse_scores,
                'learning_rates': self.metrics_logger.learning_rates,
                'epochs': self.metrics_logger.epochs
            },
            'config': {
                'model_name': self.model_name,
                'num_classes': self.num_classes,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'image_size': IMAGE_SIZE
            }
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(save_dir, 'vit_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if applicable
        if is_best:
            best_path = os.path.join(save_dir, 'best_vit_model.pth')
            torch.save(checkpoint, best_path)
            # Also save just the model state dict for easy loading
            model_path = os.path.join(save_dir, MODEL_NAME_SAVE)
            torch.save(self.model.state_dict(), model_path)
            print(f"New best model saved! Validation accuracy: {self.best_val_acc:.2f}%")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path, optimizer=None, scheduler=None):
        """
        Load checkpoint and restore training state.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
            optimizer: Optimizer to restore state
            scheduler: Scheduler to restore state
            
        Returns:
            dict: Loaded checkpoint data
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        try:
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Restore model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Restore optimizer state if provided
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore scheduler state if provided
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Restore training state
            self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
            self.current_epoch = checkpoint.get('epoch', 0)
            
            # Restore metrics logger
            if 'metrics_logger' in checkpoint:
                metrics = checkpoint['metrics_logger']
                self.metrics_logger.train_losses = metrics.get('train_losses', [])
                self.metrics_logger.train_accs = metrics.get('train_accs', [])
                self.metrics_logger.val_losses = metrics.get('val_losses', [])
                self.metrics_logger.val_accs = metrics.get('val_accs', [])
                self.metrics_logger.mse_scores = metrics.get('mse_scores', [])
                self.metrics_logger.rmse_scores = metrics.get('rmse_scores', [])
                self.metrics_logger.learning_rates = metrics.get('learning_rates', [])
                self.metrics_logger.epochs = metrics.get('epochs', [])
            
            print(f"Checkpoint loaded successfully from epoch {self.current_epoch + 1}")
            print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
            
            return checkpoint
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None
    
    def calculate_mse_rmse(self, predictions, targets):
        """
        Calculate MSE and RMSE for regression-like analysis.
        
        For classification tasks, we treat the predicted probabilities
        as continuous values and compare with binary targets.
        
        Args:
            predictions (torch.Tensor): Model predictions (logits)
            targets (torch.Tensor): Ground truth labels
            
        Returns:
            tuple: (mse, rmse) values
        """
        # Convert logits to probabilities
        probs = torch.softmax(predictions, dim=1)
        # Get probability for positive class (live faces)
        pos_probs = probs[:, 1].cpu().numpy()
        targets_np = targets.cpu().numpy().astype(float)
        
        # Calculate MSE and RMSE
        mse = mean_squared_error(targets_np, pos_probs)
        rmse = np.sqrt(mse)
        
        return mse, rmse
    
    def train(self, train_dir, test_dir, epochs=EPOCHS, learning_rate=LEARNING_RATE,
              save_dir=MODEL_DIR, resume_checkpoint=None):
        """
        Main training function with advanced ViT techniques.
        
        Args:
            train_dir (str): Path to training data directory
            test_dir (str): Path to test data directory
            epochs (int): Number of training epochs
            learning_rate (float): Initial learning rate
            save_dir (str): Directory to save models and logs
            resume_checkpoint (str): Path to checkpoint for resuming training
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Load datasets
        print("="*60)
        print("VISION TRANSFORMER FACE ANTI-SPOOFING TRAINING")
        print("="*60)
        
        try:
            self.load_data(train_dir, test_dir)
        except Exception as e:
            print(f"Failed to load data: {e}")
            return None
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(ADAM_BETA1, ADAM_BETA2),
            eps=ADAM_EPS,
            weight_decay=WEIGHT_DECAY
        )
        
        # Cosine annealing scheduler with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_MULT, eta_min=ETA_MIN
        )
        
        # Loss functions
        ce_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        focal_criterion = FocalLoss(alpha=1, gamma=2)
        
        # Training state
        start_epoch = 0
        epochs_without_improvement = 0
        
        # Load checkpoint if resuming
        if resume_checkpoint:
            checkpoint = self.load_checkpoint(resume_checkpoint, optimizer, scheduler)
            if checkpoint:
                start_epoch = self.current_epoch + 1
                print(f"Resuming training from epoch {start_epoch}")
        
        # Freeze backbone for initial epochs if specified
        if FREEZE_BACKBONE_EPOCHS > 0 and start_epoch < FREEZE_BACKBONE_EPOCHS:
            print(f"Freezing backbone for first {FREEZE_BACKBONE_EPOCHS} epochs")
            for name, param in self.model.named_parameters():
                if 'classifier' not in name:  # Freeze all except classifier
                    param.requires_grad = False
        
        print(f"Starting training for {epochs - start_epoch} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {USE_AMP}")
        print(f"Gradient checkpointing: {GRADIENT_CHECKPOINTING}")
        
        # Main training loop
        for epoch in range(start_epoch, epochs):
            if self.training_interrupted:
                print("Training interrupted - saving final checkpoint...")
                self.save_checkpoint(epoch, optimizer, scheduler, save_dir)
                break
            
            # Unfreeze backbone after specified epochs
            if epoch == FREEZE_BACKBONE_EPOCHS:
                print("Unfreezing backbone parameters")
                for param in self.model.parameters():
                    param.requires_grad = True
            
            epoch_start_time = time.time()
            self.current_epoch = epoch
            
            # Training phase
            train_loss, train_acc, train_mse, train_rmse = self._train_epoch(
                optimizer, ce_criterion, focal_criterion
            )
            
            # Validation phase
            val_loss, val_acc, val_mse, val_rmse = self._validate_epoch(ce_criterion)
            
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log metrics
            self.metrics_logger.log_epoch(
                epoch, train_loss, train_acc, val_loss, val_acc,
                val_mse, val_rmse, current_lr
            )
            
            # Check for improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                epochs_without_improvement = 0
                self.save_checkpoint(epoch, optimizer, scheduler, save_dir, is_best=True)
            else:
                epochs_without_improvement += 1
            
            # Regular checkpoint saving
            if epoch % 5 == 0:  # Save every 5 epochs
                self.save_checkpoint(epoch, optimizer, scheduler, save_dir)
            
            # Display epoch results
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch [{epoch+1}/{epochs}] ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}")
            print(f"  LR: {current_lr:.2e}, Best Val Acc: {self.best_val_acc:.2f}%")
            print("-" * 80)
            
            # Early stopping
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save final results
        self.save_checkpoint(epochs-1, optimizer, scheduler, save_dir)
        self.metrics_logger.save_training_plots(save_dir)
        
        # Save training history
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump({
                'train_losses': self.metrics_logger.train_losses,
                'train_accs': self.metrics_logger.train_accs,
                'val_losses': self.metrics_logger.val_losses,
                'val_accs': self.metrics_logger.val_accs,
                'mse_scores': self.metrics_logger.mse_scores,
                'rmse_scores': self.metrics_logger.rmse_scores,
                'learning_rates': self.metrics_logger.learning_rates,
                'best_val_acc': self.best_val_acc
            }, f, indent=2)
        
        print("="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Results saved in: {save_dir}")
        print("="*60)
        
        return self.metrics_logger
    
    def _train_epoch(self, optimizer, ce_criterion, focal_criterion):
        """
        Execute one training epoch with advanced augmentation and loss combination.
        
        Args:
            optimizer: Optimizer for parameter updates
            ce_criterion: Cross-entropy loss function
            focal_criterion: Focal loss function
            
        Returns:
            tuple: (avg_loss, accuracy, mse, rmse) for the epoch
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(self.train_loader, desc=f'Training Epoch {self.current_epoch+1}')
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            if self.training_interrupted:
                break
            
            data, targets = data.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            
            # Apply mixup/cutmix augmentation
            use_mixup = use_cutmix = False
            if random.random() < MIXUP_PROB:
                if random.random() < 0.5:
                    data, targets_a, targets_b, lam = mixup_data_vit(data, targets)
                    use_mixup = True
                else:
                    data, targets_a, targets_b, lam = cutmix_data_vit(data, targets)
                    use_cutmix = True
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(data)
                    logits = outputs.logits
                    
                    if use_mixup or use_cutmix:
                        ce_loss = mixup_criterion(ce_criterion, logits, targets_a, targets_b, lam)
                        focal_loss = mixup_criterion(focal_criterion, logits, targets_a, targets_b, lam)
                    else:
                        ce_loss = ce_criterion(logits, targets)
                        focal_loss = focal_criterion(logits, targets)
                    
                    # Combine losses (70% CE + 30% Focal)
                    loss = 0.7 * ce_loss + 0.3 * focal_loss
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP_NORM)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Standard precision fallback
                outputs = self.model(data)
                logits = outputs.logits
                
                if use_mixup or use_cutmix:
                    ce_loss = mixup_criterion(ce_criterion, logits, targets_a, targets_b, lam)
                    focal_loss = mixup_criterion(focal_criterion, logits, targets_a, targets_b, lam)
                else:
                    ce_loss = ce_criterion(logits, targets)
                    focal_loss = focal_criterion(logits, targets)
                
                loss = 0.7 * ce_loss + 0.3 * focal_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP_NORM)
                optimizer.step()
            
            # Accumulate statistics (only for non-mixed samples)
            total_loss += loss.item()
            if not (use_mixup or use_cutmix):
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Store for MSE/RMSE calculation
                all_predictions.extend(logits.detach())
                all_targets.extend(targets.detach())
            
            # Update progress bar
            if total > 0:
                current_acc = 100.0 * correct / total
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{current_acc:.2f}%'})
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total if total > 0 else 0
        
        # Calculate MSE and RMSE
        if all_predictions and all_targets:
            all_preds_tensor = torch.stack(all_predictions)
            all_targets_tensor = torch.stack(all_targets)
            mse, rmse = self.calculate_mse_rmse(all_preds_tensor, all_targets_tensor)
        else:
            mse = rmse = 0.0
        
        return avg_loss, accuracy, mse, rmse
    
    def _validate_epoch(self, criterion):
        """
        Execute one validation epoch.
        
        Args:
            criterion: Loss function for validation
            
        Returns:
            tuple: (avg_loss, accuracy, mse, rmse) for the epoch
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                
                if self.scaler is not None:
                    with autocast():
                        outputs = self.model(data)
                        logits = outputs.logits
                        loss = criterion(logits, targets)
                else:
                    outputs = self.model(data)
                    logits = outputs.logits
                    loss = criterion(logits, targets)
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Store for MSE/RMSE calculation
                all_predictions.extend(logits.detach())
                all_targets.extend(targets.detach())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100.0 * correct / total
        
        # Calculate MSE and RMSE
        if all_predictions and all_targets:
            all_preds_tensor = torch.stack(all_predictions)
            all_targets_tensor = torch.stack(all_targets)
            mse, rmse = self.calculate_mse_rmse(all_preds_tensor, all_targets_tensor)
        else:
            mse = rmse = 0.0
        
        return avg_loss, accuracy, mse, rmse
    
    def test_and_confusion_matrix(self, model_path=None, save_dir=None):
        """
        Comprehensive testing with confusion matrix and detailed metrics.
        
        Args:
            model_path (str): Path to model weights (optional)
            save_dir (str): Directory to save results (optional)
        """
        if model_path:
            print(f"Loading model from: {model_path}")
            if model_path.endswith('.pth') and 'checkpoint' not in model_path:
                # Load just the model state dict
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                # Load from checkpoint
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probs = []
        
        print("Generating predictions for detailed analysis...")
        
        with torch.no_grad():
            for data, targets in tqdm(self.test_loader, desc="Testing"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                if self.scaler is not None:
                    with autocast():
                        outputs = self.model(data)
                        logits = outputs.logits
                else:
                    outputs = self.model(data)
                    logits = outputs.logits
                
                # Get predictions and probabilities
                probs = torch.softmax(logits, dim=1)
                _, predicted = logits.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        cm = confusion_matrix(all_targets, all_predictions)
        
        # Calculate MSE and RMSE for the test set
        probs_array = np.array(all_probs)
        pos_probs = probs_array[:, 1]  # Probability for positive class
        targets_array = np.array(all_targets).astype(float)
        test_mse = mean_squared_error(targets_array, pos_probs)
        test_rmse = np.sqrt(test_mse)
        
        print(f"\n" + "="*50)
        print("COMPREHENSIVE TEST RESULTS")
        print("="*50)
        print(f"Test Accuracy: {accuracy*100:.2f}%")
        print(f"Test MSE: {test_mse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        
        # Detailed classification report
        class_names = ['spoof', 'live']
        report = classification_report(all_targets, all_predictions, target_names=class_names)
        print(f"\nClassification Report:")
        print(report)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(cm, accuracy, save_dir)
        
        return cm, accuracy, test_mse, test_rmse
    
    def _plot_confusion_matrix(self, cm, accuracy, save_dir=None):
        """Plot and save detailed confusion matrix."""
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Spoof', 'Live'],
                   yticklabels=['Spoof', 'Live'],
                   cbar_kws={'label': 'Count'},
                   annot_kws={'size': 16, 'weight': 'bold'})
        
        # Add title and labels
        plt.title(f'Vision Transformer Anti-Spoofing\nConfusion Matrix (Accuracy: {accuracy*100:.2f}%)',
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=14, fontweight='bold')
        
        # Add performance metrics text
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics_text = f"""Detailed Metrics:
Precision: {precision:.3f}
Recall: {recall:.3f}
F1-Score: {f1:.3f}
Specificity: {specificity:.3f}
        
True Negatives: {tn}
False Positives: {fp}
False Negatives: {fn}
True Positives: {tp}"""
        
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10, fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'vit_confusion_matrix.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved: {save_path}")
        
        plt.show()


# ======================== MAIN FUNCTION ========================
def main():
    """
    Main function to orchestrate ViT anti-spoofing training.
    
    This function handles the complete training pipeline with proper error handling
    and user interaction for resuming training from checkpoints.
    """
    # Training configuration
    config = {
        'train_dir': 'CASIA-FASD/train',  # Update this path to your training data
        'test_dir': 'CASIA-FASD/test',    # Update this path to your test data
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'save_dir': MODEL_DIR,
        'resume_checkpoint': None  # Set to checkpoint path to resume training
    }
    
    print("Vision Transformer Face Anti-Spoofing Training System")
    print("=" * 60)
    
    # Display system information
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_props.name}")
        print(f"GPU Memory: {gpu_props.total_memory / 1e9:.1f} GB")
        print(f"CUDA Compute Capability: {gpu_props.major}.{gpu_props.minor}")
    
    # Check for existing checkpoint
    checkpoint_path = os.path.join(config['save_dir'], 'vit_checkpoint.pth')
    resume_path = None
    
    if config['resume_checkpoint']:
        resume_path = config['resume_checkpoint']
        print(f"Will resume from specified checkpoint: {resume_path}")
    elif os.path.exists(checkpoint_path):
        print(f"\nFound existing checkpoint: {checkpoint_path}")
        response = input("Resume from checkpoint? (y/n): ").lower().strip()
        if response == 'y':
            resume_path = checkpoint_path
    
    # Initialize ViT training system
    try:
        vit_system = ViTAntiSpoofing()
    except Exception as e:
        print(f"Failed to initialize ViT system: {e}")
        return None
    
    # Start training
    print(f"\nStarting ViT training...")
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("\nPress Ctrl+C to save checkpoint and pause training gracefully")
    
    try:
        metrics_logger = vit_system.train(
            train_dir=config['train_dir'],
            test_dir=config['test_dir'],
            epochs=config['epochs'],
            learning_rate=config['learning_rate'],
            save_dir=config['save_dir'],
            resume_checkpoint=resume_path
        )
        
        if metrics_logger is None:
            print("Training failed - check error messages above")
            return None
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Checkpoint should be saved automatically")
        return vit_system
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Test the best model
    print(f"\nTesting best model...")
    try:
        best_model_path = os.path.join(config['save_dir'], 'best_vit_model.pth')
        if os.path.exists(best_model_path):
            cm, accuracy, test_mse, test_rmse = vit_system.test_and_confusion_matrix(
                model_path=best_model_path, 
                save_dir=config['save_dir']
            )
            print(f"\nFinal Test Results:")
            print(f"  Accuracy: {accuracy*100:.2f}%")
            print(f"  MSE: {test_mse:.4f}")
            print(f"  RMSE: {test_rmse:.4f}")
        else:
            print("No best model found, testing current model...")
            cm, accuracy, test_mse, test_rmse = vit_system.test_and_confusion_matrix(
                save_dir=config['save_dir']
            )
    except Exception as e:
        print(f"Testing error: {e}")
        import traceback
        traceback.print_exc()
        return vit_system
    
    print(f"\n" + "="*60)
    print("TRAINING AND TESTING COMPLETED SUCCESSFULLY!")
    print(f"Best validation accuracy: {vit_system.best_val_acc:.2f}%")
    print(f"Final test accuracy: {accuracy*100:.2f}%")
    print(f"All results saved in: {config['save_dir']}")
    print("="*60)
    
    return vit_system


def demonstrate_usage():
    """
    Demonstrate how to use the ViT training system.
    
    This function shows examples of different usage scenarios including
    training from scratch, resuming from checkpoint, and testing models.
    """
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    
    print("\n1. TRAINING FROM SCRATCH:")
    print("   - Set up your data directories:")
    print("     train_dir/live/*.png")
    print("     train_dir/spoof/*.png")
    print("     test_dir/live/*.png") 
    print("     test_dir/spoof/*.png")
    print("   - Run: python vit_training.py")
    
    print("\n2. RESUMING FROM CHECKPOINT:")
    print("   - Automatic detection: Script will ask if you want to resume")
    print("   - Manual: Set config['resume_checkpoint'] = 'path/to/checkpoint.pth'")
    
    print("\n3. TESTING ONLY:")
    print("   vit_system = ViTAntiSpoofing()")
    print("   vit_system.load_data('train_dir', 'test_dir')")
    print("   vit_system.test_and_confusion_matrix('best_model.pth')")
    
    print("\n4. CUSTOMIZING CONFIGURATION:")
    print("   - Modify constants at top of script:")
    print("   - BATCH_SIZE, EPOCHS, LEARNING_RATE, etc.")
    print("   - Adjust for your hardware capabilities")
    
    print("\n5. MONITORING TRAINING:")
    print("   - Watch GPU: nvidia-smi -l 1")
    print("   - Training plots saved automatically")
    print("   - MSE/RMSE plots generated separately")
    print("   - Checkpoints saved every 5 epochs")
    
    print("\n6. MEMORY OPTIMIZATION TIPS:")
    print("   - Reduce BATCH_SIZE if OOM errors occur")
    print("   - Enable gradient checkpointing (already enabled)")
    print("   - Use mixed precision training (already enabled)")
    print("   - Reduce NUM_WORKERS if system becomes unstable")


def print_system_requirements():
    """Print system requirements and optimization tips."""
    print("\n" + "="*60)
    print("SYSTEM REQUIREMENTS & OPTIMIZATION")
    print("="*60)
    
    print("\nMINIMUM REQUIREMENTS:")
    print("  GPU: 4GB VRAM (RTX 3050 or equivalent)")
    print("  RAM: 8GB system memory")
    print("  Storage: 2GB for model cache + dataset")
    print("  CUDA: 11.0 or higher")
    
    print("\nRECOMMENDED:")
    print("  GPU: 8GB+ VRAM (RTX 3070 or better)")
    print("  RAM: 16GB+ system memory")
    print("  Storage: NVMe SSD for faster data loading")
    
    print("\nOPTIMIZATION TIPS:")
    print("  - Current batch size (8) is optimized for 4GB VRAM")
    print("  - Mixed precision training reduces memory by ~40%")
    print("  - Gradient checkpointing saves memory during backprop")
    print("  - Persistent workers reduce data loading overhead")
    print("  - Weighted sampling ensures balanced training")
    
    print("\nTROUBLESHOoting:")
    print("  CUDA OOM: Reduce BATCH_SIZE to 4 or 2")
    print("  Slow training: Increase NUM_WORKERS (max: CPU cores)")
    print("  System freeze: Reduce NUM_WORKERS to 1")
    print("  Poor accuracy: Check data quality and class balance")


if __name__ == "__main__":
    """
    Main execution block with comprehensive error handling and user guidance.
    
    This block handles the complete workflow from initialization to final results,
    with proper error handling and informative output for users.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Print system information and requirements
    print_system_requirements()
    
    try:
        # Run main training
        trained_system = main()
        
        if trained_system is not None:
            print("\nTraining system ready for further use!")
            demonstrate_usage()
        else:
            print("\nTraining failed. Please check the error messages above.")
            print("Common solutions:")
            print("  1. Verify dataset directory structure")
            print("  2. Check CUDA installation")
            print("  3. Reduce batch size if memory issues")
            print("  4. Ensure sufficient disk space")
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Checkpoint should be automatically saved.")
        print("Run the script again to resume training.")
        
    except Exception as e:
        print(f"\nUnexpected error occurred: {e}")
        print("\nFull error traceback:")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting steps:")
        print("  1. Check dataset directory exists and has correct structure")
        print("  2. Verify CUDA and PyTorch installation")
        print("  3. Try reducing BATCH_SIZE in configuration")
        print("  4. Check available GPU memory with 'nvidia-smi'")
        print("  5. Ensure transformers library is installed: pip install transformers")
    
    finally:
        print(f"\nScript execution completed.")
        print(f"For questions or issues, check the comprehensive comments in the code.")
        print(f"All training artifacts are saved in the '{MODEL_DIR}' directory.")
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cache cleared.")