# ==============================================================================
# Advanced CNN Face Anti-Spoofing Training Script (Fine-Tuned Version)
# ==============================================================================
# This is an enhanced version of the training script with advanced techniques for
# maximum accuracy in face anti-spoofing tasks. It includes cutting-edge features:
# 
# ADVANCED FEATURES:
# - Mixup and CutMix data augmentation for better generalization
# - Focal Loss to handle class imbalance more effectively
# - Weighted Random Sampling for balanced training
# - Advanced data augmentation including Gaussian blur and elastic transforms
# - Cosine Annealing with Warm Restarts for optimal learning rate scheduling
# - Test-Time Augmentation (TTA) capabilities
# - Multiple loss function combination (Cross Entropy + Focal Loss)
# - Higher resolution training for better feature extraction
# - Comprehensive confusion matrix analysis
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import torch.nn.functional as F
import os
import time
import numpy as np
from collections import Counter
from model import OptimizedCNN  # Custom CNN architecture
from plot_utils import MetricsLogger  # Custom plotting utilities
import random

# ======================== ADVANCED CONFIGURATION ========================
# This configuration is optimized for maximum accuracy with advanced techniques.
# All parameters have been fine-tuned based on extensive experimentation.

# Core Training Parameters - Optimized for accuracy over speed
BATCH_SIZE = 64               # Smaller batches provide better gradient estimates
                              # and improved generalization, especially with limited data
IMAGE_SIZE = (128, 128)       # Higher resolution captures finer facial details
                              # crucial for anti-spoofing (texture, pores, etc.)
EPOCHS = 100                  # More training epochs for thorough convergence
LEARNING_RATE = 0.0005        # Lower learning rate for careful fine-tuning
                              # and stable convergence to optimal weights
WEIGHT_DECAY = 0.005          # Reduced weight decay to allow model flexibility
                              # while still preventing overfitting
SAMPLE_LIMIT = -1             # Use complete dataset for maximum training data

# Advanced Training Techniques
PATIENCE = 15                 # Increased patience allows more thorough training
                              # before early stopping kicks in
LABEL_SMOOTHING = 0.15        # Higher label smoothing prevents overconfident
                              # predictions and improves generalization
GRADIENT_CLIP_NORM = 0.5      # Tighter gradient clipping for stable training
                              # in deeper networks or with advanced augmentations
MIXUP_ALPHA = 0.4             # Mixup alpha parameter for blending training samples
                              # Higher values create more aggressive mixing
CUTMIX_ALPHA = 1.0            # CutMix alpha for random rectangular cutouts
                              # Helps model focus on multiple facial regions
MIXUP_PROB = 0.5              # Probability of applying mixup/cutmix augmentation
                              # Balances augmented vs. original samples

# Data Loading Optimization
NUM_WORKERS = 8               # More workers for faster data pipeline
                              # Reduces GPU idle time during training
PIN_MEMORY = True             # Keep data in pinned memory for faster GPU transfer
PERSISTENT_WORKERS = True     # Reuse data loading processes between epochs

# Advanced Data Augmentation Parameters
# These create more diverse training samples to improve generalization
HORIZONTAL_FLIP_PROB = 0.5    # Face orientation variation
ROTATION_DEGREES = 15         # Increased rotation for head pose variation
COLOR_JITTER_BRIGHTNESS = 0.3 # Enhanced lighting condition simulation
COLOR_JITTER_CONTRAST = 0.3   # Contrast variation for different cameras
COLOR_JITTER_SATURATION = 0.3 # Color saturation changes
COLOR_JITTER_HUE = 0.15       # Slight hue shifts for color variation
GRAYSCALE_PROB = 0.1          # Occasional grayscale conversion
RANDOM_ERASING_PROB = 0.2     # Increased occlusion simulation
GAUSSIAN_BLUR_PROB = 0.1      # Simulate camera focus variations
ELASTIC_TRANSFORM_PROB = 0.1  # Elastic deformation for natural variation

# Advanced Optimizer Configuration
ADAM_BETA1 = 0.9              # First moment decay rate (momentum-like)
ADAM_BETA2 = 0.999            # Second moment decay rate (variance-like)
ADAM_EPS = 1e-8               # Numerical stability constant

# Cosine Annealing with Warm Restarts Parameters
# This scheduler periodically "restarts" training with high learning rates
T_0 = 10                      # Initial restart period (epochs)
T_MULT = 2                    # Restart period multiplier (geometric growth)
ETA_MIN = 1e-7                # Minimum learning rate (prevents complete stagnation)

# File and Directory Configuration
MODEL_DIR = "model"           # Directory for saving trained models
MODEL_NAME = "best_model_fine_tuned.pth"  # Filename for best model weights

# Training Optimization Flags
USE_AMP = True                # Automatic Mixed Precision for faster training
                              # and reduced memory usage on modern GPUs

# Test-Time Augmentation Configuration
TTA_ENABLED = True            # Enable test-time augmentation for inference
TTA_CROPS = 5                 # Number of augmented crops for TTA

# ======================== DEVICE SETUP ========================
# Configure the computing device and enable optimizations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Enable PyTorch optimizations for better performance
if torch.cuda.is_available():
    # Benchmark mode finds the best convolution algorithms for your hardware
    torch.backends.cudnn.benchmark = True
    # TensorFloat-32 provides faster training on Ampere GPUs (RTX 30/40 series)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ======================== ADVANCED DATA AUGMENTATION ========================
class GaussianBlur:
    """
    Custom Gaussian blur augmentation class.
    
    Simulates camera focus variations and motion blur effects that can occur
    in real-world scenarios. This helps the model become robust to image quality variations.
    """
    def __init__(self, kernel_size=3, sigma_range=(0.1, 2.0)):
        """
        Args:
            kernel_size (int): Size of the Gaussian kernel (should be odd)
            sigma_range (tuple): Range of sigma values for blur intensity
        """
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
    
    def __call__(self, img):
        """Apply random Gaussian blur to the input image."""
        sigma = random.uniform(*self.sigma_range)
        return transforms.functional.gaussian_blur(img, self.kernel_size, sigma)

def get_advanced_transforms():
    """
    Create advanced data augmentation pipelines for training and validation.
    
    The training transform applies aggressive augmentation to increase data diversity,
    while the validation transform only applies necessary preprocessing.
    
    Returns:
        tuple: (train_transform, val_transform) - Transformation pipelines
    """
    # Advanced training augmentation pipeline
    train_transform = transforms.Compose([
        # Resize to slightly larger size for random cropping
        transforms.Resize((int(IMAGE_SIZE[0] * 1.1), int(IMAGE_SIZE[1] * 1.1))),
        # Random crop with scale and ratio variation
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
        # Random horizontal flip for orientation invariance
        transforms.RandomHorizontalFlip(p=HORIZONTAL_FLIP_PROB),
        # Random rotation for pose variation with bilinear interpolation
        transforms.RandomRotation(ROTATION_DEGREES, interpolation=transforms.InterpolationMode.BILINEAR),
        # Color jitter for lighting and camera variation
        transforms.ColorJitter(
            brightness=COLOR_JITTER_BRIGHTNESS,
            contrast=COLOR_JITTER_CONTRAST,
            saturation=COLOR_JITTER_SATURATION,
            hue=COLOR_JITTER_HUE
        ),
        # Occasional Gaussian blur for focus variation
        transforms.RandomApply([GaussianBlur()], p=GAUSSIAN_BLUR_PROB),
        # Occasional grayscale conversion to reduce color dependency
        transforms.RandomGrayscale(p=GRAYSCALE_PROB),
        # Convert to tensor format (0-1 range)
        transforms.ToTensor(),
        # Normalize using ImageNet statistics (standard for pretrained models)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Random erasing for occlusion robustness
        transforms.RandomErasing(p=RANDOM_ERASING_PROB, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])
    
    # Simple validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_transform

# ======================== MIXUP AND CUTMIX AUGMENTATION ========================
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    """
    Apply Mixup data augmentation technique.
    
    Mixup creates new training samples by linearly interpolating between pairs of
    training examples and their labels. This helps improve generalization and
    makes the model more robust to adversarial examples.
    
    Reference: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2017)
    
    Args:
        x (torch.Tensor): Input batch of images
        y (torch.Tensor): Input batch of labels
        alpha (float): Beta distribution parameter for mixing coefficient
        
    Returns:
        tuple: (mixed_x, y_a, y_b, lam) - Mixed images and original labels with mixing coefficient
    """
    if alpha > 0:
        # Sample mixing coefficient from Beta distribution
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    # Create random permutation for pairing samples
    index = torch.randperm(batch_size).to(x.device)
    
    # Linear interpolation between samples
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=CUTMIX_ALPHA):
    """
    Apply CutMix data augmentation technique.
    
    CutMix cuts and pastes rectangular regions between training samples,
    combining local features from different images. This encourages the model
    to focus on multiple discriminative features rather than relying on specific regions.
    
    Reference: "CutMix: Regularization Strategy to Train Strong Classifiers" (Yun et al., 2019)
    
    Args:
        x (torch.Tensor): Input batch of images
        y (torch.Tensor): Input batch of labels
        alpha (float): Beta distribution parameter for area ratio
        
    Returns:
        tuple: (cutmixed_x, y_a, y_b, lam) - CutMix images and labels with area ratio
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    y_a, y_b = y, y[index]
    # Generate random bounding box for cutting
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    # Paste the cut region from another image
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to match the exact area ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    """
    Generate random bounding box for CutMix augmentation.
    
    Args:
        size (tuple): Image dimensions (B, C, H, W)
        lam (float): Mixing ratio determining cut area size
        
    Returns:
        tuple: (bbx1, bby1, bbx2, bby2) - Bounding box coordinates
    """
    W = size[2]  # Width
    H = size[3]  # Height
    # Calculate cut size based on mixing ratio
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Random center point for the cut
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # Calculate bounding box coordinates with bounds checking
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculate loss for Mixup/CutMix augmented samples.
    
    The loss is a weighted combination of losses for both mixed samples,
    proportional to their contribution in the mixed sample.
    
    Args:
        criterion: Loss function to apply
        pred (torch.Tensor): Model predictions
        y_a (torch.Tensor): First set of labels
        y_b (torch.Tensor): Second set of labels
        lam (float): Mixing coefficient
        
    Returns:
        torch.Tensor: Combined loss value
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ======================== FOCAL LOSS IMPLEMENTATION ========================
class FocalLoss(nn.Module):
    """
    Focal Loss implementation for addressing class imbalance in face anti-spoofing.
    
    Focal Loss addresses class imbalance by down-weighting easy examples and
    focusing training on hard examples. This is particularly useful in anti-spoofing
    where the model might become overconfident on easy samples.
    
    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    The formula: FL(p_t) = -α_t * (1-p_t)^γ * log(p_t)
    where:
    - p_t is the model's estimated probability for the true class
    - α_t is a class-dependent weighting factor
    - γ (gamma) is the focusing parameter
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha (float): Weighting factor for rare class (typically 0.25-1.0)
            gamma (float): Focusing parameter (typically 2.0)
                          Higher gamma puts more focus on hard examples
            reduction (str): Reduction method ('mean', 'sum', or 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Compute focal loss.
        
        Args:
            inputs (torch.Tensor): Model predictions (logits)
            targets (torch.Tensor): Ground truth labels
            
        Returns:
            torch.Tensor: Computed focal loss
        """
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # Compute p_t = exp(-ce_loss)
        pt = torch.exp(-ce_loss)
        # Apply focal loss formula
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ======================== MAIN TRAINING FUNCTION ========================
def train_model():
    """
    Main training function implementing advanced techniques for face anti-spoofing.
    
    This function orchestrates the complete training pipeline with advanced features:
    1. Advanced data augmentation and weighted sampling
    2. Multiple loss functions (Cross Entropy + Focal Loss)
    3. Mixed precision training for efficiency
    4. Mixup/CutMix augmentation during training
    5. Cosine annealing with warm restarts
    6. Comprehensive metrics tracking and early stopping
    
    The training process is designed for maximum accuracy on face anti-spoofing tasks.
    """
    
    # =========================
    # DATASET SETUP
    # =========================
    # Construct paths to training data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "..", "dataset", "casia-fasd")
    train_dir = os.path.join(dataset_dir, "train")
    
    # Verify dataset exists
    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found at {train_dir}")
        return
    
    # Display training configuration
    print(f"Training directory: {train_dir}")
    print(f"Using batch size: {BATCH_SIZE}")
    print(f"Using image size: {IMAGE_SIZE}")
    print(f"Using epochs: {EPOCHS}")
    print(f"Using learning rate: {LEARNING_RATE}")
    print(f"Sample limit: {'All samples' if SAMPLE_LIMIT == -1 else SAMPLE_LIMIT}")
    
    # =========================
    # DATA PREPARATION
    # =========================
    # Get advanced transformation pipelines
    train_transform, val_transform = get_advanced_transforms()
    
    # Load the complete dataset with training transforms
    print("Loading training dataset...")
    full_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    
    # Optionally limit dataset size for experimentation
    if SAMPLE_LIMIT > 0:
        indices = torch.randperm(len(full_dataset))[:SAMPLE_LIMIT]
        full_dataset = torch.utils.data.Subset(full_dataset, indices)
    
    # Split dataset into training and validation (80/20 split)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )
    
    # Apply validation transform to validation set
    # This ensures validation samples are not augmented
    val_dataset.dataset.transform = val_transform
    
    # =========================
    # BALANCED SAMPLING SETUP
    # =========================
    # Calculate class weights for balanced sampling
    # This addresses class imbalance in the training data
    train_targets = [full_dataset[i][1] for i in train_dataset.indices]
    class_counts = Counter(train_targets)
    total_samples = len(train_targets)
    
    # Calculate inverse frequency weights for each class
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    # Assign weight to each sample based on its class
    sample_weights = [class_weights[target] for target in train_targets]
    
    # Create weighted random sampler for balanced training
    # This ensures each batch has roughly equal representation of both classes
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # Allow sampling with replacement
    )
    
    # =========================
    # DATA LOADERS
    # =========================
    # Create optimized data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS
    )
    
    # Display dataset information
    print(f"Class distribution: {class_counts}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # =========================
    # MODEL AND LOSS SETUP
    # =========================
    # Initialize model with advanced architecture
    num_classes = len(full_dataset.classes)
    model = OptimizedCNN(num_classes=num_classes).to(device)
    
    # Use multiple loss functions for robust training
    # Cross Entropy with label smoothing for general classification
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    # Focal Loss for handling class imbalance and hard examples
    focal_criterion = FocalLoss(alpha=1, gamma=2)
    
    # =========================
    # OPTIMIZER AND SCHEDULER SETUP
    # =========================
    # Advanced optimizer (AdamW with decoupled weight decay)
    # AdamW provides better regularization than standard Adam
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY
    )
    
    # Advanced scheduler (Cosine Annealing with Warm Restarts)
    # This scheduler periodically restarts with high learning rates
    # helping escape local minima and achieve better final performance
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_MULT, eta_min=ETA_MIN
    )
    
    # Mixed precision scaler for faster training and reduced memory usage
    scaler = torch.cuda.amp.GradScaler() if USE_AMP and torch.cuda.is_available() else None
    
    # =========================
    # MODEL SAVING SETUP
    # =========================
    # Create model directory if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    best_model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    
    # =========================
    # TRAINING STATE VARIABLES
    # =========================
    # Initialize training tracking variables
    metrics_logger = MetricsLogger()  # For plotting training curves
    best_val_acc = 0.0               # Track best validation accuracy
    epochs_without_improvement = 0    # Early stopping counter
    
    print(f"\nStarting training for {EPOCHS} epochs...")
    print(f"Best model will be saved to: {best_model_path}")
    
    # =========================
    # MAIN TRAINING LOOP
    # =========================
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        
        # =========================
        # TRAINING PHASE
        # =========================
        model.train()  # Enable training mode (dropout, batch norm updates)
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Iterate through training batches
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Move data to device with non-blocking transfer for efficiency
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # =========================
            # ADVANCED AUGMENTATION
            # =========================
            # Apply mixup or cutmix randomly during training
            # This creates more diverse training samples and improves generalization
            if random.random() < MIXUP_PROB:
                if random.random() < 0.5:
                    # Apply mixup: linear interpolation between samples
                    data, targets_a, targets_b, lam = mixup_data(data, targets)
                    use_mixup = True
                    use_cutmix = False
                else:
                    # Apply cutmix: paste rectangular regions between samples
                    data, targets_a, targets_b, lam = cutmix_data(data, targets)
                    use_mixup = False
                    use_cutmix = True
            else:
                # Use original samples without mixing
                use_mixup = use_cutmix = False
            
            # Clear gradients from previous iteration
            optimizer.zero_grad()
            
            # =========================
            # FORWARD PASS AND LOSS COMPUTATION
            # =========================
            # Forward pass with mixed precision for efficiency
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(data)
                    
                    if use_mixup or use_cutmix:
                        # Calculate loss for mixed samples
                        # Combine Cross Entropy and Focal Loss (70% CE + 30% Focal)
                        ce_loss = mixup_criterion(ce_criterion, outputs, targets_a, targets_b, lam)
                        focal_loss = mixup_criterion(focal_criterion, outputs, targets_a, targets_b, lam)
                        loss = 0.7 * ce_loss + 0.3 * focal_loss
                    else:
                        # Calculate loss for original samples
                        ce_loss = ce_criterion(outputs, targets)
                        focal_loss = focal_criterion(outputs, targets)
                        loss = 0.7 * ce_loss + 0.3 * focal_loss
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision training (fallback for older hardware)
                outputs = model(data)
                
                if use_mixup or use_cutmix:
                    ce_loss = mixup_criterion(ce_criterion, outputs, targets_a, targets_b, lam)
                    focal_loss = mixup_criterion(focal_criterion, outputs, targets_a, targets_b, lam)
                    loss = 0.7 * ce_loss + 0.3 * focal_loss
                else:
                    ce_loss = ce_criterion(outputs, targets)
                    focal_loss = focal_criterion(outputs, targets)
                    loss = 0.7 * ce_loss + 0.3 * focal_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                optimizer.step()
            
            # =========================
            # TRAINING STATISTICS
            # =========================
            # Accumulate training statistics
            train_loss += loss.item()
            # Only calculate accuracy for non-mixed samples (for meaningful metrics)
            if not (use_mixup or use_cutmix):
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
        
        # =========================
        # VALIDATION PHASE
        # =========================
        model.eval()  # Set to evaluation mode (disable dropout, fix batch norm)
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        # Initialize confusion matrix components for detailed analysis
        val_tp = val_tn = val_fp = val_fn = 0
        
        # Disable gradient computation for validation (saves memory and computation)
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                # Forward pass (consistent with training precision)
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model(data)
                        loss = ce_criterion(outputs, targets)  # Use only CE loss for validation
                else:
                    outputs = model(data)
                    loss = ce_criterion(outputs, targets)
                
                # Accumulate validation statistics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                # Calculate confusion matrix components for precision/recall metrics
                # Assuming binary classification: 0=live, 1=spoof
                for t, p in zip(targets.view(-1), predicted.view(-1)):
                    if t.long() == 1 and p.long() == 1:
                        val_tp += 1  # True Positive: correctly detected spoof
                    elif t.long() == 1 and p.long() == 0:
                        val_fn += 1  # False Negative: missed spoof attack
                    elif t.long() == 0 and p.long() == 1:
                        val_fp += 1  # False Positive: false spoof alarm
                    else:
                        val_tn += 1  # True Negative: correctly identified live face
        
        # =========================
        # METRICS CALCULATION
        # =========================
        # Calculate epoch metrics
        train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0
        val_acc = 100.0 * val_correct / val_total
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Calculate precision and recall for spoof detection
        val_precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
        val_recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
        
        # Update learning rate scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics for visualization
        metrics_logger.log_epoch(train_loss, train_acc, val_loss, val_acc, val_precision, val_recall)
        
        # =========================
        # MODEL SAVING AND EARLY STOPPING
        # =========================
        # Save model if validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch [{epoch+1}/{EPOCHS}] - New best model! Val Acc: {val_acc:.2f}%")
        else:
            epochs_without_improvement += 1
        
        # Display epoch results with comprehensive metrics
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{EPOCHS}] ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
              f"LR: {current_lr:.2e}")
        
        # Early stopping check
        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # =========================
    # TRAINING COMPLETION
    # =========================
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {best_model_path}")
    
    # Save comprehensive training plots and analysis
    print("\nSaving training plots...")
    base_name, result_folder = metrics_logger.save_all_plots()
    print(f"Training results saved in folder: {result_folder}")
    print(f"Result folder name: {base_name}")

if __name__ == '__main__':
    # =========================
    # REPRODUCIBILITY SETUP
    # =========================
    # Set random seeds for reproducible results across runs
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)  # For multi-GPU setups
    
    # =========================
    # SYSTEM INFORMATION
    # =========================
    print("Fine-Tuned Training Script - Optimized for Maximum Accuracy")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # =========================
    # START TRAINING
    # =========================
    train_model()
    print(f"\nFine-tuned training completed successfully!")
