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
from model import OptimizedCNN
from plot_utils import MetricsLogger
import random

# ======================== ADVANCED CONFIGURATION ========================
# Training Parameters - Fine-tuned for maximum accuracy
BATCH_SIZE = 64               # Smaller batch size for better gradient estimates
IMAGE_SIZE = (128, 128)       # Higher resolution for better feature extraction
EPOCHS = 100                  # More epochs for better convergence
LEARNING_RATE = 0.0005        # Lower LR for fine-tuning
WEIGHT_DECAY = 0.005          # Reduced weight decay
SAMPLE_LIMIT = -1             # Use full dataset

# Advanced Training Techniques
PATIENCE = 15                 # More patience for better convergence
LABEL_SMOOTHING = 0.15        # Higher label smoothing
GRADIENT_CLIP_NORM = 0.5      # Tighter gradient clipping
MIXUP_ALPHA = 0.4             # Mixup data augmentation
CUTMIX_ALPHA = 1.0            # CutMix data augmentation
MIXUP_PROB = 0.5              # Probability of applying mixup/cutmix

# Data Loading
NUM_WORKERS = 8               # More workers for faster data loading
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# Advanced Data Augmentation
HORIZONTAL_FLIP_PROB = 0.5
ROTATION_DEGREES = 15         # Increased rotation
COLOR_JITTER_BRIGHTNESS = 0.3 # Increased color jitter
COLOR_JITTER_CONTRAST = 0.3
COLOR_JITTER_SATURATION = 0.3
COLOR_JITTER_HUE = 0.15
GRAYSCALE_PROB = 0.1
RANDOM_ERASING_PROB = 0.2     # Increased random erasing
GAUSSIAN_BLUR_PROB = 0.1      # Added Gaussian blur
ELASTIC_TRANSFORM_PROB = 0.1  # Elastic deformation

# Optimizer Parameters (AdamW for better regularization)
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

# Learning Rate Scheduler (Cosine Annealing with Warm Restarts)
T_0 = 10                      # Initial restart period
T_MULT = 2                    # Restart period multiplier
ETA_MIN = 1e-7                # Minimum learning rate

# Model Saving
MODEL_DIR = "model"
MODEL_NAME = "best_model_fine_tuned.pth"

# Mixed Precision Training
USE_AMP = True                # Automatic Mixed Precision

# Test-Time Augmentation (TTA)
TTA_ENABLED = True
TTA_CROPS = 5

# ======================== DEVICE SETUP ========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Enable optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ======================== ADVANCED DATA AUGMENTATION ========================
class GaussianBlur:
    def __init__(self, kernel_size=3, sigma_range=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
    
    def __call__(self, img):
        sigma = random.uniform(*self.sigma_range)
        return transforms.functional.gaussian_blur(img, self.kernel_size, sigma)

def get_advanced_transforms():
    """Get advanced data augmentation transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((int(IMAGE_SIZE[0] * 1.1), int(IMAGE_SIZE[1] * 1.1))),  # Slightly larger for cropping
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=HORIZONTAL_FLIP_PROB),
        transforms.RandomRotation(ROTATION_DEGREES, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ColorJitter(
            brightness=COLOR_JITTER_BRIGHTNESS,
            contrast=COLOR_JITTER_CONTRAST,
            saturation=COLOR_JITTER_SATURATION,
            hue=COLOR_JITTER_HUE
        ),
        transforms.RandomApply([GaussianBlur()], p=GAUSSIAN_BLUR_PROB),
        transforms.RandomGrayscale(p=GRAYSCALE_PROB),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=RANDOM_ERASING_PROB, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_transform

# ======================== MIXUP AND CUTMIX ========================
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=CUTMIX_ALPHA):
    """Apply cutmix augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to match the exact area ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    """Generate random bounding box for cutmix"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Calculate mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ======================== FOCAL LOSS ========================
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ======================== TRAINING FUNCTION ========================
def train_model():
    """Main training function with advanced techniques"""
    
    # Data paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "..", "dataset", "casia-fasd")
    train_dir = os.path.join(dataset_dir, "train")
    
    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found at {train_dir}")
        return
    
    print(f"Training directory: {train_dir}")
    print(f"Using batch size: {BATCH_SIZE}")
    print(f"Using image size: {IMAGE_SIZE}")
    print(f"Using epochs: {EPOCHS}")
    print(f"Using learning rate: {LEARNING_RATE}")
    print(f"Sample limit: {'All samples' if SAMPLE_LIMIT == -1 else SAMPLE_LIMIT}")
    
    # Get transforms
    train_transform, val_transform = get_advanced_transforms()
    
    # Load datasets
    print("Loading training dataset...")
    full_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    
    if SAMPLE_LIMIT > 0:
        indices = torch.randperm(len(full_dataset))[:SAMPLE_LIMIT]
        full_dataset = torch.utils.data.Subset(full_dataset, indices)
    
    # Split dataset (80% train, 20% validation)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply validation transform to validation set
    val_dataset.dataset.transform = val_transform
    
    # Calculate class weights for balanced sampling
    train_targets = [full_dataset[i][1] for i in train_dataset.indices]
    class_counts = Counter(train_targets)
    total_samples = len(train_targets)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[target] for target in train_targets]
    
    # Weighted random sampler for balanced training
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create data loaders
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
    
    print(f"Class distribution: {class_counts}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model with advanced architecture
    num_classes = len(full_dataset.classes)
    model = OptimizedCNN(num_classes=num_classes).to(device)
    
    # Use multiple loss functions
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    focal_criterion = FocalLoss(alpha=1, gamma=2)
    
    # Advanced optimizer (AdamW with weight decay)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY
    )
    
    # Advanced scheduler (Cosine Annealing with Warm Restarts)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_MULT, eta_min=ETA_MIN
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if USE_AMP and torch.cuda.is_available() else None
    
    # Model saving setup
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    best_model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    
    # Training tracking
    metrics_logger = MetricsLogger()
    best_val_acc = 0.0
    epochs_without_improvement = 0
    
    print(f"\nStarting training for {EPOCHS} epochs...")
    print(f"Best model will be saved to: {best_model_path}")
    
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # Apply mixup or cutmix randomly
            if random.random() < MIXUP_PROB:
                if random.random() < 0.5:
                    # Apply mixup
                    data, targets_a, targets_b, lam = mixup_data(data, targets)
                    use_mixup = True
                    use_cutmix = False
                else:
                    # Apply cutmix
                    data, targets_a, targets_b, lam = cutmix_data(data, targets)
                    use_mixup = False
                    use_cutmix = True
            else:
                use_mixup = use_cutmix = False
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(data)
                    
                    if use_mixup or use_cutmix:
                        # Combine CE and Focal loss for mixup/cutmix
                        ce_loss = mixup_criterion(ce_criterion, outputs, targets_a, targets_b, lam)
                        focal_loss = mixup_criterion(focal_criterion, outputs, targets_a, targets_b, lam)
                        loss = 0.7 * ce_loss + 0.3 * focal_loss
                    else:
                        ce_loss = ce_criterion(outputs, targets)
                        focal_loss = focal_criterion(outputs, targets)
                        loss = 0.7 * ce_loss + 0.3 * focal_loss
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
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
            
            # Statistics
            train_loss += loss.item()
            if not (use_mixup or use_cutmix):
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_tp = val_tn = val_fp = val_fn = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model(data)
                        loss = ce_criterion(outputs, targets)
                else:
                    outputs = model(data)
                    loss = ce_criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                # Calculate confusion matrix components
                for t, p in zip(targets.view(-1), predicted.view(-1)):
                    if t.long() == 1 and p.long() == 1:
                        val_tp += 1
                    elif t.long() == 1 and p.long() == 0:
                        val_fn += 1
                    elif t.long() == 0 and p.long() == 1:
                        val_fp += 1
                    else:
                        val_tn += 1
        
        # Calculate metrics
        train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0
        val_acc = 100.0 * val_correct / val_total
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        val_precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
        val_recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        metrics_logger.log_epoch(train_loss, train_acc, val_loss, val_acc, val_precision, val_recall)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch [{epoch+1}/{EPOCHS}] - New best model! Val Acc: {val_acc:.2f}%")
        else:
            epochs_without_improvement += 1
        
        # Print epoch results
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{EPOCHS}] ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
              f"LR: {current_lr:.2e}")
        
        # Early stopping
        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {best_model_path}")
    
    # Save training plots
    print("\nSaving training plots...")
    base_name, result_folder = metrics_logger.save_all_plots()
    print(f"Training results saved in folder: {result_folder}")
    print(f"Result folder name: {base_name}")

if __name__ == '__main__':
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    print("Fine-Tuned Training Script - Optimized for Maximum Accuracy")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    train_model()
    print(f"\nFine-tuned training completed successfully!")
