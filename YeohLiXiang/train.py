# ==============================================================================
# CNN Face Anti-Spoofing Training Script
# ==============================================================================
# This script trains a CNN model for face anti-spoofing (detecting live vs fake faces)
# using the CASIA-FASD dataset. It includes various optimization techniques like:
# - Mixed precision training for faster GPU utilization
# - Advanced data augmentation for better generalization
# - Class weighting to handle imbalanced datasets
# - Early stopping to prevent overfitting
# - Comprehensive metrics logging and visualization
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time
from collections import Counter
from model import OptimizedCNN  # Custom CNN architecture
from plot_utils import MetricsLogger  # Custom plotting utilities

# ======================== CONFIGURATION VARIABLES ========================
# This section contains all hyperparameters and settings that control training behavior.
# Modify these values to experiment with different training configurations.

# Training Parameters
BATCH_SIZE = 128              # Number of samples processed in each training step
                              # Larger batch sizes improve GPU utilization but require more memory
IMAGE_SIZE = (112, 112)       # Input image dimensions (height, width) in pixels
                              # Smaller images train faster but may lose important details
EPOCHS = 50                   # Maximum number of complete passes through the training dataset
LEARNING_RATE = 0.001         # Step size for gradient descent optimization
                              # Higher values train faster but may overshoot optimal weights
WEIGHT_DECAY = 0.01           # L2 regularization strength to prevent overfitting
                              # Higher values prevent overfitting but may underfit
SAMPLE_LIMIT = -1            # Limit training dataset size; -1 uses all available samples
                              # Useful for quick experiments with smaller datasets

# Early Stopping & Scheduling Parameters
PATIENCE = 10                 # Number of epochs to wait without improvement before stopping
                              # Prevents wasting time when model stops learning
LABEL_SMOOTHING = 0.1         # Smoothing factor for one-hot encoded labels (0.0-1.0)
                              # Helps prevent overconfident predictions and improves generalization
GRADIENT_CLIP_NORM = 1.0      # Maximum allowed gradient norm to prevent exploding gradients
                              # Stabilizes training of deep neural networks

# Data Loading Configuration
NUM_WORKERS = 6               # Number of parallel processes for data loading
                              # More workers load data faster but consume more CPU/memory
PIN_MEMORY = True             # Keep data in GPU memory for faster transfer
                              # Only beneficial when using GPU training
PERSISTENT_WORKERS = True     # Keep data loading workers alive between epochs
                              # Reduces overhead from repeatedly creating/destroying processes

# Data Augmentation Parameters
# These parameters control random transformations applied to training images
# to increase dataset diversity and improve model generalization
HORIZONTAL_FLIP_PROB = 0.5    # Probability of horizontally flipping images
ROTATION_DEGREES = 10         # Maximum rotation angle in degrees (±10°)
COLOR_JITTER_BRIGHTNESS = 0.2 # Random brightness adjustment factor
COLOR_JITTER_CONTRAST = 0.2   # Random contrast adjustment factor
COLOR_JITTER_SATURATION = 0.2 # Random saturation adjustment factor
COLOR_JITTER_HUE = 0.1        # Random hue shift factor
GRAYSCALE_PROB = 0.1          # Probability of converting images to grayscale
RANDOM_ERASING_PROB = 0.1     # Probability of randomly erasing rectangular regions

# Optimizer Parameters (Adam optimizer configuration)
ADAM_BETA1 = 0.9              # Exponential decay rate for first moment estimates
ADAM_BETA2 = 0.999            # Exponential decay rate for second moment estimates
ADAM_EPS = 1e-8               # Small constant for numerical stability

# Learning Rate Scheduler Parameters (OneCycleLR configuration)
LR_PCT_START = 0.1            # Percentage of cycle spent in warmup phase
LR_ANNEAL_STRATEGY = 'cos'    # Learning rate annealing strategy ('cos' or 'linear')

# Model Saving Configuration
MODEL_FILENAME = 'cnn_pytorch.pth'  # Filename for saved model weights

# ======================== END CONFIGURATION ========================

# --- GPU Setup ---
# Automatically detect and configure the best available computing device
# CUDA (GPU) is preferred for faster training, fallback to CPU if unavailable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Define dataset paths ---
# Construct absolute paths to ensure the script works regardless of current working directory
script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory containing this script
train_dir = os.path.normpath(os.path.join(script_dir, "..", "dataset", "casia-fasd", "train"))

def train_model():
    """
    Main training function that orchestrates the entire training process.
    
    This function handles:
    1. Data loading and preprocessing
    2. Model initialization and optimization setup
    3. Training loop with validation
    4. Model saving and metrics logging
    
    Returns:
        None (saves model and plots to disk)
    """
    print(f"Training directory: {train_dir}")
    print(f"Using batch size: {BATCH_SIZE}")
    print(f"Using image size: {IMAGE_SIZE}")
    
    # Data transforms (enhanced augmentation for better generalization)
    # Training transform applies various augmentations to increase data diversity
    train_transform = transforms.Compose([
        # Resize images to target size
        transforms.Resize(IMAGE_SIZE),
        # Random horizontal flip - faces can appear from either direction
        transforms.RandomHorizontalFlip(p=HORIZONTAL_FLIP_PROB),
        # Random rotation to handle slight head tilts
        transforms.RandomRotation(degrees=ROTATION_DEGREES),
        # Color jitter simulates different lighting conditions
        transforms.ColorJitter(brightness=COLOR_JITTER_BRIGHTNESS, contrast=COLOR_JITTER_CONTRAST, 
                              saturation=COLOR_JITTER_SATURATION, hue=COLOR_JITTER_HUE),
        # Occasional grayscale conversion reduces color dependency
        transforms.RandomGrayscale(p=GRAYSCALE_PROB),
        # Convert PIL images to PyTorch tensors
        transforms.ToTensor(),
        # Normalize using ImageNet statistics (common practice for transfer learning)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Random erasing simulates occlusion and improves robustness
        transforms.RandomErasing(p=RANDOM_ERASING_PROB, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])
    
    # Load training dataset
    print("Loading training dataset...")
    # ImageFolder automatically creates class labels based on subdirectory names
    # Expected structure: train_dir/live/*.png and train_dir/spoof/*.png
    full_train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    
    # Optionally limit dataset size for faster experimentation
    if SAMPLE_LIMIT > 0:
        # Randomly sample a subset of the full dataset
        indices = torch.randperm(len(full_train_dataset))[:SAMPLE_LIMIT]
        limited_train_dataset = torch.utils.data.Subset(full_train_dataset, indices)
    else:
        # Use the complete dataset
        limited_train_dataset = full_train_dataset
    
    # Split training set into train and validation subsets (80/20 split)
    # This provides an independent validation set for monitoring overfitting
    train_size = int(0.8 * len(limited_train_dataset))
    val_size = len(limited_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        limited_train_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducible splits
    )
    
    # Create data loaders with optimizations for faster training
    # DataLoader handles batching, shuffling, and parallel data loading
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)
    
    # Get class names and dataset information
    class_names = full_train_dataset.classes  # ['live', 'spoof'] typically
    num_classes = len(class_names)
    print(f"Class names: {class_names}")
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Compute class weights for imbalanced data handling
    # This addresses the common problem where one class has significantly more samples
    print("Computing class weights...")
    # Use targets attribute for fast label access without loading images
    all_labels = full_train_dataset.targets
    label_counts = Counter(all_labels)
    print(f"Label counts: {label_counts}")
    total = sum(label_counts.values())
    # Calculate inverse frequency weights: rare classes get higher weights
    class_weights = torch.tensor([total/(num_classes*label_counts[i]) for i in range(num_classes)], dtype=torch.float32)
    class_weights = class_weights.to(device)
    print(f"Class weights: {class_weights}")
    
    # Initialize the CNN model and move to appropriate device (GPU/CPU)
    model = OptimizedCNN(num_classes=num_classes).to(device)
    print(f"Model moved to device: {device}")
    
    # Enable torch.compile for faster training on compatible GPUs
    # torch.compile optimizes the model graph for better performance
    use_compile = False
    if torch.cuda.is_available():
        # Check GPU compute capability (7.0+ required for optimal torch.compile support)
        cap_major, cap_minor = torch.cuda.get_device_capability()
        if cap_major >= 7:
            use_compile = True
    if use_compile:
        try:
            model = torch.compile(model)
            print("Model compiled for faster training")
        except Exception as e:
            print(f"torch.compile failed: {e}\nUsing regular model.")
    else:
        print("torch.compile not used: GPU capability too old or not available.")
    
    # Configure loss function and optimizer
    # CrossEntropyLoss with class weights and label smoothing for robust training
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    
    # AdamW optimizer with weight decay for better generalization
    # AdamW decouples weight decay from gradient-based update, improving performance
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, 
                           betas=(ADAM_BETA1, ADAM_BETA2), eps=ADAM_EPS)
    
    # OneCycleLR scheduler for optimal learning rate scheduling
    # This scheduler gradually increases LR to max, then decreases it following a cosine curve
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, 
                                            steps_per_epoch=len(train_loader), epochs=EPOCHS,
                                            pct_start=LR_PCT_START, anneal_strategy=LR_ANNEAL_STRATEGY)
    
    # Mixed precision training setup for faster training on modern GPUs
    # GradScaler prevents gradient underflow when using 16-bit floating point
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # Training state variables
    best_val_acc = 0.0  # Track best validation accuracy for model saving
    # Create model directory if it doesn't exist
    model_dir = os.path.join(script_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, MODEL_FILENAME)
    patience_counter = 0  # Counter for early stopping
    
    print("Starting training...")
    start_time = time.time()
    
    # Initialize metrics logger for tracking and plotting training progress
    metrics_logger = MetricsLogger()
    
    # Main training loop
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # =========================
        # TRAINING PHASE
        # =========================
        model.train()  # Set model to training mode (enables dropout, batch norm updates)
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Iterate through all training batches
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Move data to GPU for faster computation (non_blocking for async transfer)
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # Clear gradients from previous iteration (set_to_none is more efficient)
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision training for speed and memory efficiency
            if scaler is not None:
                # Use automatic mixed precision (AMP) for CUDA devices
                with torch.amp.autocast('cuda'):
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()  # Scale loss to prevent gradient underflow
                scaler.unscale_(optimizer)     # Unscale gradients for clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
                scaler.step(optimizer)         # Update model parameters
                scaler.update()                # Update scaler state
            else:
                # Standard precision training for CPU or older GPUs
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
                optimizer.step()
            
            # Update learning rate (OneCycleLR steps every batch)
            scheduler.step()
            
            # Accumulate training statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)  # Get class with highest probability
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()  # Count correct predictions
        
        # Calculate training metrics for this epoch
        train_acc = 100.0 * train_correct / train_total
        train_loss /= len(train_loader)  # Average loss per batch
        
        # =========================
        # VALIDATION PHASE
        # =========================
        model.eval()  # Set model to evaluation mode (disables dropout, fixes batch norm)
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        # Initialize confusion matrix components for detailed metrics
        val_tp = val_tn = val_fp = val_fn = 0  # True/False Positives/Negatives
        
        # Disable gradient computation for validation (saves memory and computation)
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                # Forward pass (use mixed precision for consistency)
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model(data)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                
                # Accumulate validation statistics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                # Calculate confusion matrix components for precision/recall
                # Assuming binary classification: 0=live, 1=spoof
                for t, p in zip(targets.view(-1), predicted.view(-1)):
                    if t.long() == 1 and p.long() == 1:
                        val_tp += 1  # True Positive: correctly identified spoof
                    elif t.long() == 1 and p.long() == 0:
                        val_fn += 1  # False Negative: missed spoof (classified as live)
                    elif t.long() == 0 and p.long() == 1:
                        val_fp += 1  # False Positive: false alarm (live classified as spoof)
                    else:
                        val_tn += 1  # True Negative: correctly identified live
        
        # Calculate validation metrics
        val_acc = 100.0 * val_correct / val_total
        val_loss /= len(val_loader)  # Average validation loss
        
        # Calculate precision and recall for spoof detection
        val_precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
        val_recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
        
        # Note: No need to step scheduler here since OneCycleLR steps every batch
        
        # Display epoch results
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{EPOCHS} ({epoch_time:.1f}s): "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
              f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
        
        # Log metrics for visualization and analysis
        metrics_logger.log_epoch(train_loss, train_acc, val_loss, val_acc, val_precision, val_recall)
        
        # =========================
        # MODEL SAVING & EARLY STOPPING
        # =========================
        # Save model if validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0  # Reset early stopping counter
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1  # Increment patience counter
        
        # Early stopping: stop training if no improvement for PATIENCE epochs
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # =========================
    # TRAINING COMPLETION
    # =========================
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time/60:.1f} minutes")
    print(f"Best model saved to: {best_model_path}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save comprehensive training plots and metrics
    print("\nSaving training plots...")
    base_name, result_folder = metrics_logger.save_all_plots()
    print(f"Training results saved in folder: {result_folder}")
    print(f"Result folder name: {base_name}")

if __name__ == '__main__':
    # =========================
    # CUDA OPTIMIZATIONS
    # =========================
    # Enable CUDA optimizations for faster training on compatible hardware
    if torch.cuda.is_available():
        # Enable cuDNN benchmark mode for optimized convolution algorithms
        torch.backends.cudnn.benchmark = True
        # Allow TensorFloat-32 (TF32) for faster matrix operations on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Display system information
    print("PyTorch version:", torch.__version__)
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Start the training process
    train_model()
