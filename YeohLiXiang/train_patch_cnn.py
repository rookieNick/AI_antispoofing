# ==============================================================================
# Patch-based CNN Training Script for Face Anti-Spoofing
# ==============================================================================
# This script trains patch-based CNN models for face anti-spoofing detection.
# It supports both basic patch CNN and enhanced patch-depth CNN architectures.
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time
import numpy as np # Added for comprehensive metrics
from collections import Counter
from sklearn.metrics import confusion_matrix, f1_score, mean_squared_error # Added for comprehensive metrics
from model_patch_cnn import create_patch_cnn  # Patch-based CNN models
from plot_utils import MetricsLogger  # Custom plotting utilities

# ======================== CONFIGURATION VARIABLES ========================

# Training Parameters
BATCH_SIZE = 128              # Increased batch size for faster training
IMAGE_SIZE = (112, 112)       # Input image dimensions
EPOCHS = 30                   # Maximum number of training epochs
LEARNING_RATE = 0.0001        # Lower learning rate for better convergence
WEIGHT_DECAY = 0.01           # L2 regularization strength
SAMPLE_LIMIT = 5000         # Limit training dataset size to first 10,000 images

# Patch-specific Parameters
PATCH_SIZE = 32               # Size of each patch (32x32 pixels)
NUM_PATCHES = 9               # Number of patches to extract (3x3 grid)
MODEL_TYPE = 'patch'          # Use simpler 'patch' model instead of 'patch_depth'

# Early Stopping & Scheduling Parameters
PATIENCE = 5                 # Increased patience for patch models
LABEL_SMOOTHING = 0.0         # Disable label smoothing for initial runs
GRADIENT_CLIP_NORM = 1.0      # Gradient clipping

# Data Loading Configuration
NUM_WORKERS = 8               # Reduced workers due to patch processing
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# Data Augmentation Parameters
HORIZONTAL_FLIP_PROB = 0.0
ROTATION_DEGREES = 0
COLOR_JITTER_BRIGHTNESS = 0.0
COLOR_JITTER_CONTRAST = 0.0
COLOR_JITTER_SATURATION = 0.0
COLOR_JITTER_HUE = 0.0
GRAYSCALE_PROB = 0.0
RANDOM_ERASING_PROB = 0.0

# Optimizer Parameters
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

# Learning Rate Scheduler Parameters
LR_PCT_START = 0.1
LR_ANNEAL_STRATEGY = 'cos'

# Model Saving Configuration
MODEL_FILENAME = 'patch_cnn.pth'

# ======================== END CONFIGURATION ========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset paths
script_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.normpath(os.path.join(script_dir, "..", "dataset", "casia-fasd", "train"))

def train_patch_model():
    """
    Main training function for patch-based CNN models
    """
    print("=" * 80)
    print(f"TRAINING {MODEL_TYPE.upper()} PATCH-BASED CNN FOR FACE ANTI-SPOOFING")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model Type: {MODEL_TYPE}")
    print(f"Patch Size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Number of Patches: {NUM_PATCHES}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Dataset: {train_dir}")
    print("=" * 80)
    
    # --- Data Loading and Preprocessing ---
    print("\\n📂 Loading and preprocessing data...")
    
    # Training data transforms (simple, no heavy augmentation)
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation data transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load training dataset
    full_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    
    # Apply sample limit if specified
    if SAMPLE_LIMIT > 0:
        indices = torch.randperm(len(full_dataset))[:SAMPLE_LIMIT]
        full_dataset = torch.utils.data.Subset(full_dataset, indices)
    
    # Split into train and validation sets (80/20 split)
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS
    )
    
    # Get class information
    # Fix: get class names from underlying dataset if using Subset
    if isinstance(full_dataset, torch.utils.data.Subset):
        class_names = full_dataset.dataset.classes
    else:
        class_names = full_dataset.classes
    num_classes = len(class_names)
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   Classes: {class_names}")
    print(f"   Training samples: {len(train_dataset):,}")
    print(f"   Validation samples: {len(val_dataset):,}")
    print(f"   Total samples: {dataset_size:,}")
    
    # Calculate class distribution for weighted loss
    if hasattr(full_dataset, 'targets'):
        class_counts = Counter(full_dataset.targets)
    else:
        # For subset datasets, we need to get targets differently
        all_targets = [full_dataset.dataset.targets[i] for i in full_dataset.indices] if hasattr(full_dataset, 'indices') else [full_dataset.targets[i] for i in range(len(full_dataset))]
        class_counts = Counter(all_targets)
    
    total_samples = sum(class_counts.values())
    class_weights = [total_samples / class_counts[i] for i in range(num_classes)]
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"   Class distribution: {dict(class_counts)}")
    print(f"   Class weights: {class_weights.cpu().numpy()}")
    
    # --- Model Setup ---
    print(f"\\n🏗️  Setting up {MODEL_TYPE} patch-based CNN model...")
    
    model = create_patch_cnn(
        model_type=MODEL_TYPE,
        num_classes=num_classes,
        patch_size=PATCH_SIZE,
        num_patches=NUM_PATCHES,
        dropout_rate=0.5
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model created successfully!")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # --- Loss Function and Optimizer ---
    print("\\n⚙️  Configuring training setup...")
    
    # Loss function with class weights and label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    
    # Optimizer (no scheduler, constant LR)
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPS
    )
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    print(f"✅ Training setup complete!")
    print(f"   Loss function: CrossEntropyLoss with class weights and label smoothing")
    print(f"   Optimizer: Adam (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
    print(f"   Scheduler: None (constant LR)")
    print(f"   Mixed precision: {'Enabled' if scaler else 'Disabled'}")
    
    # --- Training Setup ---
    metrics_logger = MetricsLogger()
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    print(f"\\n🚀 Starting training for {EPOCHS} epochs...")
    print("-" * 80)
    
    # --- Training Loop ---
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                optimizer.step()
            
            # No scheduler step
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            # Progress update every 20 batches
            if batch_idx % 20 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                # Check gradient norms for debugging
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                print(f"Epoch {epoch+1:3d}, Batch {batch_idx:4d}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}, Grad Norm: {total_norm:.4f}")
        
        # Calculate training metrics
        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_targets = []
        all_val_probs = []
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(data)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_val_probs.extend(probs.cpu().numpy())

                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_targets.extend(targets.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        
        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score
        val_precision = precision_score(all_val_targets, all_val_preds, average='weighted', zero_division=0)
        val_recall = recall_score(all_val_targets, all_val_preds, average='weighted', zero_division=0)
        
        # Calculate additional metrics for comprehensive plotting
        cm = confusion_matrix(all_val_targets, all_val_preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0) # Handle cases where only one class is present
        
        val_f1 = f1_score(all_val_targets, all_val_preds, average='weighted', zero_division=0)
        val_mse = mean_squared_error(all_val_targets, all_val_probs)
        val_rmse = np.sqrt(val_mse)
        
        # Determine correct predictions for confidence distribution
        correct_predictions = (np.array(all_val_preds) == np.array(all_val_targets)).tolist()

        # Log metrics
        metrics_logger.log_epoch(train_loss, train_acc, val_loss, val_acc, val_precision, val_recall)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1:3d}/{EPOCHS} - {epoch_time:.1f}s - "
              f"Train: Loss {train_loss:.4f}, Acc {train_acc:.2f}% - "
              f"Val: Loss {val_loss:.4f}, Acc {val_acc:.2f}%, "
              f"Prec {val_precision:.3f}, Rec {val_recall:.3f}")
        
        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            model_path = os.path.join(script_dir, 'model', MODEL_FILENAME)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"✅ New best model saved! Val Acc: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            
        # Early stopping check
        if patience_counter >= PATIENCE:
            print(f"\\n⏹️  Early stopping triggered after {epoch+1} epochs")
            break
    
    # --- Training Complete ---
    total_time = time.time() - start_time
    print("\\n" + "=" * 80)
    print("🎉 TRAINING COMPLETED!")
    print("=" * 80)
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {os.path.join(script_dir, 'model', MODEL_FILENAME)}")
    
    # Save training plots
    print("\\n📊 Saving training results...")
    base_name, result_folder = metrics_logger.save_all_plots(folder_type='patch_cnn') # Removed test_results
    print(f"Training results saved in folder: {result_folder}")
    print("\n✅ All done! Check the results_patch_cnn folder for training plots.")

if __name__ == "__main__":
    try:
        train_patch_model()
    except KeyboardInterrupt:
        print("\\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\\n❌ Training failed with error: {e}")
        raise
