# ==============================================================================
# DPCNN Training Script for Face Anti-Spoofing
# ==============================================================================
# This script trains Deep Pyramid CNN models for face anti-spoofing detection.
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time
from collections import Counter
from model_dpcnn import create_dpcnn_model  # DPCNN model
from plot_utils import MetricsLogger  # Custom plotting utilities

# ======================== CONFIGURATION VARIABLES ========================

# Training Parameters
import torch.backends.cudnn as cudnn

IMAGE_SIZE = (112, 112)       # Input image size
BATCH_SIZE = 64               # Batch size for training
EPOCHS = 30                   # Maximum number of training epochs
LEARNING_RATE = 0.001         # Initial learning rate
WEIGHT_DECAY = 0.0001         # L2 regularization strength
SAMPLE_LIMIT = 5000           # Limit training dataset size for speed

# DPCNN specific Parameters
MODEL_TYPE = 'standard'       # 'standard' or 'lightweight'
DROPOUT_RATE = 0.5            # Dropout rate for regularization

# Early Stopping & Scheduling Parameters
PATIENCE = 15                 # Patience for early stopping
LABEL_SMOOTHING = 0.1         # Label smoothing factor
GRADIENT_CLIP_NORM = 1.0      # Gradient clipping

# Data Loading Configuration
NUM_WORKERS = min(8, os.cpu_count() or 4)  # Dynamic worker count based on CPU cores
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# Data Augmentation Parameters
HORIZONTAL_FLIP_PROB = 0.5
ROTATION_DEGREES = 10
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2
COLOR_JITTER_SATURATION = 0.2
COLOR_JITTER_HUE = 0.1

# Optimizer Parameters
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

# Learning Rate Scheduler Parameters
SCHEDULER_STEP_SIZE = 10
SCHEDULER_GAMMA = 0.1

# Model Saving Configuration
MODEL_FILENAME = f'dpcnn.pth'

# ======================== END CONFIGURATION ========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def unwrap_dataset(ds):
    """Unwrap nested Subset objects to return the base dataset and flattened indices."""
    indices = None
    current = ds
    while isinstance(current, torch.utils.data.Subset):
        if indices is None:
            indices = list(current.indices)
        else:
            indices = [current.indices[i] for i in indices]
        current = current.dataset
    return current, indices

# Define dataset paths
script_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.normpath(os.path.join(script_dir, "..", "dataset", "casia-fasd", "train"))

def train_dpcnn_model():
    """
    Main training function for DPCNN models
    """
    print("=" * 80)
    print(f"TRAINING {MODEL_TYPE.upper()} DPCNN FOR FACE ANTI-SPOOFING")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model Type: {MODEL_TYPE}")
    print(f"Input Size: {IMAGE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Dataset: {train_dir}")
    print("=" * 80)
    
    # --- Data Loading and Preprocessing ---
    print("\\n📂 Loading and preprocessing data...")
    
    # Training data transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=HORIZONTAL_FLIP_PROB),
        transforms.RandomRotation(degrees=ROTATION_DEGREES),
        transforms.ColorJitter(
            brightness=COLOR_JITTER_BRIGHTNESS,
            contrast=COLOR_JITTER_CONTRAST,
            saturation=COLOR_JITTER_SATURATION,
            hue=COLOR_JITTER_HUE
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation data transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.CenterCrop(IMAGE_SIZE),
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
    base_val_dataset, base_val_indices = unwrap_dataset(val_dataset)
    if hasattr(base_val_dataset, 'transform'):
        base_val_dataset.transform = val_transform
    else:
        try:
            val_dataset.dataset.transform = val_transform
        except Exception:
            pass
    
    # Enable cudnn benchmark for faster convolutions
    if device.type == 'cuda':
        try:
            cudnn.benchmark = True
            cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    # Get class information
    base_dataset, base_indices = unwrap_dataset(full_dataset)
    if hasattr(base_dataset, 'classes'):
        class_names = base_dataset.classes
    else:
        class_names = getattr(base_dataset, 'classes', [])
    num_classes = len(class_names)

    # Calculate class distribution for weighted loss and sampler
    if hasattr(base_dataset, 'targets'):
        if base_indices is None:
            targets_list = list(base_dataset.targets)
        else:
            targets_list = [base_dataset.targets[i] for i in base_indices]
        class_counts = Counter(targets_list)
    else:
        try:
            if hasattr(full_dataset, 'targets'):
                class_counts = Counter(full_dataset.targets)
            elif hasattr(full_dataset, 'indices'):
                class_counts = Counter([base_dataset.targets[i] for i in full_dataset.indices])
            else:
                labels = []
                for _, label in full_dataset:
                    labels.append(label)
                class_counts = Counter(labels)
        except Exception:
            class_counts = Counter()
    
    total_samples = sum(class_counts.values())
    class_weights = [total_samples / class_counts[i] for i in range(num_classes)]
    class_weights = torch.FloatTensor(class_weights).to(device)

    # Use WeightedRandomSampler for balanced batches
    train_base, train_indices = unwrap_dataset(train_dataset)
    if hasattr(train_base, 'targets') and train_indices is not None:
        train_targets = [train_base.targets[i] for i in train_indices]
    elif hasattr(train_base, 'targets'):
        train_targets = list(train_base.targets)
    else:
        train_targets = [label for _, label in train_dataset]

    class_sample_counts = Counter(train_targets)
    print(f"Effective train class counts: {dict(class_sample_counts)}")
    sample_weights = [1.0 / class_sample_counts[t] for t in train_targets]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Create data loaders
    dl_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
        drop_last=True
    )

    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        shuffle=False,  # Sampler handles shuffling
        **dl_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **dl_kwargs
    )
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   Classes: {class_names}")
    print(f"   Training samples: {len(train_dataset):,}")
    print(f"   Validation samples: {len(val_dataset):,}")
    print(f"   Total samples: {dataset_size:,}")
    print(f"   Class distribution: {dict(class_counts)}")
    print(f"   Class weights: {class_weights.cpu().numpy()}")
    
    # --- Model Setup ---
    print(f"\\n🏗️  Setting up {MODEL_TYPE} DPCNN model...")
    
    model = create_dpcnn_model(
        model_type=MODEL_TYPE,
        num_classes=num_classes,
        dropout_rate=DROPOUT_RATE
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
    
    # Optimizer - Adam for DPCNN
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPS
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
    
    # Mixed precision training
    try:
        if device.type == 'cuda':
            try:
                scaler = torch.amp.GradScaler(device_type='cuda')
            except TypeError:
                scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None
    except Exception:
        scaler = None
    
    print(f"✅ Training setup complete!")
    print(f"   Loss function: CrossEntropyLoss with class weights and label smoothing")
    print(f"   Optimizer: {type(optimizer).__name__}")
    print(f"   Scheduler: {type(scheduler).__name__}")
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
                with torch.amp.autocast(device_type='cuda'):
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
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            # Progress update every 20 batches
            if batch_idx % 20 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:3d}, Batch {batch_idx:4d}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
        
        # Step scheduler
        scheduler.step()
        
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
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                
                if scaler:
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = model(data)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item()
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
        
        # Log metrics
        metrics_logger.log_epoch(train_loss, train_acc, val_loss, val_acc, val_precision, val_recall)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{EPOCHS} - {epoch_time:.1f}s - LR: {current_lr:.6f} - "
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
    base_name, result_folder = metrics_logger.save_all_plots(folder_type='dpcnn')
    print(f"Training results saved in folder: {result_folder}")
    print("\n✅ All done! Check the results_dpcnn folder for training plots and metrics.")

if __name__ == "__main__":
    try:
        train_dpcnn_model()
    except KeyboardInterrupt:
        print("\\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\\n❌ Training failed with error: {e}")
        raise
