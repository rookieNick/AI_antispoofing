import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time
from collections import Counter
from model import OptimizedCNN
from plot_utils import MetricsLogger

# ======================== CONFIGURATION VARIABLES ========================
# Training Parameters
BATCH_SIZE = 128              # Batch size for training
IMAGE_SIZE = (112, 112)       # Input image size (height, width)
EPOCHS = 50                   # Maximum number of training epochs
LEARNING_RATE = 0.001         # Initial learning rate
WEIGHT_DECAY = 0.01           # L2 regularization strength
SAMPLE_LIMIT = 200            # Limit training dataset to this many samples; set to -1 to use all samples

# Early Stopping & Scheduling
PATIENCE = 10                 # Early stopping patience (epochs)
LABEL_SMOOTHING = 0.1         # Label smoothing factor
GRADIENT_CLIP_NORM = 1.0      # Gradient clipping max norm

# Data Loading
NUM_WORKERS = 6               # Number of data loading workers
PIN_MEMORY = True             # Pin memory for faster GPU transfer
PERSISTENT_WORKERS = True     # Keep workers alive between epochs

# Data Augmentation Parameters
HORIZONTAL_FLIP_PROB = 0.5    # Probability of horizontal flip
ROTATION_DEGREES = 10         # Max rotation degrees
COLOR_JITTER_BRIGHTNESS = 0.2 # Color jitter brightness
COLOR_JITTER_CONTRAST = 0.2   # Color jitter contrast
COLOR_JITTER_SATURATION = 0.2 # Color jitter saturation
COLOR_JITTER_HUE = 0.1        # Color jitter hue
GRAYSCALE_PROB = 0.1          # Probability of converting to grayscale
RANDOM_ERASING_PROB = 0.1     # Probability of random erasing

# Optimizer Parameters
ADAM_BETA1 = 0.9              # Adam optimizer beta1
ADAM_BETA2 = 0.999            # Adam optimizer beta2
ADAM_EPS = 1e-8               # Adam optimizer epsilon

# Learning Rate Scheduler Parameters
LR_PCT_START = 0.1            # OneCycleLR warmup percentage
LR_ANNEAL_STRATEGY = 'cos'    # OneCycleLR annealing strategy

# Model Saving
MODEL_FILENAME = 'cnn_pytorch.pth'

# ======================== END CONFIGURATION ========================

# --- GPU Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Define dataset paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.normpath(os.path.join(script_dir, "..", "dataset", "casia-fasd", "train"))

def train_model():
    print(f"Training directory: {train_dir}")
    print(f"Using batch size: {BATCH_SIZE}")
    print(f"Using image size: {IMAGE_SIZE}")
    
    # Data transforms (enhanced augmentation for better generalization)
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=HORIZONTAL_FLIP_PROB),
        transforms.RandomRotation(degrees=ROTATION_DEGREES),
        transforms.ColorJitter(brightness=COLOR_JITTER_BRIGHTNESS, contrast=COLOR_JITTER_CONTRAST, 
                              saturation=COLOR_JITTER_SATURATION, hue=COLOR_JITTER_HUE),
        transforms.RandomGrayscale(p=GRAYSCALE_PROB),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=RANDOM_ERASING_PROB, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])
    
    # Load training dataset
    print("Loading training dataset...")
    full_train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    # If SAMPLE_LIMIT > 0, limit training dataset; if -1, use full dataset
    if SAMPLE_LIMIT > 0:
        indices = torch.randperm(len(full_train_dataset))[:SAMPLE_LIMIT]
        limited_train_dataset = torch.utils.data.Subset(full_train_dataset, indices)
    else:
        limited_train_dataset = full_train_dataset
    # Split training set into train and validation
    train_size = int(0.8 * len(limited_train_dataset))
    val_size = len(limited_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        limited_train_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders with optimizations
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)
    
    # Get class names and info
    class_names = full_train_dataset.classes
    num_classes = len(class_names)
    print(f"Class names: {class_names}")
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Compute class weights for imbalanced data
    print("Computing class weights...")
    # Use targets attribute for fast label access
    all_labels = full_train_dataset.targets
    label_counts = Counter(all_labels)
    print(f"Label counts: {label_counts}")
    total = sum(label_counts.values())
    class_weights = torch.tensor([total/(num_classes*label_counts[i]) for i in range(num_classes)], dtype=torch.float32)
    class_weights = class_weights.to(device)
    print(f"Class weights: {class_weights}")
    
    # Initialize model
    model = OptimizedCNN(num_classes=num_classes).to(device)
    print(f"Model moved to device: {device}")
    # Only use torch.compile if GPU supports CUDA capability >= 7.0
    use_compile = False
    if torch.cuda.is_available():
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
    
    # Loss function with class weights and optimized optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, 
                           betas=(ADAM_BETA1, ADAM_BETA2), eps=ADAM_EPS)
    
    # Enhanced learning rate scheduler with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, 
                                            steps_per_epoch=len(train_loader), epochs=EPOCHS,
                                            pct_start=LR_PCT_START, anneal_strategy=LR_ANNEAL_STRATEGY)
    
    # Gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # Training variables
    best_val_acc = 0.0
    # Save model in 'model' folder
    model_dir = os.path.join(script_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, MODEL_FILENAME)
    patience_counter = 0
    
    print("Starting training...")
    start_time = time.time()
    
    # Initialize metrics logger
    metrics_logger = MetricsLogger()
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Mixed precision training
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)  # Gradient clipping
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)  # Gradient clipping
                optimizer.step()
            
            scheduler.step()  # Step scheduler every batch for OneCycleLR
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_acc = 100.0 * train_correct / train_total
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_tp = val_tn = val_fp = val_fn = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                # Use mixed precision for validation too
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model(data)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                # Calculate precision/recall metrics
                for t, p in zip(targets.view(-1), predicted.view(-1)):
                    if t.long() == 1 and p.long() == 1:
                        val_tp += 1
                    elif t.long() == 1 and p.long() == 0:
                        val_fn += 1
                    elif t.long() == 0 and p.long() == 1:
                        val_fp += 1
                    else:
                        val_tn += 1
        
        val_acc = 100.0 * val_correct / val_total
        val_loss /= len(val_loader)
        
        val_precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
        val_recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
        
        # No need to step scheduler here since OneCycleLR steps every batch
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{EPOCHS} ({epoch_time:.1f}s): "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
              f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
        
        # Log metrics for plotting
        metrics_logger.log_epoch(train_loss, train_acc, val_loss, val_acc, val_precision, val_recall)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time/60:.1f} minutes")
    print(f"Best model saved to: {best_model_path}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save training plots
    print("\nSaving training plots...")
    base_name, result_folder = metrics_logger.save_all_plots()
    print(f"Training results saved in folder: {result_folder}")
    print(f"Result folder name: {base_name}")

if __name__ == '__main__':
    # Enable CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster training
        torch.backends.cudnn.allow_tf32 = True
    
    print("PyTorch version:", torch.__version__)
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    train_model()
