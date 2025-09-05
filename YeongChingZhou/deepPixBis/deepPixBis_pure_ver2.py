import torch
import torch.nn as nn
import torchvision.models as models
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import seaborn as sns
warnings.filterwarnings("ignore")

# --- Optimized DeepPixBis Architecture ---
class DeepPixBis(nn.Module):
    def __init__(self, num_classes=1, backbone='resnet18', pretrained=True):
        super(DeepPixBis, self).__init__()
        
        # Use pretrained ResNet as backbone for better performance
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feat_dim = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feat_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the final classification layer
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Optimized pixel-wise branch with fewer operations
        self.pixel_branch = nn.Sequential(
            nn.Conv2d(feat_dim, 128, 3, padding=1),  # Reduced from 256 to 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),      # Reduced from 128 to 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),                   # Direct to 1 channel
            nn.AdaptiveAvgPool2d((14, 14))
        )
        
        # Binary classification branch
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 128),              # Reduced from 256 to 128
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Extract features
        features = self.features(x)
        
        # Binary classification
        pooled = self.global_pool(features)
        pooled = torch.flatten(pooled, 1)
        binary_output = self.classifier(pooled)
        
        # Pixel-wise prediction
        pixel_output = self.pixel_branch(features)
        
        return binary_output, pixel_output

# --- Optimized Loss Function ---
class DeepPixBisLoss(nn.Module):
    def __init__(self, pixel_weight=1.0):  # Reduced from 10.0 for balance
        super().__init__()
        self.pixel_weight = pixel_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, binary_pred, pixel_pred, binary_target, pixel_target):
        # Binary loss
        binary_target = binary_target.float().unsqueeze(1)
        binary_loss = self.bce_loss(binary_pred, binary_target)
        
        # Pixel loss
        pixel_loss = self.bce_loss(pixel_pred, pixel_target)
        
        total_loss = binary_loss + self.pixel_weight * pixel_loss
        
        return total_loss, binary_loss, pixel_loss

# --- Optimized Dataset Implementation ---
class DeepPixBisDataset(Dataset):
    def __init__(self, root_dir, transform=None, pixel_size=14, cache_images=False):
        self.root_dir = root_dir
        self.transform = transform
        self.pixel_size = pixel_size
        self.cache_images = cache_images
        self.samples = []
        self.image_cache = {}
        
        print(f"Loading dataset from: {root_dir}")
        
        # Support common folder structures
        structures = [('live', 'spoof'), ('real', 'fake'), ('1', '0')]
        
        for live_name, spoof_name in structures:
            live_dir = os.path.join(root_dir, live_name)
            spoof_dir = os.path.join(root_dir, spoof_name)
            
            if os.path.exists(live_dir) and os.path.exists(spoof_dir):
                # Load live samples
                live_files = [f for f in os.listdir(live_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for f in live_files:
                    self.samples.append((os.path.join(live_dir, f), 0))
                
                # Load spoof samples
                spoof_files = [f for f in os.listdir(spoof_dir)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for f in spoof_files:
                    self.samples.append((os.path.join(spoof_dir, f), 1))
                
                print(f"Live: {len(live_files)}, Spoof: {len(spoof_files)}")
                break
        
        print(f"Total samples: {len(self.samples)}")
        
        # Pre-cache images if enabled (use only for small datasets)
        if self.cache_images and len(self.samples) < 5000:
            print("Pre-caching images...")
            for i, (img_path, _) in enumerate(tqdm(self.samples)):
                try:
                    image = Image.open(img_path).convert('RGB')
                    self.image_cache[img_path] = image
                except:
                    self.image_cache[img_path] = Image.new('RGB', (224, 224))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image from cache or disk
        if self.cache_images and img_path in self.image_cache:
            image = self.image_cache[img_path]
        else:
            try:
                image = Image.open(img_path).convert('RGB')
            except:
                image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        
        # Generate pixel map more efficiently
        pixel_map = self._generate_pixel_map(label)
        
        return image, label, pixel_map
    
    def _generate_pixel_map(self, label):
        """Generates a pixel map based on the label more efficiently."""
        if label == 0:  # Live
            # Create base map with small noise
            pixel_map = torch.randn(1, self.pixel_size, self.pixel_size) * 0.05
            pixel_map = torch.clamp(pixel_map, 0, 0.3)  # Keep values low for live
        else:  # Spoof
            # Create base map with higher values and noise
            pixel_map = torch.ones(1, self.pixel_size, self.pixel_size) * 0.8
            pixel_map += torch.randn(1, self.pixel_size, self.pixel_size) * 0.1
            pixel_map = torch.clamp(pixel_map, 0.5, 1.0)  # Keep values high for spoof
        return pixel_map

# --- GPU Setup with Memory Management ---
def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Clear cache and set memory fraction
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        
        return device
    else:
        print("Using CPU")
        return torch.device("cpu")

# --- Optimized Training Function ---
def train_deeppixbis(model, train_loader, test_loader, device, num_epochs=30, save_dir="results"):
    criterion = DeepPixBisLoss(pixel_weight=1.0)
    
    # Use AdamW with weight decay for better optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    print(f"Training for {num_epochs} epochs...")
    print("=" * 50)
    
    results = []
    best_acc = 0.0
    patience = 0
    max_patience = 8
    
    # Mixed precision training for RTX 4050
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_binary_loss = 0
        train_pixel_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (images, labels, pixel_maps) in enumerate(pbar):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            pixel_maps = pixel_maps.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Forward pass with mixed precision
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    binary_pred, pixel_pred = model(images)
                    loss, binary_loss, pixel_loss = criterion(binary_pred, pixel_pred, labels, pixel_maps)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                binary_pred, pixel_pred = model(images)
                loss, binary_loss, pixel_loss = criterion(binary_pred, pixel_pred, labels, pixel_maps)
                loss.backward()
                optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            train_binary_loss += binary_loss.item()
            train_pixel_loss += pixel_loss.item()
            
            with torch.no_grad():
                probs = torch.sigmoid(binary_pred)
                predicted = (probs > 0.5).float()
                train_total += labels.size(0)
                train_correct += (predicted.squeeze() == labels.float()).sum().item()
            
            # Update progress every 10 batches
            if batch_idx % 10 == 0:
                acc = train_correct / train_total if train_total > 0 else 0
                pbar.set_postfix({
                    'Loss': f'{loss.item():.3f}', 
                    'Acc': f'{acc:.3f}',
                    'Bin': f'{binary_loss.item():.3f}',
                    'Pix': f'{pixel_loss.item():.3f}'
                })
        
        # Validation
        test_metrics = evaluate_model(model, test_loader, device, criterion)
        
        epoch_time = time.time() - start_time
        
        # Update learning rate
        scheduler.step(test_metrics['accuracy'])
        
        # Logging
        print(f"Epoch {epoch+1:2d}: "
              f"Train Acc={train_correct/train_total:.3f}, "
              f"Test Acc={test_metrics['accuracy']:.3f}, "
              f"AUC={test_metrics['auc']:.3f}, "
              f"Time={epoch_time:.1f}s")
        
        # Save best model
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            patience = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_metrics['accuracy'],
                'auc': test_metrics['auc'],
                'epoch': epoch,
                'test_metrics': test_metrics
            }, os.path.join(save_dir, 'deeppixbis_best.pth'))
        else:
            patience += 1
        
        # Early stopping
        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Store results
        results.append({
            'epoch': epoch + 1,
            'train_acc': train_correct/train_total,
            'test_acc': test_metrics['accuracy'],
            'test_auc': test_metrics['auc'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1'],
            'train_loss': train_loss/len(train_loader),
            'test_loss': test_metrics['loss'],
            'binary_loss': train_binary_loss/len(train_loader),
            'pixel_loss': train_pixel_loss/len(train_loader),
            'epoch_time': epoch_time,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Clear cache periodically
        if epoch % 5 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return results, best_acc

# --- Comprehensive Evaluation Function ---
def evaluate_model(model, test_loader, device, criterion):
    """Comprehensive model evaluation with detailed metrics"""
    model.eval()
    test_loss = 0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels, pixel_maps in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            pixel_maps = pixel_maps.to(device, non_blocking=True)
            
            # Forward pass with mixed precision if available
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    binary_pred, pixel_pred = model(images)
                    loss, _, _ = criterion(binary_pred, pixel_pred, labels, pixel_maps)
            else:
                binary_pred, pixel_pred = model(images)
                loss, _, _ = criterion(binary_pred, pixel_pred, labels, pixel_maps)
            
            test_loss += loss.item()
            
            probs = torch.sigmoid(binary_pred)
            predicted = (probs > 0.5).float()
            
            all_preds.extend(predicted.squeeze().cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.squeeze().cpu().numpy())
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_probs)
    
    # Classification report for precision, recall, f1
    report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'precision': report['macro avg']['precision'],
        'recall': report['macro avg']['recall'],
        'f1': report['macro avg']['f1-score'],
        'loss': test_loss / len(test_loader),
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs
    }

# --- Visualization Functions ---
def plot_training_results(results_df, save_dir):
    """Create comprehensive training plots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('DeepPixBis Training Results', fontsize=16)
    
    # Accuracy plot
    axes[0, 0].plot(results_df['epoch'], results_df['train_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(results_df['epoch'], results_df['test_acc'], 'r-', label='Test', linewidth=2)
    axes[0, 0].set_title('Accuracy Over Time')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # AUC plot
    axes[0, 1].plot(results_df['epoch'], results_df['test_auc'], 'g-', linewidth=2)
    axes[0, 1].set_title('AUC Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss plot
    axes[0, 2].plot(results_df['epoch'], results_df['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 2].plot(results_df['epoch'], results_df['test_loss'], 'r-', label='Test', linewidth=2)
    axes[0, 2].set_title('Loss Over Time')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Precision, Recall, F1
    axes[1, 0].plot(results_df['epoch'], results_df['test_precision'], 'purple', label='Precision', linewidth=2)
    axes[1, 0].plot(results_df['epoch'], results_df['test_recall'], 'orange', label='Recall', linewidth=2)
    axes[1, 0].plot(results_df['epoch'], results_df['test_f1'], 'brown', label='F1', linewidth=2)
    axes[1, 0].set_title('Classification Metrics')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Component losses
    axes[1, 1].plot(results_df['epoch'], results_df['binary_loss'], 'navy', label='Binary Loss', linewidth=2)
    axes[1, 1].plot(results_df['epoch'], results_df['pixel_loss'], 'darkred', label='Pixel Loss', linewidth=2)
    axes[1, 1].set_title('Component Losses')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 2].plot(results_df['epoch'], results_df['lr'], 'darkgreen', linewidth=2)
    axes[1, 2].set_title('Learning Rate')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_plots.png'), dpi=150, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(targets, predictions, save_dir):
    """Plot confusion matrix"""
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Live', 'Spoof'], yticklabels=['Live', 'Spoof'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.show()

# --- Main Function ---
def main():
    print("Optimized DeepPixBis Training")
    print("=" * 40)
    
    # Setup
    device = setup_device()

    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"deeppixbis_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Optimized parameters for RTX 4050
    BATCH_SIZE = 128  # Adjusted for RTX 4050 memory
    INPUT_SIZE = 224
    NUM_EPOCHS = 20
    BACKBONE = 'resnet18'
    NUM_WORKERS = 4  # Adjust based on your CPU cores
    
    print(f"Batch size: {BATCH_SIZE}, Backbone: {BACKBONE}, Workers: {NUM_WORKERS}")
    
    # Optimized transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = DeepPixBisDataset('../train', train_transform, cache_images=False)
    test_dataset = DeepPixBisDataset('../test', test_transform, cache_images=False)
    
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("Error: No data found!")
        return
    
    # Optimized data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True,  # Speed up GPU transfer
        persistent_workers=True  # Keep workers alive
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS//2,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Create model with pretrained weights
    print("\nCreating model...")
    model = DeepPixBis(num_classes=1, backbone=BACKBONE, pretrained=True)
    model = model.to(device)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train
    print("\nStarting training...")
    start_time = time.time()
    results, best_acc = train_deeppixbis(model, train_loader, test_loader, device, NUM_EPOCHS, results_dir)
    total_time = time.time() - start_time
    
    # Final evaluation
    print("\nRunning final evaluation...")
    model.load_state_dict(torch.load(os.path.join(results_dir, 'deeppixbis_best.pth'))['model_state_dict'])
    final_metrics = evaluate_model(model, test_loader, device, DeepPixBisLoss())
    
    # Print comprehensive results
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Training completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best Test Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"Final Test AUC: {final_metrics['auc']:.4f}")
    print(f"Final Test Precision: {final_metrics['precision']:.4f}")
    print(f"Final Test Recall: {final_metrics['recall']:.4f}")
    print(f"Final Test F1: {final_metrics['f1']:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, 'training_results.csv'), index=False)
    
    # Save final metrics
    final_report = {
        'best_accuracy': best_acc,
        'final_auc': final_metrics['auc'],
        'final_precision': final_metrics['precision'],
        'final_recall': final_metrics['recall'],
        'final_f1': final_metrics['f1'],
        'total_training_time': total_time,
        'total_epochs': len(results),
        'model_parameters': total_params
    }
    
    pd.DataFrame([final_report]).to_csv(os.path.join(results_dir, 'final_metrics.csv'), index=False)
    
    # Create visualizations
    print("\nGenerating plots...")
    plot_training_results(results_df, results_dir)
    plot_confusion_matrix(final_metrics['targets'], final_metrics['predictions'], results_dir)
    
    return results, model, final_metrics

# --- Optimized Inference Function ---
def predict_image(model_path, image_path, device):
    """Fast inference on single image"""
    model = DeepPixBis()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if device.type == 'cuda':
            with torch.cuda.amp.autocast():
                binary_pred, pixel_pred = model(image_tensor)
        else:
            binary_pred, pixel_pred = model(image_tensor)
            
        prob = torch.sigmoid(binary_pred).item()
        prediction = "Spoof" if prob > 0.5 else "Live"
        confidence = prob if prob > 0.5 else 1 - prob
    
    return prediction, prob, confidence

if __name__ == "__main__":
    results, model, final_metrics = main()