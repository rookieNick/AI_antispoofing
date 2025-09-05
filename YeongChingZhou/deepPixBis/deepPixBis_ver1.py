import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import pandas as pd
from datetime import datetime
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# --- Optimized GPU Setup ---
def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # Faster but less reproducible
        return device
    else:
        print("Using CPU")
        return torch.device("cpu")

# --- Lightweight DeepPixBis Architecture ---
class OptimizedDeepPixBis(nn.Module):
    def __init__(self, num_classes=1, backbone='resnet18', pretrained=True):
        super(OptimizedDeepPixBis, self).__init__()
        
        # Use pretrained ResNet as backbone for faster convergence
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feat_dim = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feat_dim = 512
        elif backbone == 'mobilenet_v3_small':
            self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
            feat_dim = 576
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the final classification layer
        if 'resnet' in backbone:
            self.features = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove avgpool and fc
        else:  # MobileNet
            self.features = self.backbone.features
            feat_dim = self.backbone.classifier[0].in_features
        
        # Lightweight pixel-wise branch
        self.pixel_branch = nn.Sequential(
            nn.Conv2d(feat_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),  # 1x1 conv for efficiency
            nn.AdaptiveAvgPool2d((14, 14))  # Fixed size output
        )
        
        # Binary classification branch
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.pixel_branch, self.classifier]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out')
                elif isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, 0, 0.01)
                    nn.init.constant_(layer.bias, 0)
    
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

# --- Fast Dataset Implementation ---
class FastDeepPixBisDataset(Dataset):
    def __init__(self, root_dir, transform=None, pixel_size=14, training=True):
        self.root_dir = root_dir
        self.transform = transform
        self.pixel_size = pixel_size
        self.training = training
        self.samples = []
        
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
        
        # Pre-generate pixel maps for faster training
        self.pixel_maps = {}
        self._precompute_pixel_maps()
    
    def _precompute_pixel_maps(self):
        """Pre-compute pixel maps to avoid runtime generation"""
        print("Pre-computing pixel maps...")
        
        # Live pixel map (mostly 0s)
        live_map = torch.zeros(1, self.pixel_size, self.pixel_size)
        live_map += torch.randn(1, self.pixel_size, self.pixel_size) * 0.1
        live_map = torch.clamp(live_map, 0, 1)
        self.pixel_maps[0] = live_map
        
        # Spoof pixel map (mostly 1s)
        spoof_map = torch.ones(1, self.pixel_size, self.pixel_size)
        spoof_map += torch.randn(1, self.pixel_size, self.pixel_size) * 0.15
        spoof_map = torch.clamp(spoof_map, 0, 1)
        self.pixel_maps[1] = spoof_map
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Return black image if loading fails
            image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        
        # Get pre-computed pixel map
        pixel_map = self.pixel_maps[label].clone()
        
        # Add small random variation for training
        if self.training:
            pixel_map += torch.randn_like(pixel_map) * 0.05
            pixel_map = torch.clamp(pixel_map, 0, 1)
        
        return image, label, pixel_map

# --- Simplified Loss Function ---
class FastDeepPixBisLoss(nn.Module):
    def __init__(self, pixel_weight=5.0):  # Reduced from 10.0
        super().__init__()
        self.pixel_weight = pixel_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, binary_pred, pixel_pred, binary_target, pixel_target):
        # Binary loss
        binary_target = binary_target.float().unsqueeze(1)
        binary_loss = self.bce_loss(binary_pred, binary_target)
        
        # Pixel loss (simplified)
        pixel_loss = self.bce_loss(pixel_pred, pixel_target)
        
        total_loss = binary_loss + self.pixel_weight * pixel_loss
        
        return total_loss, binary_loss, pixel_loss

# --- Optimized Training Function ---
def train_fast_deeppixbis(model, train_loader, test_loader, device, num_epochs=15):
    criterion = FastDeepPixBisLoss(pixel_weight=5.0)
    
    # Use AdamW with cosine annealing for faster convergence
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Simplified mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    print(f"Training for {num_epochs} epochs...")
    print("=" * 50)
    
    results = []
    best_acc = 0.0
    patience = 0
    max_patience = 7
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for images, labels, pixel_maps in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            pixel_maps = pixel_maps.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if scaler:
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
            probs = torch.sigmoid(binary_pred)
            predicted = (probs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted.squeeze() == labels.float()).sum().item()
            
            # Update progress
            acc = train_correct / train_total
            pbar.set_postfix({'Loss': f'{loss.item():.3f}', 'Acc': f'{acc:.3f}'})
        
        scheduler.step()
        
        # Validation
        model.eval()
        test_loss = 0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels, pixel_maps in test_loader:
                images, labels = images.to(device), labels.to(device)
                pixel_maps = pixel_maps.to(device)
                
                if scaler:
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
        
        # Metrics
        test_acc = accuracy_score(all_targets, all_preds)
        test_auc = roc_auc_score(all_targets, all_probs)
        
        epoch_time = time.time() - start_time
        
        # Logging
        print(f"Epoch {epoch+1:2d}: Train Acc={train_correct/train_total:.3f}, "
              f"Test Acc={test_acc:.3f}, AUC={test_auc:.3f}, Time={epoch_time:.1f}s")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            patience = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': test_acc,
                'auc': test_auc,
                'epoch': epoch
            }, os.path.join(results_dir, 'fast_deeppixbis_best.pth'))
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
            'test_acc': test_acc,
            'test_auc': test_auc,
            'train_loss': train_loss/len(train_loader),
            'test_loss': test_loss/len(test_loader),
            'epoch_time': epoch_time
        })
    
    return results, best_acc

# --- Main Function ---
def main():
    print("Fast DeepPixBis Training")
    print("=" * 30)
    
    # Setup
    device = setup_device()

    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join("DeepPixBis", f"fast_deeppixbis_results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Dynamic batch size based on GPU memory
    if device.type == 'cuda':
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory >= 8:
            BATCH_SIZE = 128
            BACKBONE = 'resnet18'
        elif gpu_memory >= 4:
            BATCH_SIZE = 128
            BACKBONE = 'resnet18'
        else:
            BATCH_SIZE = 16
            BACKBONE = 'mobilenet_v3_small'
    else:
        BATCH_SIZE = 8
        BACKBONE = 'resnet18'
    
    INPUT_SIZE = 224
    NUM_EPOCHS = 5  # Reduced from 30
    
    print(f"Batch size: {BATCH_SIZE}, Backbone: {BACKBONE}")
    
    # Simplified transforms for faster processing
    train_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = FastDeepPixBisDataset('../train', train_transform, training=True)
    test_dataset = FastDeepPixBisDataset('../test', test_transform, training=False)
    
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("Error: No data found!")
        return
    
    # Data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset, BATCH_SIZE, shuffle=True, 
        num_workers=4, pin_memory=True, 
        persistent_workers=True, prefetch_factor=2
    )
    
    test_loader = DataLoader(
        test_dataset, BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    # Create model
    model = OptimizedDeepPixBis(num_classes=1, backbone=BACKBONE, pretrained=True)
    model = model.to(device)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Train
    start_time = time.time()
    results, best_acc = train_fast_deeppixbis(model, train_loader, test_loader, device, NUM_EPOCHS)
    total_time = time.time() - start_time
    
    # Final results
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best accuracy: {best_acc:.3f} ({best_acc*100:.1f}%)")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, 'fast_training_results.csv'), index=False)
    
    # Simple plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(results_df['epoch'], results_df['train_acc'], 'b-', label='Train')
    plt.plot(results_df['epoch'], results_df['test_acc'], 'r-', label='Test')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(results_df['epoch'], results_df['test_auc'], 'g-', label='AUC')
    plt.title('AUC Score')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'fast_training_plots.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return results, model

# --- Quick Inference Function ---
def predict_image(model_path, image_path, device):
    """Fast inference on single image"""
    model = OptimizedDeepPixBis()
    # model_path is expected to be the full path including the results_dir
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
        binary_pred, _ = model(image_tensor)
        prob = torch.sigmoid(binary_pred).item()
        prediction = "Spoof" if prob > 0.5 else "Live"
    
    return prediction, prob

if __name__ == "__main__":
    results, model = main()