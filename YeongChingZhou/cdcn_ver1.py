import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from datetime import datetime
import time
from tqdm import tqdm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# --- GPU Setup with Enhanced Detection ---
def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ GPU Available: {gpu_name}")
        print(f"   Memory: {gpu_memory:.1f} GB")
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        return device
    else:
        print("❌ No GPU available, using CPU")
        return torch.device("cpu")

# --- Enhanced Central Difference Convolution with Adaptive Theta ---
class AdaptiveCDCConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, dilation=1, bias=True, theta=0.7):
        super(AdaptiveCDCConv2d, self).__init__()
        
        # Standard convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                             padding, dilation, bias=bias)
        
        # Learnable theta parameter for adaptive central difference
        self.theta = nn.Parameter(torch.tensor(theta, dtype=torch.float32))
        
        # Additional convolution for enhanced feature extraction
        self.edge_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                  stride=stride, padding=0, bias=False)
        
        # Attention mechanism for feature weighting
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Standard convolution
        out_normal = self.conv(x)
        
        # Central difference operation with learnable theta
        if abs(self.theta) > 1e-8:
            # Create central difference kernel dynamically
            kernel_diff = self.conv.weight.sum(dim=(2, 3), keepdim=True)
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=None, 
                               stride=self.conv.stride, padding=0)
            
            # Ensure spatial dimensions match
            if out_diff.size() != out_normal.size():
                out_diff = F.interpolate(out_diff, size=out_normal.shape[2:], 
                                       mode='bilinear', align_corners=False)
            
            # Apply central difference with learnable theta
            cdc_out = out_normal - self.theta * out_diff
        else:
            cdc_out = out_normal
        
        # Enhanced edge detection
        edge_out = self.edge_conv(x)
        
        # Combine features
        combined_out = cdc_out + 0.3 * edge_out
        
        # Apply attention mechanism
        attention_weights = self.attention(combined_out)
        final_out = combined_out * attention_weights
        
        return final_out

# --- Enhanced CDC Block with Multi-Scale Feature Fusion ---
class EnhancedCDCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, theta=0.7, use_se=True):
        super(EnhancedCDCBlock, self).__init__()
        
        # Multi-scale convolutions
        self.conv1 = AdaptiveCDCConv2d(in_channels, out_channels//2, 3, stride, 1, theta=theta)
        self.conv1_alt = nn.Conv2d(in_channels, out_channels//2, 1, stride, 0, bias=False)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = AdaptiveCDCConv2d(out_channels, out_channels, 3, 1, 1, theta=theta)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Squeeze-and-Excitation block
        self.use_se = use_se
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 16, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 16, out_channels, 1),
                nn.Sigmoid()
            )
        
        # Enhanced shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        # Multi-scale feature extraction
        out1 = self.conv1(x)
        out1_alt = self.conv1_alt(x)
        
        # Concatenate multi-scale features
        out = torch.cat([out1, out1_alt], dim=1)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply SE attention if enabled
        if self.use_se:
            se_weights = self.se(out)
            out = out * se_weights
        
        out += residual
        out = self.relu(out)
        
        return out

# --- Advanced CDCN Architecture ---
class AdvancedCDCN(nn.Module):
    def __init__(self, num_classes=2, theta=0.7, map_size=32, dropout_rate=0.5):
        super(AdvancedCDCN, self).__init__()
        self.map_size = map_size
        
        # Enhanced stem network
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        # Progressive CDC layers with different theta values
        self.layer1 = self._make_layer(64, 64, 2, 1, theta * 1.0)
        self.layer2 = self._make_layer(64, 128, 2, 2, theta * 0.8)
        self.layer3 = self._make_layer(128, 256, 3, 2, theta * 0.6)  # More blocks
        self.layer4 = self._make_layer(256, 512, 2, 2, theta * 0.4)
        
        # Multi-scale feature aggregation
        self.feature_aggregation = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # Fixed size for consistency
        )
        
        # Enhanced classification head with multiple pathways
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512 * 2, 256),  # *2 for concat of avg and max pool
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        # Enhanced depth prediction network
        self.depth_predictor = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, blocks, stride, theta):
        layers = []
        layers.append(EnhancedCDCBlock(in_channels, out_channels, stride, theta))
        for _ in range(1, blocks):
            layers.append(EnhancedCDCBlock(out_channels, out_channels, 1, theta))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature extraction
        x = self.stem(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature_map = self.layer4(x)
        
        # Feature aggregation for depth prediction
        aggregated_features = self.feature_aggregation(feature_map)
        
        # Classification pathway
        avg_pool_features = self.global_pool(feature_map)
        max_pool_features = self.max_pool(feature_map)
        
        # Combine average and max pooling
        combined_features = torch.cat([avg_pool_features, max_pool_features], dim=1)
        combined_features = torch.flatten(combined_features, 1)
        
        cls_output = self.classifier(combined_features)
        
        # Depth prediction pathway
        depth_map = self.depth_predictor(aggregated_features)
        depth_map = F.interpolate(depth_map, size=(self.map_size, self.map_size), 
                                 mode='bilinear', align_corners=False)
        
        return cls_output, depth_map

# --- Smart Dataset with Enhanced Augmentation ---
class SmartAntiSpoofDataset(Dataset):
    def __init__(self, root_dir, transform=None, map_size=32, is_training=True):
        self.root_dir = root_dir
        self.transform = transform
        self.map_size = map_size
        self.is_training = is_training
        self.samples = []
        
        print(f"🔍 Loading dataset from: {os.path.abspath(root_dir)}")
        
        # Support multiple folder structures
        possible_structures = [
            ('live', 'spoof'),
            ('real', 'fake'),
            ('1', '0'),
            ('genuine', 'attack')
        ]
        
        live_count = spoof_count = 0
        
        for live_name, spoof_name in possible_structures:
            live_dir = os.path.join(root_dir, live_name)
            spoof_dir = os.path.join(root_dir, spoof_name)
            
            if os.path.exists(live_dir):
                live_files = [f for f in os.listdir(live_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                for filename in live_files:
                    self.samples.append((os.path.join(live_dir, filename), 0))
                live_count += len(live_files)
                print(f"   📁 {live_name}: {len(live_files)} images")
            
            if os.path.exists(spoof_dir):
                spoof_files = [f for f in os.listdir(spoof_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                for filename in spoof_files:
                    self.samples.append((os.path.join(spoof_dir, filename), 1))
                spoof_count += len(spoof_files)
                print(f"   📁 {spoof_name}: {len(spoof_files)} images")
            
            if live_count > 0 or spoof_count > 0:
                break  # Found valid structure
        
        if len(self.samples) == 0:
            print(f"❌ No images found in {root_dir}")
            print("💡 Supported folder structures:")
            for live_name, spoof_name in possible_structures:
                print(f"   - {live_name}/ and {spoof_name}/")
        else:
            print(f"✅ Total: {len(self.samples)} images loaded")
            print(f"   Balance: Live={live_count}, Spoof={spoof_count}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        # Enhanced depth map generation
        depth_map = self._generate_depth_map(label)
        
        return image, label, depth_map
    
    def _generate_depth_map(self, label):
        if label == 0:  # Live face - complex depth variation
            # Create base depth with face-like structure
            depth_map = torch.ones(1, self.map_size, self.map_size) * 0.5
            
            # Add facial structure simulation
            center_y, center_x = self.map_size // 2, self.map_size // 2
            y_coords, x_coords = torch.meshgrid(torch.arange(self.map_size, dtype=torch.float32), 
                                               torch.arange(self.map_size, dtype=torch.float32), 
                                               indexing='ij')
            
            # Nose bridge (higher depth)
            nose_mask = ((x_coords - center_x).abs() < self.map_size // 8) & \
                       ((y_coords - center_y).abs() < self.map_size // 4)
            depth_map[0][nose_mask] += 0.2
            
            # Eyes (lower depth) - Fixed the pow() issue
            eye_radius = self.map_size // 6
            eye_left = ((x_coords - center_x + self.map_size // 4) ** 2 + 
                       (y_coords - center_y + self.map_size // 6) ** 2) < (eye_radius ** 2)
            eye_right = ((x_coords - center_x - self.map_size // 4) ** 2 + 
                        (y_coords - center_y + self.map_size // 6) ** 2) < (eye_radius ** 2)
            depth_map[0][eye_left | eye_right] -= 0.15
            
            # Add realistic noise
            noise = torch.randn(1, self.map_size, self.map_size) * 0.05
            depth_map += noise
            
        else:  # Spoof face - flat with minimal variation
            depth_map = torch.zeros(1, self.map_size, self.map_size)
            # Add very minimal texture-like variation
            depth_map += torch.randn(1, self.map_size, self.map_size) * 0.02
        
        return torch.clamp(depth_map, 0, 1)

# --- Advanced Multi-task Loss with Dynamic Weighting ---
class AdvancedCDCNLoss(nn.Module):
    def __init__(self, cls_weight=1.0, depth_weight=0.5, focal_alpha=0.25, focal_gamma=2.0):
        super(AdvancedCDCNLoss, self).__init__()
        self.cls_weight = cls_weight
        self.depth_weight = depth_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        self.cls_criterion = nn.CrossEntropyLoss(reduction='none')
        self.depth_criterion = nn.MSELoss()
        
        # Dynamic weighting based on training progress
        self.register_buffer('step_count', torch.tensor(0))
        
    def focal_loss(self, inputs, targets):
        ce_loss = self.cls_criterion(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def forward(self, cls_output, depth_output, cls_target, depth_target):
        # Use focal loss for classification (handles class imbalance better)
        cls_loss = self.focal_loss(cls_output, cls_target)
        
        # MSE loss for depth prediction
        depth_loss = self.depth_criterion(depth_output, depth_target)
        
        # Dynamic weighting - emphasize depth more in later stages
        self.step_count += 1
        dynamic_depth_weight = self.depth_weight * (1 + 0.1 * torch.sigmoid(self.step_count / 1000 - 5))
        
        total_loss = self.cls_weight * cls_loss + dynamic_depth_weight * depth_loss
        
        return total_loss, cls_loss, depth_loss

# --- Enhanced Training Function ---
def train_advanced_cdcn(model, train_loader, test_loader, device, num_epochs=25):
    criterion = AdvancedCDCNLoss(cls_weight=1.0, depth_weight=0.6)
    
    # Advanced optimizer with different learning rates for different parts
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 0.001},
        {'params': classifier_params, 'lr': 0.002}  # Higher LR for classifier
    ], weight_decay=2e-4)
    
    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    # Mixed precision setup (fixed for compatibility)
    use_amp = device.type == 'cuda'
    if use_amp:
        try:
            scaler = torch.amp.GradScaler('cuda')
        except:
            # Fallback for older PyTorch versions
            scaler = torch.cuda.amp.GradScaler()
            use_amp = True
    else:
        scaler = None
        use_amp = False
    
    print(f"\n🚀 Starting Advanced CDCN Training for {num_epochs} epochs...")
    print(f"   📊 Batch size: {train_loader.batch_size}")
    print(f"   🔧 Mixed precision: {'Enabled' if use_amp else 'Disabled'}")
    print(f"   🎯 Advanced features: Focal loss, Dynamic weighting, SE blocks")
    print("=" * 60)
    
    results = []
    best_acc = 0.0
    best_f1 = 0.0
    patience_counter = 0
    patience_limit = 8
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # === TRAINING ===
        model.train()
        train_metrics = {'total_loss': 0, 'cls_loss': 0, 'depth_loss': 0, 'correct': 0, 'total': 0}
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1:2d}/{num_epochs} [TRAIN]", 
                         leave=False, ncols=100)
        
        for batch_idx, (images, labels, depth_maps) in enumerate(train_loop):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            depth_maps = depth_maps.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if use_amp and scaler:
                try:
                    with torch.amp.autocast('cuda'):
                        cls_output, depth_output = model(images)
                        total_loss, cls_loss, depth_loss = criterion(cls_output, depth_output, labels, depth_maps)
                except:
                    # Fallback for older PyTorch versions
                    with torch.cuda.amp.autocast():
                        cls_output, depth_output = model(images)
                        total_loss, cls_loss, depth_loss = criterion(cls_output, depth_output, labels, depth_maps)
                
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                cls_output, depth_output = model(images)
                total_loss, cls_loss, depth_loss = criterion(cls_output, depth_output, labels, depth_maps)
                total_loss.backward()
                optimizer.step()
            
            # Update metrics
            train_metrics['total_loss'] += total_loss.item()
            train_metrics['cls_loss'] += cls_loss.item()
            train_metrics['depth_loss'] += depth_loss.item()
            
            _, predicted = torch.max(cls_output, 1)
            train_metrics['total'] += labels.size(0)
            train_metrics['correct'] += (predicted == labels).sum().item()
            
            # Update progress bar
            current_acc = train_metrics['correct'] / train_metrics['total']
            train_loop.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Acc': f"{current_acc:.3f}",
                'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        scheduler.step()
        
        # Calculate training metrics
        train_acc = train_metrics['correct'] / train_metrics['total']
        avg_train_loss = train_metrics['total_loss'] / len(train_loader)
        
        # === EVALUATION ===
        model.eval()
        test_metrics = {'total_loss': 0, 'cls_loss': 0, 'depth_loss': 0}
        all_preds, all_targets = [], []
        
        test_loop = tqdm(test_loader, desc=f"Epoch {epoch+1:2d}/{num_epochs} [TEST] ", 
                        leave=False, ncols=100)
        
        with torch.no_grad():
            for images, labels, depth_maps in test_loop:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                depth_maps = depth_maps.to(device, non_blocking=True)
                
                if use_amp:
                    try:
                        with torch.amp.autocast('cuda'):
                            cls_output, depth_output = model(images)
                            total_loss, cls_loss, depth_loss = criterion(cls_output, depth_output, labels, depth_maps)
                    except:
                        with torch.cuda.amp.autocast():
                            cls_output, depth_output = model(images)
                            total_loss, cls_loss, depth_loss = criterion(cls_output, depth_output, labels, depth_maps)
                else:
                    cls_output, depth_output = model(images)
                    total_loss, cls_loss, depth_loss = criterion(cls_output, depth_output, labels, depth_maps)
                
                test_metrics['total_loss'] += total_loss.item()
                test_metrics['cls_loss'] += cls_loss.item()
                test_metrics['depth_loss'] += depth_loss.item()
                
                _, predicted = torch.max(cls_output, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
                current_acc = accuracy_score(all_targets, all_preds)
                test_loop.set_postfix({
                    'Loss': f"{total_loss.item():.4f}",
                    'Acc': f"{current_acc:.3f}"
                })
        
        # Calculate test metrics
        test_acc = accuracy_score(all_targets, all_preds)
        test_f1 = f1_score(all_targets, all_preds, average='weighted')
        test_precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        test_recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        avg_test_loss = test_metrics['total_loss'] / len(test_loader)
        
        # Model checkpointing with multiple criteria
        improved = False
        if test_acc > best_acc or (test_acc == best_acc and test_f1 > best_f1):
            best_acc = test_acc
            best_f1 = test_f1
            improved = True
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': test_acc,
                'f1_score': test_f1,
                'test_loss': avg_test_loss
            }, 'advanced_cdcn_best.pth')
        else:
            patience_counter += 1
        
        # Timing calculations
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        eta = (elapsed / (epoch + 1)) * (num_epochs - epoch - 1)
        
        # Enhanced logging
        print(f"\n📊 Epoch {epoch+1:2d}/{num_epochs} Summary:")
        print(f"   ⏱️  Time: {epoch_time:.1f}s | ETA: {eta/60:.1f}min | Total: {elapsed/60:.1f}min")
        print(f"   🚂 Train: Loss={avg_train_loss:.4f} | Acc={train_acc:.4f} ({train_acc*100:.1f}%)")
        print(f"   🎯 Test:  Loss={avg_test_loss:.4f} | Acc={test_acc:.4f} ({test_acc*100:.1f}%) {'🌟' if improved else ''}")
        print(f"   📈 Metrics: P={test_precision:.3f} | R={test_recall:.3f} | F1={test_f1:.3f}")
        print(f"   🔄 LR: {optimizer.param_groups[0]['lr']:.2e} | Patience: {patience_counter}/{patience_limit}")
        
        # Store results
        results.append({
            'epoch': epoch + 1,
            'train_total_loss': avg_train_loss,
            'train_cls_loss': train_metrics['cls_loss'] / len(train_loader),
            'train_depth_loss': train_metrics['depth_loss'] / len(train_loader),
            'train_accuracy': train_acc,
            'test_total_loss': avg_test_loss,
            'test_cls_loss': test_metrics['cls_loss'] / len(test_loader),
            'test_depth_loss': test_metrics['depth_loss'] / len(test_loader),
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time
        })
        
        # Early stopping
        if patience_counter >= patience_limit:
            print(f"\n⚠️ Early stopping triggered after {patience_limit} epochs without improvement")
            break
        
        # Memory cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    return results, total_time, best_acc, best_f1

# --- Enhanced Visualization Function ---
def create_enhanced_plots(results, save_dir):
    """Create comprehensive training plots"""
    results_df = pd.DataFrame(results)
    
    # Create a more comprehensive plot layout
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Total Loss curves
    plt.subplot(2, 4, 1)
    plt.plot(results_df['epoch'], results_df['train_total_loss'], 'b-', label='Train', linewidth=2)
    plt.plot(results_df['epoch'], results_df['test_total_loss'], 'r-', label='Test', linewidth=2)
    plt.title('Total Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Classification Loss
    plt.subplot(2, 4, 2)
    plt.plot(results_df['epoch'], results_df['train_cls_loss'], 'b-', label='Train', linewidth=2)
    plt.plot(results_df['epoch'], results_df['test_cls_loss'], 'r-', label='Test', linewidth=2)
    plt.title('Classification Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Depth Loss
    plt.subplot(2, 4, 3)
    plt.plot(results_df['epoch'], results_df['train_depth_loss'], 'b-', label='Train', linewidth=2)
    plt.plot(results_df['epoch'], results_df['test_depth_loss'], 'r-', label='Test', linewidth=2)
    plt.title('Depth Prediction Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Accuracy
    plt.subplot(2, 4, 4)
    plt.plot(results_df['epoch'], results_df['train_accuracy'], 'b-', label='Train', linewidth=2)
    plt.plot(results_df['epoch'], results_df['test_accuracy'], 'r-', label='Test', linewidth=2)
    plt.title('Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # 5. F1 Score
    plt.subplot(2, 4, 5)
    plt.plot(results_df['epoch'], results_df['test_f1'], 'g-', label='F1 Score', linewidth=2)
    plt.title('F1 Score', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # 6. Precision & Recall
    plt.subplot(2, 4, 6)
    plt.plot(results_df['epoch'], results_df['test_precision'], 'orange', label='Precision', linewidth=2)
    plt.plot(results_df['epoch'], results_df['test_recall'], 'purple', label='Recall', linewidth=2)
    plt.title('Precision & Recall', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # 7. Learning Rate
    plt.subplot(2, 4, 7)
    plt.plot(results_df['epoch'], results_df['learning_rate'], 'brown', linewidth=2)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # 8. Training Time per Epoch
    plt.subplot(2, 4, 8)
    plt.plot(results_df['epoch'], results_df['epoch_time'], 'cyan', linewidth=2)
    plt.title('Training Time per Epoch', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/comprehensive_training_plots.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create confusion matrix for final epoch
    return results_df

# --- Main Execution with Comprehensive Setup ---
def main():
    print("🚀 Advanced CDCN Anti-Spoofing System")
    print("=" * 50)
    
    # Device setup
    device = setup_device()
    
    # Optimized configuration for various hardware
    if device.type == 'cuda':
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory < 8:  # Less than 8GB VRAM
            INPUT_SIZE = (224, 224)
            BATCH_SIZE = 128
            NUM_WORKERS = 6
        else:  # 8GB+ VRAM
            INPUT_SIZE = (256, 256)
            BATCH_SIZE = 64
            NUM_WORKERS = 4
    else:  # CPU
        INPUT_SIZE = (224, 224)
        BATCH_SIZE = 16
        NUM_WORKERS = 2
    
    MAP_SIZE = 32
    NUM_EPOCHS = 10
    DROPOUT_RATE = 0.5
    
    print(f"\n⚙️ Configuration:")
    print(f"   Input Size: {INPUT_SIZE}")
    print(f"   Depth Map Size: {MAP_SIZE}×{MAP_SIZE}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Device: {device}")
    print(f"   Workers: {NUM_WORKERS}")
    
    # Enhanced data transforms
    train_transform = transforms.Compose([
        transforms.Resize((int(INPUT_SIZE[0] * 1.1), int(INPUT_SIZE[1] * 1.1))),
        transforms.RandomCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.1),
        transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33))
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_dataset = SmartAntiSpoofDataset('train', transform=train_transform, map_size=MAP_SIZE, is_training=True)
    test_dataset = SmartAntiSpoofDataset('test', transform=test_transform, map_size=MAP_SIZE, is_training=False)
    
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("❌ Dataset loading failed!")
        print("💡 Please ensure your data is organized as:")
        print("   📁 train/live/ (or train/real/, train/1/, train/genuine/)")
        print("   📁 train/spoof/ (or train/fake/, train/0/, train/attack/)")
        print("   📁 test/live/ (or test/real/, test/1/, test/genuine/)")
        print("   📁 test/spoof/ (or test/fake/, test/0/, test/attack/)")
        return
    
    # Create optimized data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=device.type=='cuda',
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=device.type=='cuda',
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    
    print(f"   📊 Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")
    
    # Create advanced model (disable compilation to avoid Triton issues)
    model = AdvancedCDCN(num_classes=2, theta=0.7, map_size=MAP_SIZE, dropout_rate=DROPOUT_RATE)
    model = model.to(device)
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assume float32
    
    print(f"\n🧠 Advanced CDCN Model:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {model_size_mb:.1f} MB")
    
    # Train the model
    print("\n" + "="*60)
    results, total_time, best_acc, best_f1 = train_advanced_cdcn(
        model, train_loader, test_loader, device, NUM_EPOCHS
    )
    
    # Final results
    print(f"\n🎉 Advanced CDCN Training Complete!")
    print("=" * 50)
    print(f"⏱️  Total Training Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"📈 Best Test Accuracy: {best_acc:.4f} ({best_acc*100:.1f}%)")
    print(f"📈 Best F1 Score: {best_f1:.4f}")
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"advanced_cdcn_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{results_dir}/detailed_training_results.csv", index=False)
    
    # Create comprehensive plots
    print(f"\n📊 Creating visualization plots...")
    results_df = create_enhanced_plots(results, results_dir)
    
    # Final model evaluation with detailed metrics
    print(f"\n📊 Final Comprehensive Model Evaluation:")
    print("-" * 40)
    
    # Load best model for evaluation
    if os.path.exists('advanced_cdcn_best.pth'):
        checkpoint = torch.load('advanced_cdcn_best.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded best model from epoch {checkpoint['epoch'] + 1}")
    
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc="Final Evaluation"):
            images, labels = images.to(device), labels.to(device)
            cls_output, _ = model(images)
            
            probs = F.softmax(cls_output, dim=1)
            _, predicted = torch.max(cls_output, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])  # Probability of spoof class
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    # Per-class metrics
    live_precision = precision_score(all_targets, all_preds, pos_label=0, zero_division=0)
    live_recall = recall_score(all_targets, all_preds, pos_label=0, zero_division=0)
    spoof_precision = precision_score(all_targets, all_preds, pos_label=1, zero_division=0)
    spoof_recall = recall_score(all_targets, all_preds, pos_label=1, zero_division=0)
    
    print(f"   Overall Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"   Weighted Precision: {precision:.4f}")
    print(f"   Weighted Recall:    {recall:.4f}")
    print(f"   Weighted F1 Score:  {f1:.4f}")
    print(f"\n   Live  Class: P={live_precision:.3f}, R={live_recall:.3f}")
    print(f"   Spoof Class: P={spoof_precision:.3f}, R={spoof_recall:.3f}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Live', 'Spoof'], 
                yticklabels=['Live', 'Spoof'])
    plt.title('Confusion Matrix - Advanced CDCN', fontsize=16, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save final evaluation results
    eval_results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'live_precision': live_precision,
        'live_recall': live_recall,
        'spoof_precision': spoof_precision,
        'spoof_recall': spoof_recall,
        'confusion_matrix': cm.tolist(),
        'total_training_time': total_time,
        'best_epoch': checkpoint.get('epoch', -1) + 1 if os.path.exists('advanced_cdcn_best.pth') else -1
    }
    
    import json
    with open(f"{results_dir}/final_evaluation.json", 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\n📁 All results saved to: {results_dir}/")
    print(f"💾 Best model saved as: advanced_cdcn_best.pth")
    print(f"✅ Advanced CDCN training and evaluation completed successfully!")
    
    return results, model, results_dir

if __name__ == "__main__":
    try:
        results, model, results_dir = main()
        print(f"\n🎯 Training completed successfully!")
        print(f"📁 Check results in: {results_dir}/")
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()