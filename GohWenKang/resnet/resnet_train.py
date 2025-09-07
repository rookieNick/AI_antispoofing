import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import signal
import sys
import math

class SpatialAttention(nn.Module):
    """Spatial Attention Module - focuses on important spatial regions"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Generate spatial attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Average pooling across channels
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max pooling across channels
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(attention_input))
        return x * attention_map

class ChannelAttention(nn.Module):
    """Channel Attention Module - focuses on important feature channels"""
    def __init__(self, in_planes, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction_ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class CBAM(nn.Module):
    """Convolutional Block Attention Module - combines channel and spatial attention"""
    def __init__(self, in_planes, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class AttentionResidualBlock(nn.Module):
    """Enhanced ResNet block with attention mechanism"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, 
                 use_attention=True, reduction_ratio=16):
        super(AttentionResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.use_attention = use_attention
        
        if use_attention:
            self.attention = CBAM(out_channels, reduction_ratio)
            
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply attention if enabled
        if self.use_attention:
            out = self.attention(out)
        
        # Downsample residual if needed
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        
        return out

class ResNetAttention(nn.Module):
    """ResNet with Attention for Anti-Spoofing"""
    def __init__(self, num_classes=2, layers=[3, 4, 6, 3], base_channels=64, 
                 use_attention=True, dropout_rate=0.3):
        super(ResNetAttention, self).__init__()
        
        self.in_channels = base_channels
        self.use_attention = use_attention
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=7, stride=2, 
                              padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers with attention
        self.layer1 = self._make_layer(base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 8, layers[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final attention mechanism on global features
        if use_attention:
            self.global_attention = nn.Sequential(
                nn.Linear(base_channels * 8, base_channels * 8 // 4),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(base_channels * 8 // 4, base_channels * 8),
                nn.Sigmoid()
            )
        
        # Classification head with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(base_channels * 8, base_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),  # Less dropout in final layer
            nn.Linear(base_channels * 2, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(AttentionResidualBlock(self.in_channels, out_channels, 
                                           stride, downsample, self.use_attention))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(AttentionResidualBlock(out_channels, out_channels, 
                                               use_attention=self.use_attention))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights using appropriate methods"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual layers with attention
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Apply global attention to features
        if self.use_attention:
            attention_weights = self.global_attention(x)
            x = x * attention_weights
        
        # Classification
        x = self.classifier(x)
        
        return x

class CASIAFASDDataset(Dataset):
    """Same dataset class as in your ViT code"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        print(f"  Looking for data in: {data_dir}")
        
        # Check if data directory exists
        if not os.path.exists(data_dir):
            print(f"  ❌ Directory does not exist: {data_dir}")
            return
        
        # Load live images (label = 1)
        live_dir = os.path.join(data_dir, 'live')
        print(f"  Checking live directory: {live_dir}")
        if os.path.exists(live_dir):
            live_files = [f for f in os.listdir(live_dir) if f.lower().endswith('.png')]
            print(f"  Found {len(live_files)} PNG files in live directory")
            for img_name in live_files:
                self.images.append(os.path.join(live_dir, img_name))
                self.labels.append(1)  # live = 1
        else:
            print(f"  ❌ Live directory not found: {live_dir}")
        
        # Load spoof images (label = 0)
        spoof_dir = os.path.join(data_dir, 'spoof')
        print(f"  Checking spoof directory: {spoof_dir}")
        if os.path.exists(spoof_dir):
            spoof_files = [f for f in os.listdir(spoof_dir) if f.lower().endswith('.png')]
            print(f"  Found {len(spoof_files)} PNG files in spoof directory")
            for img_name in spoof_files:
                self.images.append(os.path.join(spoof_dir, img_name))
                self.labels.append(0)  # spoof = 0
        else:
            print(f"  ❌ Spoof directory not found: {spoof_dir}")
        
        print(f"  Total loaded: {len(self.images)} images from {data_dir}")
        print(f"  Live: {sum(self.labels)}, Spoof: {len(self.labels) - sum(self.labels)}")
        
        if len(self.images) == 0:
            print(f"  ⚠️ No images found in {data_dir}")
            print(f"  Expected structure: {data_dir}/live/*.png and {data_dir}/spoof/*.png")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.images)}")
            
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            if self.transform:
                return self.transform(Image.new('RGB', (224, 224), (0, 0, 0))), label
            return Image.new('RGB', (224, 224), (0, 0, 0)), label

class ResNetAttentionAntiSpoofing:
    def __init__(self, num_classes=2, device=None, use_attention=True, dropout_rate=0.3):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.checkpoint_saved = False
        
        # Performance optimizations for RTX 3050
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            print("✓ CUDA optimizations enabled")
        
        # Initialize ResNet+Attention model
        self.model = ResNetAttention(
            num_classes=num_classes, 
            layers=[3, 4, 6, 3],  # Same as original - standard ResNet50 structure
            base_channels=64, 
            use_attention=use_attention,
            dropout_rate=dropout_rate
        )
        
        self.model.to(self.device)
        
        # Mixed precision training for speed boost
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
            print("✓ Mixed precision training enabled")
        
        # Setup signal handler for graceful interruption
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Enhanced data transforms - optimized for anti-spoofing
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.25), ratio=(0.3, 3.3))
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ ResNet+Attention model initialized")
        print(f"✓ Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"✓ Attention mechanism: {'Enabled' if use_attention else 'Disabled'}")
    
    def load_data(self, train_dir, test_dir, batch_size=8, num_workers=2):
        # Create datasets
        train_dataset = CASIAFASDDataset(train_dir, transform=self.train_transform)
        test_dataset = CASIAFASDDataset(test_dir, transform=self.test_transform)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            drop_last=True
        )
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Testing samples: {len(test_dataset)}")
        
        return self.train_loader, self.test_loader
        
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully by saving checkpoint"""
        print("\n🛑 Training interrupted! Saving checkpoint...")
        self.save_checkpoint_now = True
    
    def save_checkpoint(self, epoch, model, optimizer, scheduler, history, save_dir, is_best=False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'best_val_acc': getattr(self, 'best_val_acc', 0.0),
            'model_config': {
                'num_classes': self.num_classes,
                'use_attention': True  # Always true for this model
            }
        }
        
        checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(save_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            print(f"✓ Best checkpoint saved at epoch {epoch+1}")
        
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path, optimizer=None, scheduler=None):
        """Load training checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return None
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"✓ Checkpoint loaded from epoch {checkpoint['epoch']+1}")
        return checkpoint
    
    def calculate_detailed_metrics(self, all_labels, all_predictions):
        """Calculate comprehensive metrics"""
        accuracy = accuracy_score(all_labels, all_predictions)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_predictions, average=None, labels=[0, 1]
        )
        
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='macro'
        )
        
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'spoof_precision': precision[0],
            'spoof_recall': recall[0],
            'spoof_f1': f1[0],
            'live_precision': precision[1],
            'live_recall': recall[1],
            'live_f1': f1[1],
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1
        }
        
        return metrics
    
    def calculate_mse_rmse(self, all_probabilities, all_labels):
        """Calculate MSE and RMSE for classification probabilities"""
        # Convert labels to one-hot encoding
        num_classes = len(np.unique(all_labels))
        one_hot_labels = np.zeros((len(all_labels), num_classes))
        for i, label in enumerate(all_labels):
            one_hot_labels[i, label] = 1.0
        
        # Calculate MSE between predicted probabilities and one-hot labels
        mse = mean_squared_error(one_hot_labels, all_probabilities)
        rmse = np.sqrt(mse)
        
        return mse, rmse
    
    def train(self, epochs=15, learning_rate=3e-4, weight_decay=5e-4, save_dir='GohWenKang/resnet/resnet_attention_models', 
              resume_from_checkpoint=None, label_smoothing=0.1):
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup optimizer with AdamW for better regularization
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing scheduler for better convergence
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=5,  # Restart every 5 epochs
            T_mult=2,  # Double the restart period
            eta_min=1e-7
        )
        
        # Focal loss to handle class imbalance better than standard cross entropy
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1, gamma=2, label_smoothing=0.1):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.label_smoothing = label_smoothing
                
            def forward(self, inputs, targets):
                # Apply label smoothing
                num_classes = inputs.size(1)
                targets_one_hot = F.one_hot(targets, num_classes).float()
                targets_smooth = targets_one_hot * (1 - self.label_smoothing) + \
                               self.label_smoothing / num_classes
                
                ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
                return focal_loss.mean()
        
        criterion = FocalLoss(alpha=1, gamma=2, label_smoothing=label_smoothing)
        mse_criterion = nn.MSELoss()
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'train_mse': [],
            'train_rmse': [],
            'val_loss': [],
            'val_acc': [],
            'val_mse': [],
            'val_rmse': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_macro_f1': [],
            'learning_rates': []
        }
        
        start_epoch = 0
        best_val_acc = 0.0
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            checkpoint = self.load_checkpoint(resume_from_checkpoint, optimizer, scheduler)
            if checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                history = checkpoint['history']
                best_val_acc = checkpoint.get('best_val_acc', 0.0)
                print(f"🔄 Resuming training from epoch {start_epoch+1}")
                print(f"📊 Best validation accuracy so far: {best_val_acc:.2f}%")
        
        self.best_val_acc = best_val_acc
        self.save_checkpoint_now = False
        
        print(f"🚀 Starting ResNet+Attention training with MSE/RMSE tracking on {self.device}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"🎯 Using Focal Loss with label smoothing: {label_smoothing}")
        print(f"⚡ Cosine annealing scheduler with warm restarts")
        print(f"📈 Tracking MSE/RMSE on probability predictions")
        
        for epoch in range(start_epoch, epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_probabilities = []
            train_labels_list = []
            
            train_pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for batch_idx, (images, labels) in enumerate(train_pbar):
                if self.save_checkpoint_now:
                    print("\n💾 Saving checkpoint due to interruption...")
                    self.save_checkpoint(epoch, self.model, optimizer, scheduler, history, save_dir)
                    print("✅ Checkpoint saved! You can resume training later.")
                    return history
                
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Mixed precision training
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                train_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Store probabilities and labels for MSE/RMSE calculation
                train_probabilities.extend(probabilities.detach().cpu().numpy())
                train_labels_list.extend(labels.cpu().numpy())
                
                current_acc = 100.*train_correct/train_total if train_total > 0 else 0
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
            
            # Calculate training MSE/RMSE
            train_mse, train_rmse = self.calculate_mse_rmse(train_probabilities, train_labels_list)
            
            # Update learning rate
            scheduler.step()
            
            # Validation phase
            val_loss, val_metrics, val_mse, val_rmse = self.evaluate_detailed_with_mse()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save history
            epoch_train_loss = train_loss / len(self.train_loader)
            epoch_train_acc = 100. * train_correct / train_total
            
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            history['train_mse'].append(train_mse)
            history['train_rmse'].append(train_rmse)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_metrics['accuracy'] * 100)
            history['val_mse'].append(val_mse)
            history['val_rmse'].append(val_rmse)
            history['val_precision'].append(val_metrics['weighted_precision'])
            history['val_recall'].append(val_metrics['weighted_recall'])
            history['val_f1'].append(val_metrics['weighted_f1'])
            history['val_macro_f1'].append(val_metrics['macro_f1'])
            history['learning_rates'].append(current_lr)
            
            val_acc = val_metrics['accuracy'] * 100
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
            print(f'  Train MSE: {train_mse:.4f}, Train RMSE: {train_rmse:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  Val MSE: {val_mse:.4f}, Val RMSE: {val_rmse:.4f}')
            print(f'  Val F1: {val_metrics["weighted_f1"]:.4f}, Macro F1: {val_metrics["macro_f1"]:.4f}')
            print(f'  Learning Rate: {current_lr:.6f}')
            
            # Save checkpoint every epoch
            self.save_checkpoint(epoch, self.model, optimizer, scheduler, history, save_dir)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.best_val_acc = best_val_acc
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'best_resnet_attention_model.pth'))
                self.save_checkpoint(epoch, self.model, optimizer, scheduler, history, save_dir, is_best=True)
                print(f'  🏆 New best model saved! Val Acc: {best_val_acc:.2f}%')
            
            print('-' * 80)
        
        # Save training history and final model
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'final_resnet_attention_model.pth'))
        
        # Plot training curves
        self.plot_comprehensive_training_curves(history, save_dir)
        
        print("🎉 Training completed successfully!")
        return history
    
    def evaluate_detailed_with_mse(self):
        """Detailed evaluation with comprehensive metrics including MSE/RMSE"""
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        val_loss /= len(self.test_loader)
        metrics = self.calculate_detailed_metrics(all_labels, all_predictions)
        val_mse, val_rmse = self.calculate_mse_rmse(all_probabilities, all_labels)
        
        return val_loss, metrics, val_mse, val_rmse
    
    def evaluate_detailed(self):
        """Detailed evaluation with comprehensive metrics (without MSE/RMSE for compatibility)"""
        val_loss, metrics, _, _ = self.evaluate_detailed_with_mse()
        return val_loss, metrics
    
    def test_and_confusion_matrix(self, model_path=None, save_dir='GohWenKang/resnet/resnet_attention_models/results'):
        """Generate comprehensive test results including MSE/RMSE"""
        os.makedirs(save_dir, exist_ok=True)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        print("Generating predictions for comprehensive analysis...")
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate comprehensive metrics
        metrics = self.calculate_detailed_metrics(all_labels, all_predictions)
        test_mse, test_rmse = self.calculate_mse_rmse(all_probabilities, all_labels)
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Add MSE/RMSE to metrics
        metrics['test_mse'] = test_mse
        metrics['test_rmse'] = test_rmse
        
        # Print detailed results
        print(f"\n{'='*60}")
        print(f"📊 RESNET+ATTENTION TEST RESULTS (WITH MSE/RMSE)")
        print(f"{'='*60}")
        print(f"🎯 Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"📈 Probability MSE: {test_mse:.4f}")
        print(f"📈 Probability RMSE: {test_rmse:.4f}")
        print(f"\n📈 Class-wise Metrics:")
        print(f"   Spoof (Class 0):")
        print(f"     Precision: {metrics['spoof_precision']:.4f}")
        print(f"     Recall: {metrics['spoof_recall']:.4f}")
        print(f"     F1-Score: {metrics['spoof_f1']:.4f}")
        print(f"   Live (Class 1):")
        print(f"     Precision: {metrics['live_precision']:.4f}")
        print(f"     Recall: {metrics['live_recall']:.4f}")
        print(f"     F1-Score: {metrics['live_f1']:.4f}")
        print(f"\n🌟 Average Metrics:")
        print(f"   Macro F1-Score: {metrics['macro_f1']:.4f}")
        print(f"   Weighted F1-Score: {metrics['weighted_f1']:.4f}")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm, metrics['accuracy'], save_dir)
        
        # Classification report
        class_names = ['Spoof', 'Live']
        report = classification_report(all_labels, all_predictions, target_names=class_names)
        print(f"\n📋 Detailed Classification Report:")
        print(report)
        
        # Save metrics to file
        with open(os.path.join(save_dir, 'test_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return cm, metrics
    
    def plot_confusion_matrix(self, cm, accuracy, save_dir, figsize=(12, 5)):
        """Plot confusion matrix"""
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Raw confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Spoof', 'Live'], 
                   yticklabels=['Spoof', 'Live'],
                   annot_kws={'size': 12}, ax=ax1)
        ax1.set_title('Raw Confusion Matrix', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        # Normalized confusion matrix
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                   xticklabels=['Spoof', 'Live'], 
                   yticklabels=['Spoof', 'Live'],
                   annot_kws={'size': 12}, ax=ax2)
        ax2.set_title('Normalized Confusion Matrix', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        
        fig.suptitle(f'ResNet+Attention Results - Accuracy: {accuracy*100:.2f}%', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Confusion matrix saved as {save_path}")
        return cm_normalized
    
    def plot_comprehensive_training_curves(self, history, save_dir):
        """Plot comprehensive training curves including MSE/RMSE"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Loss curves
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(history['train_loss'], label='Training Loss', marker='o', linewidth=2)
        ax1.plot(history['val_loss'], label='Validation Loss', marker='s', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Accuracy curves
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(history['train_acc'], label='Training Accuracy', marker='o', linewidth=2)
        ax2.plot(history['val_acc'], label='Validation Accuracy', marker='s', linewidth=2)
        ax2.set_title('Training and Validation Accuracy', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. MSE curves
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(history['train_mse'], label='Training MSE', marker='o', linewidth=2)
        ax3.plot(history['val_mse'], label='Validation MSE', marker='s', linewidth=2)
        ax3.set_title('Training and Validation MSE', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('MSE')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. RMSE curves
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(history['train_rmse'], label='Training RMSE', marker='o', linewidth=2)
        ax4.plot(history['val_rmse'], label='Validation RMSE', marker='s', linewidth=2)
        ax4.set_title('Training and Validation RMSE', fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('RMSE')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. F1 Scores
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(history['val_f1'], label='Weighted F1', marker='o', linewidth=2)
        ax5.plot(history['val_macro_f1'], label='Macro F1', marker='s', linewidth=2)
        ax5.set_title('F1 Score Evolution', fontweight='bold')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('F1 Score')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Precision and Recall
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(history['val_precision'], label='Precision', marker='o', linewidth=2)
        ax6.plot(history['val_recall'], label='Recall', marker='s', linewidth=2)
        ax6.set_title('Precision and Recall', fontweight='bold')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Score')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Learning Rate
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.plot(history['learning_rates'], marker='o', linewidth=2, color='red')
        ax7.set_title('Learning Rate Schedule', fontweight='bold')
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Learning Rate')
        ax7.set_yscale('log')
        ax7.grid(True, alpha=0.3)
        
        # 8. Overfitting Monitor
        ax8 = fig.add_subplot(gs[2, 1])
        acc_gap = [train - val for train, val in zip(history['train_acc'], history['val_acc'])]
        ax8.plot(acc_gap, marker='o', linewidth=2, color='orange')
        ax8.axhline(y=5, color='red', linestyle='--', alpha=0.7)
        ax8.set_title('Overfitting Monitor', fontweight='bold')
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Train-Val Accuracy Gap (%)')
        ax8.grid(True, alpha=0.3)
        
        # 9. MSE vs Accuracy correlation
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.scatter(history['val_mse'], history['val_acc'], alpha=0.7, color='purple')
        ax9.set_title('Validation MSE vs Accuracy', fontweight='bold')
        ax9.set_xlabel('Validation MSE')
        ax9.set_ylabel('Validation Accuracy (%)')
        ax9.grid(True, alpha=0.3)
        
        plt.suptitle('ResNet+Attention Training Dashboard (with MSE/RMSE)', fontsize=16, fontweight='bold')
        
        save_path = os.path.join(save_dir, 'training_curves_with_mse.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Training curves saved as {save_path}")

def main():
    """Main training function optimized for RTX 3050 with MSE/RMSE tracking"""
    config = {
        'train_dir': 'CASIA-FASD/train',  
        'test_dir': 'CASIA-FASD/test',    
        'batch_size': 8,                  # Optimal for RTX 3050 (4GB VRAM)
        'epochs': 20,
        'learning_rate': 3e-4,            # Good starting point for ResNet+Attention
        'weight_decay': 5e-4,             # Strong regularization
        'num_workers': 2,                 # Conservative for Ryzen 7 7435HS
        'save_dir': 'GohWenKang/resnet/resnet_attention_models',  # Different directory
        'resume_checkpoint': None,
        'use_attention': True,            # Enable attention mechanisms
        'dropout_rate': 0.3,              # Dropout for regularization
        'label_smoothing': 0.1            # Label smoothing for better generalization
    }
    
    print("🔧 Initializing ResNet+Attention Anti-Spoofing Model (WITH MSE/RMSE TRACKING)...")
    print("🎯 Optimized for RTX 3050 (4GB VRAM) + Ryzen 7 7435HS")
    print("🧠 Architecture Features:")
    print(f"   - Residual Learning with Skip Connections")
    print(f"   - CBAM Attention (Channel + Spatial)")
    print(f"   - Focal Loss for Class Imbalance")
    print(f"   - Cosine Annealing with Warm Restarts")
    print(f"   - Mixed Precision Training")
    print(f"   - MSE/RMSE on Probability Predictions")
    print(f"   - Dropout Rate: {config['dropout_rate']}")
    print(f"   - Label Smoothing: {config['label_smoothing']}")
    print(f"   - Save Directory: {config['save_dir']}")
    
    # Initialize model
    model = ResNetAttentionAntiSpoofing(
        num_classes=2,
        use_attention=config['use_attention'],
        dropout_rate=config['dropout_rate']
    )
    
    print(f"💻 Using device: {model.device}")
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"🎮 GPU: {gpu_props.name}")
        print(f"💾 GPU Memory: {gpu_props.total_memory / 1e9:.1f} GB")
        print(f"⚡ CUDA Compute Capability: {gpu_props.major}.{gpu_props.minor}")
    
    # Load data
    print(f"\n📂 Loading data...")
    try:
        train_loader, test_loader = model.load_data(
            config['train_dir'], 
            config['test_dir'], 
            config['batch_size'], 
            config['num_workers']
        )
        
        if train_loader is None or test_loader is None:
            print("❌ Failed to create data loaders")
            return None, None, None, None
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None
    
    # Check for existing checkpoint
    checkpoint_path = os.path.join(config['save_dir'], 'checkpoint.pth')
    resume_path = None
    
    if config['resume_checkpoint']:
        resume_path = config['resume_checkpoint']
        print(f"🔄 Will resume from specified checkpoint: {resume_path}")
    elif os.path.exists(checkpoint_path):
        print(f"\n🔍 Found existing checkpoint: {checkpoint_path}")
        try:
            resume_choice = input("Do you want to resume from checkpoint? (y/n): ").lower().strip()
            resume_path = checkpoint_path if resume_choice == 'y' else None
        except (KeyboardInterrupt, EOFError):
            print("\nStarting fresh training...")
            resume_path = None
    
    # Train model
    print(f"\n🚀 Starting ResNet+Attention training with MSE/RMSE tracking...")
    print("💡 Tip: Press Ctrl+C to save checkpoint and pause training")
    
    try:
        history = model.train(
            epochs=config['epochs'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            save_dir=config['save_dir'],
            resume_from_checkpoint=resume_path,
            label_smoothing=config['label_smoothing']
        )
        
        if history is None:
            print("❌ Training returned None")
            return None, None, None, None
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None
    
    # Test and generate results
    print(f"\n🧪 Testing best ResNet+Attention model...")
    try:
        results_dir = os.path.join(config['save_dir'], 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        best_model_path = os.path.join(config['save_dir'], 'best_resnet_attention_model.pth')
        if os.path.exists(best_model_path):
            cm, metrics = model.test_and_confusion_matrix(
                model_path=best_model_path, 
                save_dir=results_dir
            )
        else:
            print("⚠️ No best model found, testing current model...")
            cm, metrics = model.test_and_confusion_matrix(save_dir=results_dir)
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return model, history, None, None
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"🎉 RESNET+ATTENTION TRAINING SUMMARY (WITH MSE/RMSE)")
    print(f"{'='*80}")
    print(f"🏆 Best Test Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"🎯 Best Weighted F1-Score: {metrics['weighted_f1']:.4f}")
    print(f"📊 Best Macro F1-Score: {metrics['macro_f1']:.4f}")
    print(f"📈 Test MSE: {metrics['test_mse']:.4f}")
    print(f"📈 Test RMSE: {metrics['test_rmse']:.4f}")
    
    # MSE/RMSE interpretation
    print(f"\n🔍 MSE/RMSE Interpretation:")
    if metrics['test_mse'] < 0.1:
        print(f"   ✅ Very confident predictions (MSE < 0.1)")
    elif metrics['test_mse'] < 0.2:
        print(f"   ✓ Confident predictions (MSE < 0.2)")
    else:
        print(f"   ⚠️ Less confident predictions (MSE ≥ 0.2)")
    
    # Architecture advantages
    print(f"\n🧠 ResNet+Attention with MSE/RMSE Advantages:")
    print(f"   ✓ Skip connections prevent vanishing gradients")
    print(f"   ✓ Channel attention focuses on important features")
    print(f"   ✓ Spatial attention highlights critical regions")
    print(f"   ✓ Focal loss handles class imbalance")
    print(f"   ✓ MSE/RMSE tracks prediction confidence")
    print(f"   ✓ Memory efficient for RTX 3050")
    
    # Performance analysis
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    overfitting_gap = final_train_acc - final_val_acc
    
    print(f"\n🔍 Generalization Analysis:")
    print(f"   Final Training Accuracy: {final_train_acc:.2f}%")
    print(f"   Final Validation Accuracy: {final_val_acc:.2f}%")
    print(f"   Accuracy Gap: {overfitting_gap:.2f}%")
    print(f"   Final Train MSE: {history['train_mse'][-1]:.4f}")
    print(f"   Final Val MSE: {history['val_mse'][-1]:.4f}")
    
    if overfitting_gap > 10:
        print("   ⚠️  High overfitting detected")
    elif overfitting_gap > 5:
        print("   ⚠️  Moderate overfitting")
    else:
        print("   ✅ Good generalization!")
    
    print(f"📁 All results saved in: {results_dir}")
    print(f"📁 Model files saved in: {config['save_dir']}")
    print(f"{'='*80}")
    
    return model, history, cm, metrics

if __name__ == "__main__":
    print("🎯 ResNet+Attention Anti-Spoofing Training (WITH MSE/RMSE)")
    print("🛡️  Optimized for RTX 3050 + Ryzen 7 7435HS")
    print("📊 Now tracking MSE/RMSE on probability predictions!")
    print("=" * 60)
    
    # Run main training
    try:
        model, history, cm, metrics = main()
        if model is not None:
            print("✅ ResNet+Attention training with MSE/RMSE completed successfully!")
            print("📈 MSE/RMSE graphs will now be available in training curves!")
        else:
            print("❌ Training failed - check error messages above")
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        print("💾 Checkpoint should be saved automatically")
        print("🔄 Run the script again to resume training")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("💡 Try reducing batch_size if you got CUDA OOM error")