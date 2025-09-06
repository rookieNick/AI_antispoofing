"""
🎯 CDCN (Central Difference Convolutional Network) FOR FACE ANTI-SPOOFING
================================================================================
RTX 4050 Optimized CDCN implementation for face anti-spoofing
Batch Size: 64, Epochs: 20, Optimized for Speed and Accuracy
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
from PIL import Image 
import numpy as np
import pandas as pd
from datetime import datetime
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Import comprehensive test analysis module
import sys
sys.path.append('..')
from comprehensive_test_analysis import ComprehensiveTestAnalyzer, create_test_metrics_dict

# --- GPU Setup ---
def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ GPU Available: {gpu_name}")
        print(f"   Memory: {gpu_memory:.1f} GB")
        torch.backends.cudnn.benchmark = True
        return device
    else:
        print("❌ No GPU available, using CPU")
        return torch.device("cpu")

# --- RTX 4050 Optimized Central Difference Convolution ---
class OptimizedCDCConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, dilation=1, bias=True, theta=0.7):
        super(OptimizedCDCConv2d, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                             padding, dilation, bias=bias)
        
        # Fixed theta for speed (no learnable parameter)
        self.theta = theta
        
        # Simplified edge convolution
        self.edge_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                  stride=stride, padding=0, bias=False)
        
        # Lightweight attention
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, max(1, out_channels // 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, out_channels // 8), out_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Standard convolution
        conv_out = self.conv(x)
        
        # Simplified central difference (vertical only for speed)
        diff_v = F.conv2d(x, self._get_diff_kernel_v(x), padding=1, groups=1)
        
        # Edge enhancement
        edge_out = self.edge_conv(torch.abs(diff_v))
        
        # Fast combination
        combined = self.theta * conv_out + (1 - self.theta) * edge_out
        
        # Apply lightweight attention
        attention_weights = self.attention(combined)
        output = combined * attention_weights
        
        return output
    
    def _get_diff_kernel_v(self, x):
        if not hasattr(self, '_cached_kernel_v') or self._cached_kernel_v.device != x.device:
            kernel = torch.zeros(x.size(1), x.size(1), 3, 3, device=x.device)
            for i in range(x.size(1)):
                kernel[i, i, 0, 1] = 1.0
                kernel[i, i, 2, 1] = -1.0
            self._cached_kernel_v = kernel
        return self._cached_kernel_v

# --- RTX 4050 Optimized CDCN Model ---
class CDCN_RTX4050(nn.Module):
    def __init__(self, num_classes=2, input_channels=3):
        super(CDCN_RTX4050, self).__init__()
        
        # Optimized CDC blocks for RTX 4050 (batch size 64)
        self.cdc1 = OptimizedCDCConv2d(input_channels, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.1)
        
        self.cdc2 = OptimizedCDCConv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.15)
        
        self.cdc3 = OptimizedCDCConv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.2)
        
        self.cdc4 = OptimizedCDCConv2d(128, 256, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout2d(0.25)
        
        # Optimized classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # CDC feature extraction
        x = F.relu(self.bn1(self.cdc1(x)), inplace=True)
        x = self.pool1(self.dropout1(x))
        
        x = F.relu(self.bn2(self.cdc2(x)), inplace=True)
        x = self.pool2(self.dropout2(x))
        
        x = F.relu(self.bn3(self.cdc3(x)), inplace=True)
        x = self.pool3(self.dropout3(x))
        
        x = F.relu(self.bn4(self.cdc4(x)), inplace=True)
        x = self.pool4(self.dropout4(x))
        
        # Classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

# --- Dataset Class ---
class FaceAntiSpoofingDataset(Dataset):
    def __init__(self, data_dir, transform=None, limit_samples=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        live_dir = os.path.join(data_dir, 'live')
        spoof_dir = os.path.join(data_dir, 'spoof')
        
        live_count = 0
        spoof_count = 0
        
        # Live samples (label = 0)
        if os.path.exists(live_dir):
            for filename in os.listdir(live_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    if limit_samples and live_count >= limit_samples:
                        break
                    self.samples.append((os.path.join(live_dir, filename), 0))
                    live_count += 1
        
        # Spoof samples (label = 1)
        if os.path.exists(spoof_dir):
            for filename in os.listdir(spoof_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    if limit_samples and spoof_count >= limit_samples:
                        break
                    self.samples.append((os.path.join(spoof_dir, filename), 1))
                    spoof_count += 1
        
        print(f"Found structure: live({live_count}) / spoof({spoof_count})")
        print(f"Total {os.path.basename(data_dir)} samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- RTX 4050 Optimized Training Functions ---
def train_epoch_rtx4050(model, train_loader, optimizer, criterion, device, epoch, scheduler=None):
    """RTX 4050 optimized training with micro-batching for batch size 64"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Micro-batching setup for RTX 4050
    micro_batch_size = 16  # Actual batch size that fits in 6GB
    accumulation_steps = 4  # 16 * 4 = 64 effective batch size
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        # Split batch into micro-batches
        micro_batches = torch.split(images, micro_batch_size)
        micro_labels = torch.split(labels, micro_batch_size)
        
        epoch_loss = 0
        optimizer.zero_grad()
        
        # Process micro-batches
        for micro_img, micro_lbl in zip(micro_batches, micro_labels):
            micro_img = micro_img.to(device, non_blocking=True)
            micro_lbl = micro_lbl.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(micro_img)
            loss = criterion(outputs, micro_lbl) / accumulation_steps
            loss.backward()
            
            epoch_loss += loss.item() * accumulation_steps
            
            # Statistics
            _, predicted = torch.max(outputs, 1)
            total += micro_lbl.size(0)
            correct += predicted.eq(micro_lbl).sum().item()
            
            # Cleanup micro-batch
            del micro_img, micro_lbl, outputs, predicted
        
        # Gradient clipping and optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Step scheduler if provided
        if scheduler is not None:
            scheduler.step()
        
        total_loss += epoch_loss
        
        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'Loss': f'{epoch_loss:.3f}',
            'Acc': f'{100.*correct/total:.1f}%',
            'LR': f'{current_lr:.2e}',
            'GPU': f'{torch.cuda.memory_allocated()/1e9:.1f}GB'
        })
        
        # Memory cleanup every 10 batches
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    return {
        'train_acc': correct / total,
        'train_loss': total_loss / len(train_loader)
    }

def evaluate_epoch_rtx4050(model, test_loader, criterion, device):
    """RTX 4050 optimized evaluation"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy()[:, 1])
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Cleanup
            del images, labels, outputs, probabilities, predicted, loss
    
    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)
    
    try:
        auc_score = roc_auc_score(all_targets, all_probabilities)
    except:
        auc_score = 0.0
    
    return {
        'test_acc': accuracy,
        'test_loss': avg_loss,
        'auc': auc_score,
        'targets': all_targets,
        'predictions': all_predictions,
        'probabilities': all_probabilities
    }

# --- Training Visualization Function ---
def create_training_visualization(results, save_dir, model_name="CDCN_RTX4050"):
    """Create comprehensive training metrics visualization"""
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Extract metrics from results
    epochs = [r['epoch'] for r in results]
    train_loss = [r['train_loss'] for r in results]
    test_loss = [r['test_loss'] for r in results]
    train_acc = [r['train_acc'] * 100 for r in results]
    test_acc = [r['test_acc'] * 100 for r in results]
    
    # Create the main figure with 6 subplots (2x3 grid)
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Color scheme
    train_color = '#FF6B6B'
    val_color = '#4ECDC4'
    
    # 1. Training and Validation Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, train_loss, color=train_color, linewidth=2.5, label='Training Loss', marker='o', markersize=4)
    ax1.plot(epochs, test_loss, color=val_color, linewidth=2.5, label='Validation Loss', marker='s', markersize=4)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#FAFAFA')
    
    # 2. Training and Validation Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, train_acc, color=train_color, linewidth=2.5, label='Training Accuracy', marker='o', markersize=4)
    ax2.plot(epochs, test_acc, color=val_color, linewidth=2.5, label='Validation Accuracy', marker='s', markersize=4)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#FAFAFA')
    ax2.set_ylim([0, 100])
    
    # 3. Learning Rate Schedule
    ax3 = fig.add_subplot(gs[0, 2])
    if 'lr' in results[0]:
        lrs = [r['lr'] for r in results]
        ax3.plot(epochs, lrs, color='#45B7D1', linewidth=2.5, marker='d', markersize=4)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold', pad=20)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#FAFAFA')
    
    # 4. Performance Metrics
    ax4 = fig.add_subplot(gs[1, 0])
    if len(epochs) > 0:
        best_acc = max(test_acc)
        best_epoch = test_acc.index(best_acc) + 1
        ax4.bar(['Best Acc', 'Final Acc'], [best_acc, test_acc[-1]], color=['#F39C12', '#E74C3C'])
        ax4.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
        ax4.set_ylabel('Accuracy (%)', fontsize=12)
        ax4.set_ylim([0, 100])
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor('#FAFAFA')
    
    # 5. Training Time Analysis
    ax5 = fig.add_subplot(gs[1, 1])
    if 'time' in results[0]:
        times = [r['time'] for r in results]
        ax5.plot(epochs, times, color='#E74C3C', linewidth=2.5, marker='*', markersize=6)
        ax5.set_title('Training Time per Epoch', fontsize=14, fontweight='bold', pad=20)
        ax5.set_xlabel('Epoch', fontsize=12)
        ax5.set_ylabel('Time (seconds)', fontsize=12)
    ax5.grid(True, alpha=0.3)
    ax5.set_facecolor('#FAFAFA')
    
    # 6. Model Info Panel
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    model_info = f"""RTX 4050 Optimized CDCN

Architecture:
• CDC Blocks: 32→64→128→256
• Micro-batching: 16×4=64
• Global Avg Pool
• FC: 256→128→64→2

Optimization:
• Batch Size: 64 (effective)
• Epochs: 20
• Optimizer: AdamW
• Scheduler: OneCycleLR
• GPU: RTX 4050 (6GB)"""
    
    ax6.text(0.1, 0.9, model_info, fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='#F8F9FA', alpha=0.8),
             transform=ax6.transAxes)
    ax6.set_title('Model Configuration', fontsize=14, fontweight='bold', pad=20)
    
    # Add main title
    fig.suptitle(f'{model_name} Training Dashboard - RTX 4050 Optimized', fontsize=18, fontweight='bold', y=0.98)
    
    # Save the plot
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name.lower()}_training_dashboard.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Training dashboard saved: {save_path}")

# --- Main Training Function ---
def main():
    print("🚀 RTX 4050 Optimized CDCN Anti-Spoofing System")
    print("=" * 55)
    
    # Device setup
    device = setup_device()
    
    # RTX 4050 Optimized Configuration
    INPUT_SIZE = (224, 224)
    BATCH_SIZE = 64  # Effective batch size via micro-batching
    NUM_EPOCHS = 20  # As requested
    NUM_WORKERS = 6  # Optimized for RTX 4050
    LIMIT_SAMPLES = 8000  # Balanced for speed and accuracy
    
    print(f"\n⚙️ RTX 4050 Configuration:")
    print(f"   Input Size: {INPUT_SIZE}")
    print(f"   Effective Batch Size: {BATCH_SIZE}")
    print(f"   Micro-batch Size: 16 (4 steps)")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Workers: {NUM_WORKERS}")
    print(f"   Limit per class: {LIMIT_SAMPLES}")
    print(f"   🚀 GPU: RTX 4050 (6GB) Optimized")
    print()
    
    # RTX 4050 optimized data transforms
    train_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE[0] + 32, INPUT_SIZE[1] + 32)),
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)
        ], p=0.7),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.05))
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Datasets
    print("Loading datasets...")
    train_dataset = FaceAntiSpoofingDataset('../train', train_transform, LIMIT_SAMPLES)
    test_dataset = FaceAntiSpoofingDataset('../test', test_transform, LIMIT_SAMPLES)
    
    # RTX 4050 optimized data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64,  # Full batch size for data loader
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=True, 
        prefetch_factor=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32,  # Smaller test batch for memory
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=True, 
        prefetch_factor=2
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Testing batches: {len(test_loader)}")
    print()
    
    # Model
    model = CDCN_RTX4050(num_classes=2)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()
    
    # RTX 4050 optimized optimizer and scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.001,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # OneCycleLR for fast convergence
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.003,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"rtx4050_cdcn_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Training
    print("🚀 Starting RTX 4050 Optimized CDCN Training...")
    print("=" * 55)
    
    # RTX 4050 optimizations
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_per_process_memory_fraction(0.85)  # Use 85% of 6GB
        print("✅ RTX 4050 optimizations enabled")
    
    start_time = time.time()
    results = []
    best_acc = 0.0
    best_auc = 0.0
    patience = 7
    patience_counter = 0
    best_model_path = os.path.join(results_dir, 'rtx4050_cdcn_best.pth')
    
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        # Training with micro-batching
        train_metrics = train_epoch_rtx4050(model, train_loader, optimizer, criterion, device, epoch, scheduler)
        test_metrics = evaluate_epoch_rtx4050(model, test_loader, criterion, device)
        
        # Save best model
        if test_metrics['test_acc'] > best_acc:
            best_acc = test_metrics['test_acc']
            best_auc = test_metrics['auc']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': best_acc,
                'auc': best_auc
            }, best_model_path)
            print(f"💾 New best model saved! Acc: {best_acc:.4f}, AUC: {best_auc:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered after {patience} epochs without improvement")
            break
        
        epoch_time = time.time() - epoch_start
        result = {
            'epoch': epoch,
            'train_acc': train_metrics['train_acc'],
            'test_acc': test_metrics['test_acc'],
            'train_loss': train_metrics['train_loss'],
            'test_loss': test_metrics['test_loss'],
            'auc': test_metrics['auc'],
            'time': epoch_time,
            'lr': optimizer.param_groups[0]['lr']
        }
        results.append(result)
        
        print(f"Epoch {epoch:2d}: Train Acc={train_metrics['train_acc']:.3f}, "
              f"Test Acc={test_metrics['test_acc']:.3f}, AUC={test_metrics['auc']:.3f}, "
              f"Time={epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    
    # Load best model and run final evaluation
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Running final comprehensive evaluation...")
    final_test_metrics = evaluate_epoch_rtx4050(model, test_loader, criterion, device)
    
    # Prepare test metrics for comprehensive analysis
    test_metrics_dict = create_test_metrics_dict(
        y_true=final_test_metrics['targets'],
        y_pred=final_test_metrics['predictions'], 
        y_probs=final_test_metrics['probabilities']
    )
    
    # Print final results
    print(f"\n{'='*55}")
    print("RTX 4050 OPTIMIZED CDCN - FINAL RESULTS")
    print(f"{'='*55}")
    print(f"Training Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Average Time per Epoch: {total_time/len(results):.1f}s")
    print(f"Total Epochs: {len(results)}")
    print(f"Best Validation Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print()
    print("Final Test Metrics:")
    print(f"├─ Accuracy: {final_test_metrics['test_acc']:.4f} ({final_test_metrics['test_acc']*100:.2f}%)")
    print(f"├─ AUC Score: {final_test_metrics['auc']:.4f}")
    print(f"├─ Precision: {test_metrics_dict['precision']:.4f}")
    print(f"├─ Recall: {test_metrics_dict['recall']:.4f}")
    print(f"└─ F1-Score: {test_metrics_dict['f1']:.4f}")
    print()
    
    print("Generating comprehensive test analysis...")
    
    # Run comprehensive test analysis
    analyzer = ComprehensiveTestAnalyzer()
    results_save_dir = analyzer.run_complete_analysis(
        test_metrics=test_metrics_dict,
        model_name="RTX4050_CDCN",
        base_save_dir=results_dir
    )
    
    # Generate training visualization dashboard
    print("Generating training metrics visualization...")
    create_training_visualization(results, results_dir, "RTX4050_CDCN")
    
    # Save training results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, 'training_results.csv'), index=False)
    
    # Save final metrics
    final_report = {
        'model': 'RTX4050_CDCN',
        'accuracy': final_test_metrics['test_acc'],
        'auc': final_test_metrics['auc'],
        'precision': test_metrics_dict['precision'],
        'recall': test_metrics_dict['recall'],
        'f1_score': test_metrics_dict['f1'],
        'training_time': total_time,
        'epochs': len(results),
        'best_epoch': checkpoint.get('epoch', -1),
        'avg_epoch_time': total_time/len(results)
    }
    
    pd.DataFrame([final_report]).to_csv(os.path.join(results_dir, 'final_report.csv'), index=False)
    
    print(f"✅ Complete analysis saved to: {results_dir}")
    
    return results, model, test_metrics_dict

if __name__ == "__main__":
    try:
        print("🎯 RTX 4050 Optimized CDCN (Central Difference Convolutional Network)")
        print("=" * 70)
        print("🔥 BATCH SIZE 64 + 20 EPOCHS + SPEED OPTIMIZED")
        print("=" * 70)
        print()
        
        results, model, final_metrics = main()
        
        print(f"\n🎯 RTX 4050 CDCN training completed successfully!")
        print("✅ Comprehensive test analysis generated")
        print("📁 Check results directory for complete analysis")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
