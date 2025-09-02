import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTForImageClassification, ViTConfig
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import signal
import sys

class CASIAFASDDataset(Dataset):
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

class ViTAntiSpoofing:
    def __init__(self, model_name='google/vit-base-patch16-224', num_classes=2, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.checkpoint_saved = False
        
        # Performance optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # Faster convolutions
            torch.backends.cuda.matmul.allow_tf32 = True  # Faster matmul on RTX 30xx
            print("✓ CUDA optimizations enabled")
        
        # Initialize ViT model
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Enable gradient checkpointing for memory efficiency (RTX 3050 optimization)
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled (saves memory)")
        
        self.model.to(self.device)
        
        # Mixed precision training for speed boost
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
            print("✓ Mixed precision training enabled (faster training)")
        
        # Setup signal handler for graceful interruption
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_data(self, train_dir, test_dir, batch_size=16, num_workers=4):
        # Create datasets
        train_dataset = CASIAFASDDataset(train_dir, transform=self.train_transform)
        test_dataset = CASIAFASDDataset(test_dir, transform=self.test_transform)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
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
        
        # FIXED: Return the data loaders
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
            'best_val_acc': getattr(self, 'best_val_acc', 0.0)
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
    
    def train(self, epochs=10, learning_rate=1e-4, weight_decay=1e-4, save_dir='models', resume_from_checkpoint=None):
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
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
        
        print(f"🚀 Starting training on {self.device}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        if self.use_amp:
            print("⚡ Mixed precision training active")
        
        for epoch in range(start_epoch, epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for batch_idx, (images, labels) in enumerate(train_pbar):
                # Check for interruption
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
                        loss = criterion(outputs.logits, labels)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images)
                    loss = criterion(outputs.logits, labels)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.logits.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # Validation phase
            val_loss, val_acc = self.evaluate()
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save history
            epoch_train_loss = train_loss / len(self.train_loader)
            epoch_train_acc = 100. * train_correct / train_total
            
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save checkpoint every epoch
            self.save_checkpoint(epoch, self.model, optimizer, scheduler, history, save_dir)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.best_val_acc = best_val_acc
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'best_vit_model.pth'))
                self.save_checkpoint(epoch, self.model, optimizer, scheduler, history, save_dir, is_best=True)
                print(f'  🏆 New best model saved! Val Acc: {best_val_acc:.2f}%')
            
            print('-' * 60)
        
        # Save training history and final model
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'final_vit_model.pth'))
        
        # Plot training curves
        self.plot_training_curves(history, save_dir)
        
        print("🎉 Training completed successfully!")
        return history
    
    def evaluate(self):
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = criterion(outputs.logits, labels)
                else:
                    outputs = self.model(images)
                    loss = criterion(outputs.logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(self.test_loader)
        val_acc = 100. * val_correct / val_total
        
        return val_loss, val_acc
    
    def test_and_confusion_matrix(self, model_path=None):
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        print("Generating predictions for confusion matrix...")
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.logits.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        cm = confusion_matrix(all_labels, all_predictions)
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print(f"Test Accuracy (%): {accuracy*100:.2f}%")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm, accuracy)
        
        # Classification report
        class_names = ['spoof', 'live']
        report = classification_report(all_labels, all_predictions, target_names=class_names)
        print("\nClassification Report:")
        print(report)
        
        return cm, accuracy
    
    def plot_confusion_matrix(self, cm, accuracy, save_path='confusion_matrix.png'):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['live', 'spoof'], 
                   yticklabels=['live', 'spoof'],
                   annot_kws={'size': 16})
        plt.title(f'Final Confusion Matrix\nAccuracy: {accuracy*100:.2f}%', fontsize=14)
        plt.xlabel('Predicted label', fontsize=12)
        plt.ylabel('True label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Confusion matrix saved as {save_path}")
    
    def plot_training_curves(self, history, save_dir):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(history['train_loss'], label='Training Loss', marker='o')
        ax1.plot(history['val_loss'], label='Validation Loss', marker='s')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(history['train_acc'], label='Training Accuracy', marker='o')
        ax2.plot(history['val_acc'], label='Validation Accuracy', marker='s')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Configuration
    config = {
        'train_dir': 'CASIA-FASD/train',  # Update this path - currently: CASIA-FASD/train
        'test_dir': 'CASIA-FASD/test',    # Update this path - currently: CASIA-FASD/test  
        'batch_size': 8,                  # Safe for RTX 3050 4GB (try 16 if you want to push it)
        'epochs': 20,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_workers': 2,                 # Reduced for stability
        'save_dir': 'vit_models',
        'resume_checkpoint': None         # Set to checkpoint path to resume
    }
    
    print("🔧 Initializing ViT Anti-Spoofing Model...")
    
    # Initialize model
    vit_model = ViTAntiSpoofing()
    
    print(f"💻 Using device: {vit_model.device}")
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"🎮 GPU: {gpu_props.name}")
        print(f"💾 GPU Memory: {gpu_props.total_memory / 1e9:.1f} GB")
        print(f"⚡ CUDA Compute Capability: {gpu_props.major}.{gpu_props.minor}")
    
    # Load data
    print(f"\n📂 Loading data...")
    try:
        train_loader, test_loader = vit_model.load_data(
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
    print(f"\n🚀 Starting training...")
    print("💡 Tip: Press Ctrl+C to save checkpoint and pause training")
    
    try:
        history = vit_model.train(
            epochs=config['epochs'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            save_dir=config['save_dir'],
            resume_from_checkpoint=resume_path
        )
        
        if history is None:
            print("❌ Training returned None - something went wrong")
            return None, None, None, None
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None
    
    # Test and generate confusion matrix
    print(f"\n🧪 Testing best model...")
    try:
        best_model_path = os.path.join(config['save_dir'], 'best_vit_model.pth')
        if os.path.exists(best_model_path):
            cm, accuracy = vit_model.test_and_confusion_matrix(model_path=best_model_path)
        else:
            print("⚠️ No best model found, testing current model...")
            cm, accuracy = vit_model.test_and_confusion_matrix()
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return vit_model, history, None, None
    
    print(f"\n🎉 Final Results:")
    print(f"🏆 Best Test Accuracy: {accuracy*100:.2f}%")
    print(f"📊 Confusion Matrix:\n{cm}")
    
    return vit_model, history, cm, accuracy

def resume_training_example():
    """Example function showing how to resume training"""
    print("📖 To resume training from a checkpoint:")
    print("1. Set resume_checkpoint in config:")
    print("   config['resume_checkpoint'] = 'vit_models/checkpoint.pth'")
    print("2. Or the script will auto-detect and ask you")
    print("3. Training will continue from where it left off")
    print("4. All optimizer states and history are preserved")

def monitor_gpu():
    """Show GPU monitoring commands"""
    print("\n🖥️  GPU MONITORING COMMANDS:")
    print("=" * 50)
    print("📊 Check GPU usage:")
    print("   nvidia-smi")
    print("\n🔄 Continuous monitoring (updates every 1 second):")
    print("   watch -n 1 nvidia-smi")
    print("\n📈 Monitor specific GPU processes:")
    print("   nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used --format=csv --loop=1")
    print("\n🛑 Kill process if needed:")
    print("   nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits")
    print("   kill -9 <process_id>")
    print("=" * 50)

if __name__ == "__main__":
    print("🎯 ViT Anti-Spoofing Training Script")
    print("=" * 50)
    
    # Show GPU monitoring info
    monitor_gpu()
    
    # Run main training
    try:
        model, history, cm, accuracy = main()
        if model is not None:
            print("✅ Training completed successfully!")
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