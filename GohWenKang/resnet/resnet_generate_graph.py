import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import numpy as np

def load_training_history(model_dir='resnet_attention_models'):
    """Load training history from saved files."""
    
    # Try to load from JSON file first
    json_path = os.path.join(model_dir, 'training_history.json')
    if os.path.exists(json_path):
        print(f"Loading training history from: {json_path}")
        with open(json_path, 'r') as f:
            history = json.load(f)
        return history
    
    # Try to load from checkpoint
    checkpoint_paths = [
        os.path.join(model_dir, 'best_checkpoint.pth'),
        os.path.join(model_dir, 'checkpoint.pth')
    ]
    
    for checkpoint_path in checkpoint_paths:
        if os.path.exists(checkpoint_path):
            print(f"Loading training history from checkpoint: {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'history' in checkpoint:
                    return checkpoint['history']
                else:
                    print("No training history found in checkpoint")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
    
    print("No training history found!")
    return None

def plot_comprehensive_training_curves(history, save_dir='resnet_attention_models'):
    """Plot comprehensive training curves as separate PNG files"""
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss', marker='o', linewidth=2, color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', marker='s', linewidth=2, color='orange')
    plt.title('Training & Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_01_loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: train_01_loss_curves.png")
    
    # 2. Accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_acc'], label='Training Accuracy', marker='o', linewidth=2, color='blue')
    plt.plot(history['val_acc'], label='Validation Accuracy', marker='s', linewidth=2, color='orange')
    plt.title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_02_accuracy_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: train_02_accuracy_curves.png")
    
    # 3. F1 Score comparison
    plt.figure(figsize=(10, 6))
    plt.plot(history['val_f1'], label='Weighted F1', marker='o', linewidth=2, color='blue')
    plt.plot(history['val_macro_f1'], label='Macro F1', marker='s', linewidth=2, color='orange')
    plt.title('F1 Score Evolution', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_03_f1_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: train_03_f1_evolution.png")
    
    # 4. Precision and Recall
    plt.figure(figsize=(10, 6))
    plt.plot(history['val_precision'], label='Precision', marker='o', linewidth=2, color='blue')
    plt.plot(history['val_recall'], label='Recall', marker='s', linewidth=2, color='orange')
    plt.title('Precision and Recall Evolution', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_04_precision_recall.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: train_04_precision_recall.png")
    
    # 5. Learning Rate Schedule
    plt.figure(figsize=(10, 6))
    plt.plot(history['learning_rates'], marker='o', linewidth=2, color='red')
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_05_learning_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: train_05_learning_rate.png")
    
    # 6. Training vs Validation Gap Analysis (Overfitting Monitor)
    plt.figure(figsize=(10, 6))
    acc_gap = [train - val for train, val in zip(history['train_acc'], history['val_acc'])]
    plt.plot(acc_gap, marker='o', linewidth=2, color='orange')
    plt.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='Overfitting Alert (5%)')
    plt.title('Overfitting Monitor (Train - Val Accuracy)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Gap (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_06_overfitting_monitor.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: train_06_overfitting_monitor.png")

def plot_additional_training_curves(history, save_dir='resnet_attention_models'):
    """Plot additional training curves including MSE/RMSE if available"""
    
    # Check if MSE/RMSE data is available
    if 'train_mse' in history and 'val_mse' in history:
        # MSE curves
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_mse'], label='Training MSE', marker='o', linewidth=2, color='orange')
        plt.plot(history['val_mse'], label='Validation MSE', marker='s', linewidth=2, color='red')
        plt.title('Mean Squared Error (MSE)', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'train_07_mse_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: train_07_mse_curves.png")
    else:
        print("⚠ MSE data not available (ResNet+Attention is classification-only)")
    
    if 'train_rmse' in history and 'val_rmse' in history:
        # RMSE curves
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_rmse'], label='Training RMSE', marker='o', linewidth=2, color='purple')
        plt.plot(history['val_rmse'], label='Validation RMSE', marker='s', linewidth=2, color='brown')
        plt.title('Root Mean Squared Error (RMSE)', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'train_08_rmse_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: train_08_rmse_curves.png")
    else:
        print("⚠ RMSE data not available (ResNet+Attention is classification-only)")

def plot_attention_analysis(history, save_dir='resnet_attention_models'):
    """Plot ResNet+Attention specific analysis curves"""
    
    # 7. Attention Impact Analysis (if available)
    if 'attention_weights' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(history['attention_weights'], marker='o', linewidth=2, color='green')
        plt.title('Attention Weights Evolution', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Average Attention Weight')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'train_09_attention_weights.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: train_09_attention_weights.png")
    
    # 8. Model Complexity Analysis
    plt.figure(figsize=(10, 6))
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    # Calculate training efficiency (accuracy improvement per epoch)
    acc_improvement = np.diff(history['val_acc'])
    acc_improvement = np.insert(acc_improvement, 0, history['val_acc'][0])  # Add initial value
    
    plt.plot(epochs, acc_improvement, marker='o', linewidth=2, color='purple', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Validation Accuracy Improvement per Epoch', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Improvement (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_10_learning_efficiency.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: train_10_learning_efficiency.png")

def generate_training_summary(history, save_dir='resnet_attention_models'):
    """Generate training summary visualization for ResNet+Attention"""
    
    plt.figure(figsize=(12, 8))
    plt.axis('off')
    
    # Calculate final metrics
    final_train_acc = history['train_acc'][-1] if history['train_acc'] else 0
    final_val_acc = history['val_acc'][-1] if history['val_acc'] else 0
    final_f1 = history['val_f1'][-1] if history['val_f1'] else 0
    final_precision = history['val_precision'][-1] if history['val_precision'] else 0
    final_recall = history['val_recall'][-1] if history['val_recall'] else 0
    overfitting_gap = final_train_acc - final_val_acc
    
    # Best metrics
    best_val_acc = max(history['val_acc']) if history['val_acc'] else 0
    best_f1 = max(history['val_f1']) if history['val_f1'] else 0
    
    # Calculate convergence epoch (where best accuracy was achieved)
    best_acc_epoch = history['val_acc'].index(best_val_acc) + 1 if history['val_acc'] else 0
    
    # Determine overfitting status
    if overfitting_gap < 5:
        overfitting_status = "✓ Good Generalization"
    elif overfitting_gap < 10:
        overfitting_status = "⚠ Monitor Carefully"
    else:
        overfitting_status = "✗ High Overfitting"
    
    # Format learning rate safely
    final_lr = history['learning_rates'][-1] if history['learning_rates'] else None
    lr_text = f"{final_lr:.2e}" if final_lr is not None else 'N/A'
    
    # Calculate total epochs
    total_epochs = len(history['train_acc']) if history['train_acc'] else 0
    
    summary_text = f"""
    RESNET+ATTENTION TRAINING SUMMARY
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Final Performance:
    • Training Accuracy: {final_train_acc:.2f}%
    • Validation Accuracy: {final_val_acc:.2f}%
    • Weighted F1-Score: {final_f1:.4f}
    • Precision: {final_precision:.4f}
    • Recall: {final_recall:.4f}
    
    Best Performance:
    • Best Validation Accuracy: {best_val_acc:.2f}%
    • Best F1-Score: {best_f1:.4f}
    • Achieved at Epoch: {best_acc_epoch}
    
    Overfitting Analysis:
    • Final Accuracy Gap: {overfitting_gap:.2f}%
    • Status: {overfitting_status}
    
    Training Details:
    • Total Epochs: {total_epochs}
    • Final Learning Rate: {lr_text}
    • Best Epoch: {best_acc_epoch}/{total_epochs}
    
    ResNet+Attention Features:
    ✓ Residual Skip Connections
    ✓ CBAM Attention (Channel + Spatial)
    ✓ Focal Loss for Class Imbalance
    ✓ Cosine Annealing Scheduler
    ✓ Mixed Precision Training
    ✓ RTX 3050 Optimized
    ✓ Gradient Clipping
    ✓ Label Smoothing
    
    Architecture Highlights:
    • Base Channels: 64
    • Residual Blocks: [2, 2, 2, 2]
    • Attention Reduction Ratio: 16
    • Dropout Rate: 0.3
    • Global Attention on Features
    """
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_11_training_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: train_11_training_summary.png")

def generate_model_comparison_chart(save_dir='resnet_attention_models'):
    """Generate a comparison chart showing ResNet+Attention advantages"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    comparison_text = """
    RESNET+ATTENTION VS OTHER ARCHITECTURES
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    ResNet+Attention Advantages:
    ┌─────────────────────────────────────────────────────────────┐
    │ ✓ Skip Connections: Prevent vanishing gradients            │
    │ ✓ CBAM Attention: Focus on important features & regions    │
    │ ✓ Focal Loss: Better handling of class imbalance           │
    │ ✓ Memory Efficient: Optimized for RTX 3050 (4GB VRAM)     │
    │ ✓ Gradient Stability: Residual learning path               │
    │ ✓ Feature Refinement: Multi-level attention mechanisms     │
    └─────────────────────────────────────────────────────────────┘
    
    Comparison with Other Models:
    
    📊 vs Standard CNN:
       • Better gradient flow (skip connections)
       • More focused feature learning (attention)
       • Better convergence stability
    
    🔍 vs Vision Transformer (ViT):
       • Lower memory requirements
       • Faster training on smaller datasets
       • Better inductive bias for images
       • More interpretable attention maps
    
    ⚡ vs EfficientNet:
       • Simpler architecture (easier to debug)
       • More robust attention mechanism
       • Better gradient flow
       • Less prone to overfitting
    
    🎯 Anti-Spoofing Specific Benefits:
       • Channel attention focuses on texture differences
       • Spatial attention highlights facial regions
       • Residual connections preserve fine details
       • Focal loss handles live/spoof imbalance
    
    💡 Training Optimizations:
       • Mixed precision for RTX 3050
       • Cosine annealing with warm restarts
       • Gradient clipping for stability
       • Label smoothing for regularization
    """
    
    ax.text(0.05, 0.95, comparison_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_12_model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: train_12_model_comparison.png")

def main():
    """Main function to generate all training curves for ResNet+Attention"""
    print("=" * 70)
    print("RESNET+ATTENTION TRAINING CURVES GENERATOR")
    print("=" * 70)
    
    # Load training history
    history = load_training_history('resnet_attention_models')
    
    if history is None:
        print("\nERROR: Could not load training history!")
        print("Make sure you have one of these files:")
        print("  - resnet_attention_models/training_history.json")
        print("  - resnet_attention_models/best_checkpoint.pth")
        print("  - resnet_attention_models/checkpoint.pth")
        return
    
    print("\nTraining history loaded successfully!")
    print(f"Available metrics: {list(history.keys())}")
    print(f"Number of epochs: {len(history['train_acc']) if 'train_acc' in history else 'Unknown'}")
    
    # Check if best model exists
    best_model_path = 'resnet_attention_models/best_resnet_attention_model.pth'
    if os.path.exists(best_model_path):
        print(f"✓ Best model found: {best_model_path}")
    else:
        print(f"⚠ Best model not found at: {best_model_path}")
    
    # Generate core training curves
    print("\nGenerating core training curves...")
    plot_comprehensive_training_curves(history, 'resnet_attention_models')
    
    # Generate additional curves (MSE/RMSE if available)
    print("\nChecking for additional metrics...")
    plot_additional_training_curves(history, 'resnet_attention_models')
    
    # Generate ResNet+Attention specific analysis
    print("\nGenerating ResNet+Attention specific analysis...")
    plot_attention_analysis(history, 'resnet_attention_models')
    
    # Generate training summary
    print("\nGenerating training summary...")
    generate_training_summary(history, 'resnet_attention_models')
    
    # Generate model comparison chart
    print("\nGenerating model comparison chart...")
    generate_model_comparison_chart('resnet_attention_models')
    
    print("\n" + "=" * 70)
    print("ALL RESNET+ATTENTION TRAINING CURVES GENERATED!")
    print("=" * 70)
    print("Generated files in 'resnet_attention_models/' directory:")
    print("  1. train_01_loss_curves.png")
    print("  2. train_02_accuracy_curves.png") 
    print("  3. train_03_f1_evolution.png")
    print("  4. train_04_precision_recall.png")
    print("  5. train_05_learning_rate.png")
    print("  6. train_06_overfitting_monitor.png")
    print("  7. train_07_mse_curves.png (if MSE data available)")
    print("  8. train_08_rmse_curves.png (if RMSE data available)")
    print("  9. train_09_attention_weights.png (if available)")
    print(" 10. train_10_learning_efficiency.png")
    print(" 11. train_11_training_summary.png")
    print(" 12. train_12_model_comparison.png")
    
    print("\nNOTE: MSE/RMSE graphs will only be generated if your")
    print("ResNet+Attention model tracked these metrics during training.")
    print("Currently, the model focuses on classification metrics only.")
    
    print("\n🎯 Use best_resnet_attention_model.pth for final testing!")

if __name__ == "__main__":
    main()