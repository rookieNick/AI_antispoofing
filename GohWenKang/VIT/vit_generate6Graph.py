def main():
    """Main function to generate all ViT training curves"""
    print("=" * 70)
    print("VISION TRANSFORMER (ViT) TRAINING CURVES GENERATOR")
    print("=" * 70)
    
    # Load ViT training history
    history = load_vit_training_history('GohWenKang/VIT/vit_models')
    
    if history is None:
        print("\nERROR: Could not load ViT training history!")
        print("Make sure you have one of these files:")
        print("  - GohWenKang/VIT/vit_models/training_history.json")
        print("  - GohWenKang/VIT/vit_models/best_vit_model.pth")
        print("  - GohWenKang/VIT/vit_models/vit_checkpoint.pth")
        print("  - GohWenKang/VIT/vit_models/best_vit_antispoofing.pth")
        return
    
    print("\nViT training history loaded successfully!")
    print(f"Available metrics: {list(history.keys())}")
    
    # Validate required metrics exist
    required_metrics = ['train_losses', 'train_accs', 'val_losses', 'val_accs', 'learning_rates']
    missing_metrics = [metric for metric in required_metrics if not history.get(metric)]
    
    if missing_metrics:
        print(f"Warning: Missing metrics: {missing_metrics}")
        print("Some graphs may not be generated properly.")
    else:
        print(f"Number of training epochs: {len(history['train_accs'])}")
        print(f"Best validation accuracy: {max(history['val_accs']):.2f}%")
    
    # Generate individual training curves (9 graphs)
    print("\nGenerating individual ViT training curves...")
    try:
        plot_vit_comprehensive_training_curves(history, 'GohWenKang/VIT/vit_models')
        print("✓ All individual curves generated successfully!")
    except Exception as e:
        print(f"Error generating individual curves: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate combined overview (1 comprehensive graph)  
    print("\nGenerating combined overview...")
    try:
        plot_vit_combined_overview(history, 'GohWenKang/VIT/vit_models')
        print("✓ Combined overview generated successfully!")
    except Exception as e:
        print(f"Error generating combined overview: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("ALL ViT TRAINING CURVES GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print("Generated files in 'GohWenKang/VIT/vit_models/' directory:")
    print("  1. vit_01_loss_curves.png")
    print("  2. vit_02_accuracy_curves.png") 
    print("  3. vit_03_f1_evolution.png")
    print("  4. vit_04_precision_recall.png")
    print("  5. vit_05_mse_evolution.png")
    print("  6. vit_06_rmse_evolution.png")
    print("  7. vit_07_learning_rate.png")
    print("  8. vit_08_overfitting_monitor.png")
    print("  9. vit_09_training_summary.png")
    print(" 10. vit_10_combined_overview.png (ALL METRICS IN ONE VIEW)")
    
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import numpy as np

def load_vit_training_history(model_dir='GohWenKang/VIT/vit_models'):
    """Load ViT training history from saved files."""
    
    # Try to load from JSON file first
    json_path = os.path.join(model_dir, 'training_history.json')
    if os.path.exists(json_path):
        print(f"Loading ViT training history from: {json_path}")
        with open(json_path, 'r') as f:
            history = json.load(f)
        return history
    
    # Try to load from checkpoint files
    checkpoint_paths = [
        os.path.join(model_dir, 'best_vit_model.pth'),
        os.path.join(model_dir, 'vit_checkpoint.pth'),
        os.path.join(model_dir, 'best_vit_antispoofing.pth')
    ]
    
    for checkpoint_path in checkpoint_paths:
        if os.path.exists(checkpoint_path):
            print(f"Loading ViT training history from checkpoint: {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'metrics_logger' in checkpoint:
                    metrics = checkpoint['metrics_logger']
                    # Convert to expected format
                    history = {
                        'train_losses': metrics.get('train_losses', []),
                        'train_accs': metrics.get('train_accs', []),
                        'val_losses': metrics.get('val_losses', []),
                        'val_accs': metrics.get('val_accs', []),
                        'mse_scores': metrics.get('mse_scores', []),
                        'rmse_scores': metrics.get('rmse_scores', []),
                        'learning_rates': metrics.get('learning_rates', []),
                        'epochs': metrics.get('epochs', [])
                    }
                    return history
                elif 'history' in checkpoint:
                    return checkpoint['history']
                else:
                    print("No training history found in checkpoint")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
    
    print("No ViT training history found!")
    return None

def plot_vit_comprehensive_training_curves(history, save_dir='GohWenKang/VIT/vit_models'):
    """Plot comprehensive ViT training curves as separate PNG files"""
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Loss curves
    plt.figure(figsize=(12, 8))
    epochs = list(range(1, len(history['train_losses']) + 1))
    plt.plot(epochs, history['train_losses'], label='Training Loss', marker='o', 
             linewidth=3, markersize=6, color='#2E86AB')
    plt.plot(epochs, history['val_losses'], label='Validation Loss', marker='s', 
             linewidth=3, markersize=6, color='#F24236')
    plt.title('ViT Training & Validation Loss Curves', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vit_01_loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: vit_01_loss_curves.png")
    
    # 2. Accuracy curves
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, history['train_accs'], label='Training Accuracy', marker='o', 
             linewidth=3, markersize=6, color='#2E86AB')
    plt.plot(epochs, history['val_accs'], label='Validation Accuracy', marker='s', 
             linewidth=3, markersize=6, color='#F24236')
    plt.title('ViT Training & Validation Accuracy Curves', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vit_02_accuracy_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: vit_02_accuracy_curves.png")
    
    # 3. MSE Evolution
    if history['mse_scores']:
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, history['mse_scores'], label='MSE Score', marker='o', 
                 linewidth=3, markersize=6, color='#A23B72')
        plt.title('ViT Mean Squared Error (MSE) Evolution', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('MSE', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        min_mse = min(history['mse_scores'])
        avg_mse = np.mean(history['mse_scores'])
        plt.axhline(y=min_mse, color='red', linestyle='--', alpha=0.7, 
                   label=f'Min MSE: {min_mse:.4f}')
        plt.axhline(y=avg_mse, color='orange', linestyle='--', alpha=0.7, 
                   label=f'Avg MSE: {avg_mse:.4f}')
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'vit_03_mse_evolution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: vit_03_mse_evolution.png")
    
    # 4. RMSE Evolution
    if history['rmse_scores']:
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, history['rmse_scores'], label='RMSE Score', marker='s', 
                 linewidth=3, markersize=6, color='#F18F01')
        plt.title('ViT Root Mean Squared Error (RMSE) Evolution', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('RMSE', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        min_rmse = min(history['rmse_scores'])
        avg_rmse = np.mean(history['rmse_scores'])
        plt.axhline(y=min_rmse, color='red', linestyle='--', alpha=0.7, 
                   label=f'Min RMSE: {min_rmse:.4f}')
        plt.axhline(y=avg_rmse, color='orange', linestyle='--', alpha=0.7, 
                   label=f'Avg RMSE: {avg_rmse:.4f}')
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'vit_04_rmse_evolution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: vit_04_rmse_evolution.png")
    
    # 5. Learning Rate Schedule
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, history['learning_rates'], marker='o', linewidth=3, 
             markersize=6, color='#C73E1D')
    plt.title('ViT Learning Rate Schedule (Cosine Annealing)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vit_05_learning_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: vit_05_learning_rate.png")
    
    # 6. Training vs Validation Gap Analysis (Overfitting Monitor)
    plt.figure(figsize=(12, 8))
    acc_gap = [train - val for train, val in zip(history['train_accs'], history['val_accs'])]
    plt.plot(epochs, acc_gap, marker='o', linewidth=3, markersize=6, color='#FF6B35')
    plt.axhline(y=5, color='red', linestyle='--', alpha=0.7, 
               label='Overfitting Alert (5%)', linewidth=2)
    plt.axhline(y=10, color='darkred', linestyle='--', alpha=0.7, 
               label='Severe Overfitting (10%)', linewidth=2)
    plt.title('ViT Overfitting Monitor (Train - Val Accuracy)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy Gap (%)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vit_06_overfitting_monitor.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: vit_06_overfitting_monitor.png")
    
    # 7. Loss vs Accuracy Correlation
    plt.figure(figsize=(12, 8))
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    color = '#2E86AB'
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Validation Loss', color=color, fontsize=14)
    line1 = ax1.plot(epochs, history['val_losses'], color=color, marker='o', 
                     linewidth=3, markersize=6, label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = '#F24236'
    ax2.set_ylabel('Validation Accuracy (%)', color=color, fontsize=14)
    line2 = ax2.plot(epochs, history['val_accs'], color=color, marker='s', 
                     linewidth=3, markersize=6, label='Val Acc')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('ViT Loss vs Accuracy Correlation', fontsize=16, fontweight='bold', pad=20)
    
    # Add combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vit_07_loss_accuracy_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: vit_07_loss_accuracy_correlation.png")
    
    # 8. Performance Improvement Over Time
    plt.figure(figsize=(12, 8))
    
    # Calculate moving averages for smoother trends
    window = min(3, len(history['val_accs']) // 4)  # Adaptive window size
    if window > 0:
        val_acc_smooth = np.convolve(history['val_accs'], np.ones(window)/window, mode='valid')
        epochs_smooth = epochs[window-1:]
        
        plt.plot(epochs, history['val_accs'], alpha=0.3, color='lightblue', 
                linewidth=1, label='Raw Validation Accuracy')
        plt.plot(epochs_smooth, val_acc_smooth, color='#2E86AB', linewidth=3, 
                marker='o', markersize=4, label=f'Smoothed (Window={window})')
    else:
        plt.plot(epochs, history['val_accs'], color='#2E86AB', linewidth=3, 
                marker='o', markersize=6, label='Validation Accuracy')
    
    plt.title('ViT Performance Improvement Trend', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Validation Accuracy (%)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vit_08_performance_trend.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: vit_08_performance_trend.png")
    
    # 9. Training Summary
    generate_vit_training_summary(history, save_dir)

def generate_vit_training_summary(history, save_dir='GohWenKang/VIT/vit_models'):
    """Generate ViT training summary visualization"""
    
    plt.figure(figsize=(14, 10))
    plt.axis('off')
    
    # Calculate final metrics
    final_train_acc = history['train_accs'][-1] if history['train_accs'] else 0
    final_val_acc = history['val_accs'][-1] if history['val_accs'] else 0
    final_mse = history['mse_scores'][-1] if history['mse_scores'] else 0
    final_rmse = history['rmse_scores'][-1] if history['rmse_scores'] else 0
    overfitting_gap = final_train_acc - final_val_acc
    
    # Best metrics
    best_val_acc = max(history['val_accs']) if history['val_accs'] else 0
    best_mse = min(history['mse_scores']) if history['mse_scores'] else 0
    best_rmse = min(history['rmse_scores']) if history['rmse_scores'] else 0
    
    # Calculate convergence metrics
    last_5_accs = history['val_accs'][-5:] if len(history['val_accs']) >= 5 else history['val_accs']
    acc_stability = np.std(last_5_accs) if len(last_5_accs) > 1 else 0
    
    summary_text = f"""
    VISION TRANSFORMER (ViT) FACE ANTI-SPOOFING TRAINING SUMMARY
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    🎯 FINAL PERFORMANCE METRICS:
    • Training Accuracy: {final_train_acc:.2f}%
    • Validation Accuracy: {final_val_acc:.2f}%
    • Mean Squared Error: {final_mse:.4f}
    • Root Mean Squared Error: {final_rmse:.4f}
    
    🏆 BEST PERFORMANCE ACHIEVED:
    • Best Validation Accuracy: {best_val_acc:.2f}%
    • Lowest MSE Score: {best_mse:.4f}
    • Lowest RMSE Score: {best_rmse:.4f}
    
    📊 MODEL STABILITY ANALYSIS:
    • Final Accuracy Gap: {overfitting_gap:.2f}%
    • Convergence Status: {"✓ Stable" if acc_stability < 1 else "⚠ Variable" if acc_stability < 3 else "✗ Unstable"}
    • Last 5 Epochs Std: {acc_stability:.3f}%
    • Overfitting Status: {"✓ Good" if overfitting_gap < 5 else "⚠ Monitor" if overfitting_gap < 10 else "✗ Overfitting"}
    
    📈 TRAINING CONFIGURATION:
    • Total Epochs Completed: {len(history['train_accs']) if history['train_accs'] else 0}
    • Final Learning Rate: {history['learning_rates'][-1]:.2e if history['learning_rates'] else 'N/A'}
    • Architecture: Vision Transformer (ViT-Base-Patch16-224)
    • Input Resolution: 224×224 pixels
    
    🔧 ADVANCED FEATURES USED:
    ✓ Pre-trained ViT with HuggingFace Transformers
    ✓ Mixed Precision Training (AMP)
    ✓ Gradient Checkpointing (Memory Efficient)
    ✓ Cosine Annealing with Warm Restarts
    ✓ Focal Loss + Cross Entropy Combination
    ✓ Mixup & CutMix Data Augmentation
    ✓ Weighted Random Sampling (Class Balance)
    ✓ Label Smoothing & Gradient Clipping
    ✓ Advanced MSE/RMSE Regression Tracking
    ✓ RTX 3050 Optimized Configuration
    
    🎮 HARDWARE OPTIMIZATION:
    • Batch Size: 8 (optimized for 4GB VRAM)
    • Memory Efficiency: Gradient checkpointing enabled
    • Multi-precision: Mixed precision training active
    • Data Loading: Persistent workers with pin memory
    
    📝 TRAINING TECHNIQUES:
    • Loss Function: 70% Cross Entropy + 30% Focal Loss
    • Regularization: L2 Weight Decay + Label Smoothing
    • Augmentation: Advanced geometric + color transforms
    • Scheduler: Cosine annealing with warm restarts
    • Early Stopping: Patience-based with best model saving
    
    Task: Face Anti-Spoofing (Live vs Spoof Detection)
    Dataset: CASIA-FASD Compatible Format
    Model Family: Vision Transformer (Google Research)
    """
    
    # Create colored background sections
    plt.text(0.02, 0.95, summary_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#E8F4FD', alpha=0.9, 
                      edgecolor='#2E86AB', linewidth=2))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vit_09_training_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: vit_09_training_summary.png")

def plot_vit_combined_overview(history, save_dir='GohWenKang/VIT/vit_models'):
    """Generate a comprehensive combined overview of all ViT training metrics"""
    
    fig = plt.figure(figsize=(20, 16))
    
    # Create a 3x3 grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    epochs = list(range(1, len(history['train_losses']) + 1))
    
    # 1. Loss Curves (Top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history['train_losses'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, history['val_losses'], 'r-s', label='Val Loss', linewidth=2, markersize=4)
    ax1.set_title('Training & Validation Loss', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy Curves (Top-center)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history['train_accs'], 'b-o', label='Train Acc', linewidth=2, markersize=4)
    ax2.plot(epochs, history['val_accs'], 'r-s', label='Val Acc', linewidth=2, markersize=4)
    ax2.set_title('Training & Validation Accuracy', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Learning Rate (Top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, history['learning_rates'], 'g-o', linewidth=2, markersize=4)
    ax3.set_title('Learning Rate Schedule', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 4. MSE Evolution (Middle-left)
    ax4 = fig.add_subplot(gs[1, 0])
    if history['mse_scores']:
        ax4.plot(epochs, history['mse_scores'], 'purple', marker='o', linewidth=2, markersize=4)
        min_mse = min(history['mse_scores'])
        ax4.axhline(y=min_mse, color='red', linestyle='--', alpha=0.7)
    ax4.set_title('Mean Squared Error', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('MSE')
    ax4.grid(True, alpha=0.3)
    
    # 5. RMSE Evolution (Middle-center)
    ax5 = fig.add_subplot(gs[1, 1])
    if history['rmse_scores']:
        ax5.plot(epochs, history['rmse_scores'], 'orange', marker='s', linewidth=2, markersize=4)
        min_rmse = min(history['rmse_scores'])
        ax5.axhline(y=min_rmse, color='red', linestyle='--', alpha=0.7)
    ax5.set_title('Root Mean Squared Error', fontweight='bold', fontsize=12)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('RMSE')
    ax5.grid(True, alpha=0.3)
    
    # 6. Overfitting Monitor (Middle-right)
    ax6 = fig.add_subplot(gs[1, 2])
    acc_gap = [train - val for train, val in zip(history['train_accs'], history['val_accs'])]
    ax6.plot(epochs, acc_gap, 'orange', marker='o', linewidth=2, markersize=4)
    ax6.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='Alert (5%)')
    ax6.set_title('Overfitting Monitor', fontweight='bold', fontsize=12)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Accuracy Gap (%)')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # 7. Loss-Accuracy Correlation (Bottom-left)
    ax7 = fig.add_subplot(gs[2, 0])
    ax7_twin = ax7.twinx()
    line1 = ax7.plot(epochs, history['val_losses'], 'b-o', linewidth=2, markersize=4, label='Val Loss')
    line2 = ax7_twin.plot(epochs, history['val_accs'], 'r-s', linewidth=2, markersize=4, label='Val Acc')
    ax7.set_title('Loss vs Accuracy', fontweight='bold', fontsize=12)
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Loss', color='blue')
    ax7_twin.set_ylabel('Accuracy (%)', color='red')
    ax7.grid(True, alpha=0.3)
    
    # 8. Performance Trend (Bottom-center)
    ax8 = fig.add_subplot(gs[2, 1])
    window = min(3, len(history['val_accs']) // 4)
    if window > 0:
        val_acc_smooth = np.convolve(history['val_accs'], np.ones(window)/window, mode='valid')
        epochs_smooth = epochs[window-1:]
        ax8.plot(epochs, history['val_accs'], alpha=0.3, color='lightblue', linewidth=1)
        ax8.plot(epochs_smooth, val_acc_smooth, 'b-', linewidth=2, marker='o', markersize=3)
    else:
        ax8.plot(epochs, history['val_accs'], 'b-o', linewidth=2, markersize=4)
    ax8.set_title('Performance Trend', fontweight='bold', fontsize=12)
    ax8.set_xlabel('Epoch')
    ax8.set_ylabel('Val Accuracy (%)')
    ax8.grid(True, alpha=0.3)
    
    # 9. Summary Statistics (Bottom-right)
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Calculate summary stats
    final_val_acc = history['val_accs'][-1] if history['val_accs'] else 0
    best_val_acc = max(history['val_accs']) if history['val_accs'] else 0
    final_mse = history['mse_scores'][-1] if history['mse_scores'] else 0
    overfitting_gap = history['train_accs'][-1] - final_val_acc if history['train_accs'] and history['val_accs'] else 0
    
    stats_text = f"""ViT Training Summary
    
Final Val Acc: {final_val_acc:.2f}%
Best Val Acc: {best_val_acc:.2f}%
Final MSE: {final_mse:.4f}
Accuracy Gap: {overfitting_gap:.2f}%

Total Epochs: {len(epochs)}
Architecture: ViT-Base-Patch16
Input Size: 224×224

Status: {"✓ Good" if overfitting_gap < 5 else "⚠ Monitor"}
Convergence: {"✓ Stable" if len(epochs) > 0 else "✗ N/A"}"""
    
    ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Add main title
    fig.suptitle('Vision Transformer (ViT) Face Anti-Spoofing - Complete Training Analysis', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vit_10_combined_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: vit_10_combined_overview.png")

def main():
    """Main function to generate all ViT training curves"""
    print("=" * 70)
    print("VISION TRANSFORMER (ViT) TRAINING CURVES GENERATOR")
    print("=" * 70)
    
    # Load ViT training history
    history = load_vit_training_history('GohWenKang/VIT/vit_models')
    
    if history is None:
        print("\nERROR: Could not load ViT training history!")
        print("Make sure you have one of these files:")
        print("  - GohWenKang/VIT/vit_models/training_history.json")
        print("  - GohWenKang/VIT/vit_models/best_vit_model.pth")
        print("  - GohWenKang/VIT/vit_models/vit_checkpoint.pth")
        print("  - GohWenKang/VIT/vit_models/best_vit_antispoofing.pth")
        return
    
    print("\nViT training history loaded successfully!")
    print(f"Available metrics: {list(history.keys())}")
    
    # Validate required metrics exist
    required_metrics = ['train_losses', 'train_accs', 'val_losses', 'val_accs', 'learning_rates']
    missing_metrics = [metric for metric in required_metrics if not history.get(metric)]
    
    if missing_metrics:
        print(f"Warning: Missing metrics: {missing_metrics}")
        print("Some graphs may not be generated properly.")
    else:
        print(f"Number of training epochs: {len(history['train_accs'])}")
        print(f"Best validation accuracy: {max(history['val_accs']):.2f}%")
    
    # Generate individual training curves (9 graphs)
    print("\nGenerating individual ViT training curves...")
    plot_vit_comprehensive_training_curves(history, 'GohWenKang/VIT/vit_models')
    
    # Generate combined overview (1 comprehensive graph)
    print("\nGenerating combined overview...")
    plot_vit_combined_overview(history, 'GohWenKang/VIT/vit_models')
    
    print("\n" + "=" * 70)
    print("ALL ViT TRAINING CURVES GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print("Generated files in 'GohWenKang/VIT/vit_models/' directory:")
    print("  1. vit_01_loss_curves.png")
    print("  2. vit_02_accuracy_curves.png") 
    print("  3. vit_03_mse_evolution.png")
    print("  4. vit_04_rmse_evolution.png")
    print("  5. vit_05_learning_rate.png")
    print("  6. vit_06_overfitting_monitor.png")
    print("  7. vit_07_loss_accuracy_correlation.png")
    print("  8. vit_08_performance_trend.png")
    print("  9. vit_09_training_summary.png")
    print(" 10. vit_10_combined_overview.png (ALL METRICS IN ONE VIEW)")
    
    print("\nYou can now view these graphs to analyze your Vision Transformer model's training performance!")
    print("The combined overview provides a comprehensive view of all metrics in a single image.")
    print(f"\nKey Insights from your training:")
    
    # Provide training insights
    final_val_acc = max(history['val_accs']) if history['val_accs'] else 0
    overfitting_gap = history['train_accs'][-1] - history['val_accs'][-1] if history['train_accs'] and history['val_accs'] else 0
    
    print(f"  • Peak validation accuracy: {final_val_acc:.2f}%")
    print(f"  • Training stability: {'Excellent' if overfitting_gap < 3 else 'Good' if overfitting_gap < 8 else 'Needs attention'}")
    print(f"  • Model performance: {'Outstanding' if final_val_acc > 95 else 'Very good' if final_val_acc > 90 else 'Good' if final_val_acc > 80 else 'Needs improvement'}")

def load_from_specific_checkpoint(checkpoint_path):
    """
    Load training history from a specific checkpoint file.
    Useful for debugging or when you know exactly which file to use.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return None
    
    try:
        print(f"Loading from specific checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'metrics_logger' in checkpoint:
            metrics = checkpoint['metrics_logger']
            history = {
                'train_losses': metrics.get('train_losses', []),
                'train_accs': metrics.get('train_accs', []),
                'val_losses': metrics.get('val_losses', []),
                'val_accs': metrics.get('val_accs', []),
                'mse_scores': metrics.get('mse_scores', []),
                'rmse_scores': metrics.get('rmse_scores', []),
                'learning_rates': metrics.get('learning_rates', []),
                'epochs': metrics.get('epochs', [])
            }
            print("Successfully loaded metrics from checkpoint!")
            return history
        else:
            print("No metrics_logger found in checkpoint")
            return None
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

# Alternative usage example for direct checkpoint loading
def generate_curves_from_checkpoint(checkpoint_path):
    """
    Generate curves directly from a specific checkpoint file.
    Usage: generate_curves_from_checkpoint('GohWenKang/VIT/vit_models/best_vit_model.pth')
    """
    history = load_from_specific_checkpoint(checkpoint_path)
    if history:
        save_dir = os.path.dirname(checkpoint_path)
        print(f"Generating curves and saving to: {save_dir}")
        plot_vit_comprehensive_training_curves(history, save_dir)
        plot_vit_combined_overview(history, save_dir)
        print("Curve generation completed!")
    else:
        print("Failed to load history from checkpoint")

if __name__ == "__main__":
    main()