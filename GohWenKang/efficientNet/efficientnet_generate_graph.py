import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import numpy as np

def load_training_history(model_dir='efficientnet_models'):
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

def plot_comprehensive_training_curves(history, save_dir='efficientnet_models'):
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

def plot_additional_training_curves(history, save_dir='efficientnet_models'):
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

def generate_training_summary(history, save_dir='efficientnet_models'):
    """Generate training summary visualization"""
    
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
    
    summary_text = f"""
    EFFICIENTNET+META TRAINING SUMMARY
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Final Performance:
    • Training Accuracy: {final_train_acc:.2f}%
    • Validation Accuracy: {final_val_acc:.2f}%
    • Weighted F1-Score: {final_f1:.4f}
    • Precision: {final_precision:.4f}
    • Recall: {final_recall:.4f}
    
    Best Performance:
    • Best Validation Accuracy: {best_val_acc:.2f}%
    • Best F1-Score: {best_f1:.4f}
    
    Overfitting Analysis:
    • Final Accuracy Gap: {overfitting_gap:.2f}%
    • Status: {"✓ Good" if overfitting_gap < 5 else "⚠ Monitor" if overfitting_gap < 10 else "✗ Overfitting"}
    
    Training Details:
    • Total Epochs: {len(history['train_acc']) if history['train_acc'] else 0}
    • Final Learning Rate: {history['learning_rates'][-1]:.2e} if history['learning_rates'] else 'N/A'
    
    Model Features:
    ✓ EfficientNet Compound Scaling
    ✓ Meta-Learning Adaptation
    ✓ Multi-Scale Attention Modules
    ✓ Hybrid Loss (Classification + Regression)
    ✓ RTX 3050 Optimized
    ✓ Mixed Precision Training
    """
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_09_training_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: train_09_training_summary.png")

def main():
    """Main function to generate all training curves"""
    print("=" * 60)
    print("EFFICIENTNET TRAINING CURVES GENERATOR")
    print("=" * 60)
    
    # Load training history
    history = load_training_history('efficientnet_models')
    
    if history is None:
        print("\nERROR: Could not load training history!")
        print("Make sure you have one of these files:")
        print("  - efficientnet_models/training_history.json")
        print("  - efficientnet_models/best_checkpoint.pth")
        print("  - efficientnet_models/checkpoint.pth")
        return
    
    print("\nTraining history loaded successfully!")
    print(f"Available metrics: {list(history.keys())}")
    print(f"Number of epochs: {len(history['train_acc']) if 'train_acc' in history else 'Unknown'}")
    
    # Generate core training curves (the 6 main graphs)
    print("\nGenerating core training curves...")
    plot_comprehensive_training_curves(history, 'efficientnet_models')
    
    # Generate additional curves if data is available
    print("\nGenerating additional training curves...")
    plot_additional_training_curves(history, 'efficientnet_models')
    
    # Generate training summary
    print("\nGenerating training summary...")
    generate_training_summary(history, 'efficientnet_models')
    
    print("\n" + "=" * 60)
    print("ALL TRAINING CURVES GENERATED SUCCESSFULLY!")
    print("=" * 60)
    print("Generated files in 'efficientnet_models/' directory:")
    print("  1. train_01_loss_curves.png")
    print("  2. train_02_accuracy_curves.png") 
    print("  3. train_03_f1_evolution.png")
    print("  4. train_04_precision_recall.png")
    print("  5. train_05_learning_rate.png")
    print("  6. train_06_overfitting_monitor.png")
    print("  7. train_07_mse_curves.png (if available)")
    print("  8. train_08_rmse_curves.png (if available)")
    print("  9. train_09_training_summary.png")
    
    print("\nYou can now view these graphs to analyze your EfficientNet+Meta model's training performance!")

if __name__ == "__main__":
    main()