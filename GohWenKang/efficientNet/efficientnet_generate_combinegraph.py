import torch
import matplotlib.pyplot as plt
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

def create_combined_training_curves():
    """Create one combined picture with all 6 training curves."""
    
    # Load training history
    history = load_training_history('efficientnet_models')
    
    if history is None:
        print("ERROR: Could not load training history!")
        print("Creating with dummy data for demonstration...")
        # Create dummy data
        epochs = list(range(1, 21))
        history = {
            'train_loss': [0.8 - 0.03*i for i in epochs],
            'val_loss': [0.9 - 0.035*i for i in epochs],
            'train_acc': [60 + 1.5*i for i in epochs],
            'val_acc': [55 + 1.3*i for i in epochs],
            'val_f1': [0.5 + 0.02*i for i in epochs],
            'val_macro_f1': [0.45 + 0.018*i for i in epochs],
            'val_precision': [0.6 + 0.015*i for i in epochs],
            'val_recall': [0.5 + 0.02*i for i in epochs],
            'learning_rates': [0.001 * (0.9**i) for i in epochs]
        }
    
    print(f"Creating combined plot with {len(history.get('train_acc', []))} epochs of data...")
    
    # Create the combined figure
    plt.figure(figsize=(20, 12))
    plt.suptitle('EfficientNet+Meta Training Curves', fontsize=20, fontweight='bold', y=0.95)
    
    # 1. Loss curves (subplot 1)
    plt.subplot(2, 3, 1)
    plt.plot(history['train_loss'], label='Training Loss', marker='o', linewidth=2, color='blue', markersize=3)
    plt.plot(history['val_loss'], label='Validation Loss', marker='s', linewidth=2, color='orange', markersize=3)
    plt.title('Training & Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Accuracy curves (subplot 2)
    plt.subplot(2, 3, 2)
    plt.plot(history['train_acc'], label='Training Accuracy', marker='o', linewidth=2, color='blue', markersize=3)
    plt.plot(history['val_acc'], label='Validation Accuracy', marker='s', linewidth=2, color='orange', markersize=3)
    plt.title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. F1 Score (subplot 3)
    plt.subplot(2, 3, 3)
    plt.plot(history['val_f1'], label='Weighted F1', marker='o', linewidth=2, color='blue', markersize=3)
    plt.plot(history['val_macro_f1'], label='Macro F1', marker='s', linewidth=2, color='orange', markersize=3)
    plt.title('F1 Score Evolution', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Precision and Recall (subplot 4)
    plt.subplot(2, 3, 4)
    plt.plot(history['val_precision'], label='Precision', marker='o', linewidth=2, color='blue', markersize=3)
    plt.plot(history['val_recall'], label='Recall', marker='s', linewidth=2, color='orange', markersize=3)
    plt.title('Precision and Recall Evolution', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Learning Rate (subplot 5)
    plt.subplot(2, 3, 5)
    plt.plot(history['learning_rates'], marker='o', linewidth=2, color='red', markersize=3)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # 6. Overfitting Monitor (subplot 6)
    plt.subplot(2, 3, 6)
    acc_gap = [train - val for train, val in zip(history['train_acc'], history['val_acc'])]
    plt.plot(acc_gap, marker='o', linewidth=2, color='orange', markersize=3)
    plt.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='Overfitting Alert (5%)')
    plt.title('Overfitting Monitor', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Gap (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    # Save the file
    output_file = 'combined_training_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"SUCCESS: Created {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024:.1f} KB")
    
    return output_file

if __name__ == "__main__":
    print("=" * 60)
    print("SIMPLE COMBINED TRAINING CURVES GENERATOR")
    print("=" * 60)
    
    try:
        output_file = create_combined_training_curves()
        print("\n" + "=" * 60)
        print("COMBINED TRAINING CURVES GENERATED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Output file: {output_file}")
        print("The file contains all 6 training graphs in one picture:")
        print("  Row 1: Loss, Accuracy, F1 Score")
        print("  Row 2: Precision/Recall, Learning Rate, Overfitting Monitor")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()