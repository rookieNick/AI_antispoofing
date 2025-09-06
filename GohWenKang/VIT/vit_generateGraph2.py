# ==============================================================================
# Enhanced ViT Model Evaluation with Comprehensive Graphs
# ==============================================================================
# This script includes all requested metrics and visualizations:
# - Validation Precision, Recall, F1 Score
# - Confusion Matrix in Percentage
# - Confidence Score Distribution
# - Residual Analysis
# ==============================================================================

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    precision_recall_fscore_support, mean_squared_error,
    precision_score, recall_score, f1_score
)
from tqdm import tqdm
import pandas as pd
from scipy import stats

# Apply PyTorch 2.6 fix
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

def inspect_model_file(model_path):
    """Inspect what's inside the model file to understand its structure."""
    print(f"Inspecting model file: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        return None
    
    try:
        # Try loading with weights_only=False to see the structure
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict):
            print("Model file contains a dictionary with keys:")
            for key in checkpoint.keys():
                print(f"  - {key}: {type(checkpoint[key])}")
            return checkpoint
        else:
            print(f"Model file contains: {type(checkpoint)}")
            return checkpoint
            
    except Exception as e:
        print(f"Error loading model file: {e}")
        return None

def find_available_models():
    """Find all available model files in common directories."""
    search_dirs = ['vit_models', '.', 'models', 'checkpoints']
    model_files = []
    
    for dir_name in search_dirs:
        if os.path.exists(dir_name):
            for file in os.listdir(dir_name):
                if file.endswith('.pth'):
                    full_path = os.path.join(dir_name, file)
                    model_files.append(full_path)
    
    return model_files

def load_model_correctly():
    """Load the model with the correct method based on file inspection."""
    
    # First, find available model files
    print("Searching for model files...")
    model_files = find_available_models()
    
    if not model_files:
        print("No .pth files found. Looking for 'best_vit_model.pth' specifically...")
        if not os.path.exists('best_vit_model.pth'):
            print("ERROR: 'best_vit_model.pth' not found in current directory")
            print("Please ensure the model file exists in the current directory")
            return None, None
        else:
            model_files = ['best_vit_model.pth']
    
    print(f"Found {len(model_files)} model file(s):")
    for i, file in enumerate(model_files):
        file_size = os.path.getsize(file) / (1024*1024)  # MB
        print(f"  {i+1}. {file} ({file_size:.1f} MB)")
    
    # Use best_vit_model.pth if available, otherwise use first found
    if 'best_vit_model.pth' in model_files:
        chosen_model = 'best_vit_model.pth'
        print(f"\n✓ Using: {chosen_model}")
    elif any('best_vit_model.pth' in f for f in model_files):
        chosen_model = [f for f in model_files if 'best_vit_model.pth' in f][0]
        print(f"\n✓ Using: {chosen_model}")
    else:
        chosen_model = model_files[0]
        print(f"\n⚠️  'best_vit_model.pth' not found, using: {chosen_model}")
    
    # Inspect the model file
    checkpoint_data = inspect_model_file(chosen_model)
    if checkpoint_data is None:
        return None, None
    
    # Import required modules
    from transformers import ViTForImageClassification
    
    # Initialize the model
    print("\nInitializing ViT model...")
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=2,
        ignore_mismatched_sizes=True,
        cache_dir='./vit_cache'
    )
    
    # Load the weights based on what we found
    try:
        if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
            print("Loading from checkpoint format (with model_state_dict key)")
            model.load_state_dict(checkpoint_data['model_state_dict'])
            
            # Show additional checkpoint info
            if 'epoch' in checkpoint_data:
                print(f"Checkpoint from epoch: {checkpoint_data['epoch']}")
            if 'best_val_acc' in checkpoint_data:
                print(f"Best validation accuracy: {checkpoint_data['best_val_acc']:.2f}%")
                
        elif isinstance(checkpoint_data, dict):
            print("Loading state dict directly (dictionary format)")
            model.load_state_dict(checkpoint_data)
        else:
            print("Unknown model format - attempting direct load")
            model.load_state_dict(checkpoint_data)
        
        print("✓ Model loaded successfully!")
        return model, chosen_model
        
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("\nTrying alternative loading methods...")
        
        # Try loading with strict=False (allows missing/unexpected keys)
        try:
            if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
                missing_keys, unexpected_keys = model.load_state_dict(
                    checkpoint_data['model_state_dict'], strict=False
                )
            else:
                missing_keys, unexpected_keys = model.load_state_dict(
                    checkpoint_data, strict=False
                )
            
            print(f"✓ Loaded with strict=False")
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
            
            return model, chosen_model
            
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
            return None, None

class CASIAFASDDataset:
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        
        # Load images
        for class_name, label in [('live', 1), ('spoof', 0)]:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                files = [f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for f in files:
                    self.images.append(os.path.join(class_dir, f))
                    self.labels.append(label)
        
        print(f"Loaded {len(self.images)} test images")
        print(f"Live: {sum(self.labels)}, Spoof: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        from PIL import Image
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except:
            # Fallback to dummy image
            if self.transform:
                return self.transform(Image.new('RGB', (224, 224))), label
            return Image.new('RGB', (224, 224)), label

def plot_comprehensive_results(all_targets, all_predictions, all_probs, model_path):
    """Create comprehensive visualization plots."""
    
    # Setup the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Confusion Matrix in Percentage
    plt.subplot(3, 3, 1)
    cm = confusion_matrix(all_targets, all_predictions)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues',
               xticklabels=['Spoof', 'Live'], yticklabels=['Spoof', 'Live'])
    plt.title('Confusion Matrix (Percentage)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # 2. Validation Precision, Recall, F1 Score
    plt.subplot(3, 3, 2)
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    metrics = ['Precision', 'Recall', 'F1-Score']
    values = [precision, recall, f1]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    plt.ylim(0, 1)
    plt.title('Validation Metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Score')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Confidence Score Distribution
    plt.subplot(3, 3, 3)
    probs_array = np.array(all_probs)
    live_probs = probs_array[:, 1]  # Confidence for 'live' class
    spoof_probs = probs_array[:, 0]  # Confidence for 'spoof' class
    
    # Separate by true labels
    live_indices = np.array(all_targets) == 1
    spoof_indices = np.array(all_targets) == 0
    
    plt.hist(live_probs[live_indices], bins=30, alpha=0.7, label='Live (True)', 
             color='green', density=True)
    plt.hist(live_probs[spoof_indices], bins=30, alpha=0.7, label='Live (Spoof)', 
             color='red', density=True)
    
    plt.xlabel('Confidence Score (Live Class)')
    plt.ylabel('Density')
    plt.title('Confidence Score Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Residual Analysis - Prediction vs True Label
    plt.subplot(3, 3, 4)
    residuals = np.array(all_targets) - live_probs
    
    plt.scatter(live_probs, residuals, alpha=0.6, c=all_targets, cmap='RdYlBu')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    plt.xlabel('Predicted Probability (Live)')
    plt.ylabel('Residuals (True - Predicted)')
    plt.title('Residual Analysis', fontsize=14, fontweight='bold')
    plt.colorbar(label='True Label')
    plt.grid(True, alpha=0.3)
    
    # 5. ROC-like Analysis (Confidence vs Accuracy)
    plt.subplot(3, 3, 5)
    confidence_bins = np.linspace(0, 1, 21)
    accuracies = []
    bin_centers = []
    
    for i in range(len(confidence_bins)-1):
        lower = confidence_bins[i]
        upper = confidence_bins[i+1]
        mask = (live_probs >= lower) & (live_probs < upper)
        
        if np.sum(mask) > 0:
            bin_accuracy = accuracy_score(
                np.array(all_targets)[mask], 
                np.array(all_predictions)[mask]
            )
            accuracies.append(bin_accuracy)
            bin_centers.append((lower + upper) / 2)
    
    plt.plot(bin_centers, accuracies, 'o-', linewidth=2, markersize=6)
    plt.xlabel('Confidence Score (Live)')
    plt.ylabel('Accuracy')
    plt.title('Confidence vs Accuracy', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # 6. Class-wise Performance
    plt.subplot(3, 3, 6)
    class_report = classification_report(all_targets, all_predictions, 
                                       target_names=['Spoof', 'Live'], 
                                       output_dict=True)
    
    classes = ['Spoof', 'Live']
    precision_vals = [class_report[cls]['precision'] for cls in classes]
    recall_vals = [class_report[cls]['recall'] for cls in classes]
    f1_vals = [class_report[cls]['f1-score'] for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    plt.bar(x - width, precision_vals, width, label='Precision', alpha=0.8)
    plt.bar(x, recall_vals, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1_vals, width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Class-wise Performance', fontsize=14, fontweight='bold')
    plt.xticks(x, classes)
    plt.legend()
    plt.ylim(0, 1)
    
    # 7. Error Analysis
    plt.subplot(3, 3, 7)
    errors = np.array(all_targets) != np.array(all_predictions)
    error_confidences = live_probs[errors]
    correct_confidences = live_probs[~errors]
    
    plt.hist(correct_confidences, bins=20, alpha=0.7, label='Correct', 
             color='green', density=True)
    plt.hist(error_confidences, bins=20, alpha=0.7, label='Incorrect', 
             color='red', density=True)
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    plt.title('Error Analysis', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Prediction Probability Scatter
    plt.subplot(3, 3, 8)
    colors = ['red' if pred != true else 'green' 
              for pred, true in zip(all_predictions, all_targets)]
    
    plt.scatter(range(len(live_probs)), live_probs, c=colors, alpha=0.6, s=20)
    plt.axhline(y=0.5, color='blue', linestyle='--', alpha=0.8, label='Decision Threshold')
    plt.xlabel('Sample Index')
    plt.ylabel('Predicted Probability (Live)')
    plt.title('Prediction Probabilities', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Statistical Summary
    plt.subplot(3, 3, 9)
    plt.axis('off')
    
    # Calculate additional metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    mse = mean_squared_error(all_targets, live_probs)
    rmse = np.sqrt(mse)
    
    # Statistical tests
    live_conf_true = live_probs[live_indices]
    live_conf_false = live_probs[spoof_indices]
    t_stat, p_value = stats.ttest_ind(live_conf_true, live_conf_false)
    
    summary_text = f"""
    STATISTICAL SUMMARY
    {'='*30}
    
    Overall Accuracy: {accuracy:.3f}
    Precision (Weighted): {precision:.3f}
    Recall (Weighted): {recall:.3f}
    F1-Score (Weighted): {f1:.3f}
    
    MSE: {mse:.4f}
    RMSE: {rmse:.4f}
    
    Confidence Statistics:
    - Live (True): μ={np.mean(live_conf_true):.3f}, σ={np.std(live_conf_true):.3f}
    - Live (Spoof): μ={np.mean(live_conf_false):.3f}, σ={np.std(live_conf_false):.3f}
    
    T-test (p-value): {p_value:.2e}
    
    Total Samples: {len(all_targets)}
    True Live: {np.sum(all_targets)}
    True Spoof: {len(all_targets) - np.sum(all_targets)}
    """
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    plt.savefig('comprehensive_evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION RESULTS SAVED")
    print("="*80)
    print(f"Saved to: comprehensive_evaluation_results.png")

def run_comprehensive_evaluation(model, model_path):
    """Run the comprehensive evaluation with all requested metrics."""
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import torch.nn.functional as F
    
    # Configuration
    TEST_DATA_DIRS = [
        "CASIA-FASD/test",
        "dataset/casia-fasd/test", 
        "test",
        "../dataset/casia-fasd/test"
    ]
    
    # Find test data directory
    test_dir = None
    for dir_path in TEST_DATA_DIRS:
        if os.path.exists(dir_path):
            test_dir = dir_path
            break
    
    if not test_dir:
        print("Test data directory not found. Please specify the correct path:")
        for dir_path in TEST_DATA_DIRS:
            print(f"  Tried: {dir_path}")
        return
    
    print(f"Using test directory: {test_dir}")
    
    # Setup data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        dataset = CASIAFASDDataset(test_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return
    
    # Run testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probs = []
    
    print("Running comprehensive evaluation...")
    with torch.no_grad():
        for data, targets in tqdm(dataloader, desc="Evaluating"):
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            _, predicted = logits.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Generate comprehensive plots and analysis
    plot_comprehensive_results(all_targets, all_predictions, all_probs, model_path)
    
    # Print detailed results
    print("\n" + "="*80)
    print("DETAILED EVALUATION RESULTS")
    print("="*80)
    
    # Calculate all metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    probs_array = np.array(all_probs)
    live_probs = probs_array[:, 1]
    mse = mean_squared_error(all_targets, live_probs)
    rmse = np.sqrt(mse)
    
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print("-" * 50)
    report = classification_report(all_targets, all_predictions, 
                                 target_names=['Spoof', 'Live'])
    print(report)
    
    print("\nEvaluation completed successfully!")
    print("Check 'comprehensive_evaluation_results.png' for detailed visualizations.")

def main():
    """Main function to run comprehensive evaluation."""
    print("="*80)
    print("COMPREHENSIVE ViT MODEL EVALUATION")
    print("="*80)
    
    # Step 1: Load the model
    model, model_path = load_model_correctly()
    
    if model is None:
        print("Failed to load model. Please check the error messages above.")
        return
    
    # Step 2: Run comprehensive evaluation
    print("\nStarting comprehensive evaluation with separate graph generation...")
    run_comprehensive_evaluation(model, model_path)

if __name__ == "__main__":
    main()