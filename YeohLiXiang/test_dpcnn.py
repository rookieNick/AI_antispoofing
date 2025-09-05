# ==============================================================================
# DPCNN Testing Script for Face Anti-Spoofing
# ==============================================================================
# This script tests DPCNN models on the test dataset and generates
# comprehensive evaluation metrics and visualizations.
# ==============================================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import numpy as np
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from model_dpcnn import create_dpcnn_model  # DPCNN models
from plot_utils import (MetricsLogger, create_results_folder, get_next_index, plot_confusion_matrix, 
                       save_metrics_summary, plot_roc_curve, plot_comprehensive_dashboard, 
                       plot_vit_performance_analysis, plot_mse_rmse_dashboard, calculate_eer_hter)

# ======================== CONFIGURATION VARIABLES ========================

# Test Parameters
BATCH_SIZE = 32               # Batch size for testing
IMAGE_SIZE = (112, 112)       # Input image dimensions
MODEL_TYPE = 'standard'       # 'standard' or 'lightweight' - must match training
SAMPLE_SIZE = -1              # Number of samples to test (-1 means full dataset)

# Model Configuration
MODEL_FILENAME = f'dpcnn.pth'

# Dataset Configuration
CLASS_NAMES = ['live', 'spoof']  # Class names for confusion matrix

# ======================== END CONFIGURATION ========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset paths
script_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.normpath(os.path.join(script_dir, "..", "dataset", "casia-fasd", "test"))
model_path = os.path.join(script_dir, 'model', MODEL_FILENAME)

def test_dpcnn_model():
    """
    Main testing function for DPCNN models
    """
    print("=" * 80)
    print(f"TESTING {MODEL_TYPE.upper()} DPCNN FOR FACE ANTI-SPOOFING")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model Type: {MODEL_TYPE}")
    print(f"Input Size: {IMAGE_SIZE}")
    print(f"Test Dataset: {test_dir}")
    print(f"Model Path: {model_path}")
    print("=" * 80)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please train the model first using train_dpcnn.py")
        return
    
    # --- Data Loading and Preprocessing ---
    print("\\n📂 Loading test dataset...")
    
    # Test data transforms (no augmentation, same as validation)
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    # Optionally limit the number of test samples using SAMPLE_SIZE
    original_test_len = len(test_dataset)
    if SAMPLE_SIZE != -1 and SAMPLE_SIZE is not None:
        if SAMPLE_SIZE <= 0:
            print(f"⚠️  SAMPLE_SIZE set to {SAMPLE_SIZE}. Using full dataset instead.")
            sample_size = original_test_len
        else:
            sample_size = min(SAMPLE_SIZE, original_test_len)
        print(f"ℹ️  Using sample_size={sample_size} of {original_test_len} test samples")
        indices = list(range(sample_size))
        test_dataset = Subset(test_dataset, indices)

    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Don't shuffle for consistent results
        num_workers=4,
        pin_memory=True
    )
    
    num_classes = len(CLASS_NAMES)
    
    print(f"✅ Test dataset loaded successfully!")
    print(f"   Classes: {CLASS_NAMES}")
    print(f"   Test samples: {len(test_dataset):,}")
    print(f"   Number of batches: {len(test_loader)}")
    
    # --- Model Setup ---
    print(f"\\n🏗️  Loading {MODEL_TYPE} DPCNN model...")
    
    model = create_dpcnn_model(
        model_type=MODEL_TYPE,
        num_classes=num_classes,
        dropout_rate=0.5  # Will be ignored in eval mode
    ).to(device)
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model.eval()
        print(f"✅ Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    
    # --- Testing Phase ---
    print("\\n🧪 Starting model evaluation...")
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    test_loss = 0.0  # Initialize test_loss
    test_correct = 0
    test_total = 0
    
    # For calculating per-class metrics
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    # Define criterion for test loss calculation
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(data)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Calculate accuracy
            test_total += targets.size(0)
            test_correct += (predicted == targets).sum().item()
            
            # Per-class accuracy
            for i in range(targets.size(0)):
                label = targets[i]
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
            
            # Progress update
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx+1}/{len(test_loader)}")
    
    # Convert to numpy arrays for metrics calculation
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate overall accuracy and loss
    test_accuracy = 100.0 * test_correct / test_total
    test_loss /= len(test_loader)  # Average the test loss
    
    # Calculate per-class accuracy
    per_class_acc = []
    for i in range(num_classes):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            per_class_acc.append(acc)
            print(f"   {CLASS_NAMES[i]} accuracy: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
        else:
            per_class_acc.append(0.0)
    
    # Calculate detailed metrics
    test_precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
    test_recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
    test_f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
    
    # Calculate per-class precision, recall, f1
    test_precision_per_class = precision_score(all_targets, all_predictions, average=None, zero_division=0)
    test_recall_per_class = recall_score(all_targets, all_predictions, average=None, zero_division=0)
    test_f1_per_class = f1_score(all_targets, all_predictions, average=None, zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Calculate specificity (for binary classification)
    if num_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    else:
        specificity = 0
        sensitivity = test_recall
    
    # Calculate ROC curve and AUC (for binary classification)
    if num_classes == 2:
        # Use probabilities for positive class (class 1)
        y_scores = all_probabilities[:, 1]
        fpr, tpr, _ = roc_curve(all_targets, y_scores)
        roc_auc = auc(fpr, tpr)
    else:
        roc_auc = 0
    
    # --- Print Results ---
    print("\\n" + "=" * 80)
    print("📊 TEST RESULTS")
    print("=" * 80)
    print(f"Overall Test Accuracy:    {test_accuracy:.2f}%")
    print(f"Overall Test Precision:   {test_precision:.4f}")
    print(f"Overall Test Recall:      {test_recall:.4f}")
    print(f"Overall Test F1-Score:    {test_f1:.4f}")
    if num_classes == 2:
        print(f"Specificity:              {specificity:.4f}")
        print(f"Sensitivity:              {sensitivity:.4f}")
        print(f"ROC AUC:                  {roc_auc:.4f}")
    
    print("\\nPer-Class Metrics:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name}:")
        print(f"    Accuracy:   {per_class_acc[i]:.2f}%")
        print(f"    Precision:  {test_precision_per_class[i]:.4f}")
        print(f"    Recall:     {test_recall_per_class[i]:.4f}")
        print(f"    F1-Score:   {test_f1_per_class[i]:.4f}")
    
    print(f"\\nConfusion Matrix:")
    print(cm)
    
    # Calculate EER and HTER
    y_scores_binary = all_probabilities[:, 1] if num_classes == 2 else all_probabilities[:, 0]
    eer, hter = calculate_eer_hter(all_targets, y_scores_binary)
    
    # Store results for saving
    results = {
        'accuracy': test_accuracy,
        'loss': test_loss,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'specificity': specificity,
        'confusion_matrix': {
            'tp': int(tp) if num_classes == 2 else 0,
            'tn': int(tn) if num_classes == 2 else 0,
            'fp': int(fp) if num_classes == 2 else 0,
            'fn': int(fn) if num_classes == 2 else 0
        },
        'y_true': all_targets,
        'y_scores': y_scores_binary,
        'eer': eer,
        'hter': hter,
        'mse': np.mean((np.array(all_targets) - np.array(y_scores_binary)) ** 2),
        'rmse': np.sqrt(np.mean((np.array(all_targets) - np.array(y_scores_binary)) ** 2))
    }
    
    print(f"\nAdditional Metrics:")
    print(f"  - EER (Equal Error Rate): {eer:.2f}%")
    print(f"  - HTER (Half Total Error Rate): {hter:.2f}%")
    print(f"  - MSE (Mean Squared Error): {results['mse']:.6f}")
    print(f"  - RMSE (Root Mean Squared Error): {results['rmse']:.6f}")
    
    # Save test results plots
    print("\nSaving comprehensive test results...")
    results_dir = create_results_folder(folder_type='dpcnn')
    index = get_next_index(results_dir, "test")
    date_str = datetime.now().strftime('%Y%m%d')
    base_name = f"test_{MODEL_TYPE}_dpcnn_{index}_{date_str}"
    
    # Create a dedicated folder for this test result set
    result_folder = os.path.join(results_dir, f"{base_name}")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    # Save original plots
    cm_path = os.path.join(result_folder, "confusion_matrix.png")
    plot_confusion_matrix(results['confusion_matrix'], CLASS_NAMES, cm_path)
    print(f"✅ Confusion matrix saved: {cm_path}")
    
    # Save ROC curve (for binary classification)
    if num_classes == 2:
        roc_path = os.path.join(result_folder, "roc_curve.png")
        roc_auc = plot_roc_curve(results['y_true'], results['y_scores'], roc_path)
        print(f"✅ ROC curve saved: {roc_path} (AUC: {roc_auc:.4f})")
    
    summary_path = os.path.join(result_folder, "summary.txt")
    save_metrics_summary(results, summary_path)
    print(f"✅ Metrics summary saved: {summary_path}")
    
    # Save comprehensive analysis dashboards
    print("\n🔥 Generating comprehensive analysis dashboards...")
    save_prefix = os.path.join(result_folder, "advanced")
    
    plot_comprehensive_dashboard(results, save_prefix)
    print(f"✅ EfficientNet+Meta Learning style dashboard saved")
    
    plot_vit_performance_analysis(results, save_prefix)
    print(f"✅ ViT-style performance analysis saved")
    
    plot_mse_rmse_dashboard(results, save_prefix)
    print(f"✅ MSE & RMSE analysis dashboard saved")
    
    print(f"\n🎯 All comprehensive test results saved in folder: {result_folder}")
    print(f"📊 Generated {len(os.listdir(result_folder))} analysis files including:")
    print(f"   - Traditional confusion matrix and ROC curves")
    print(f"   - EfficientNet+Meta Learning comprehensive dashboard")
    print(f"   - ViT-style performance analysis with MSE/RMSE focus")
    print(f"   - Detailed MSE & RMSE analysis dashboard")
    print(f"   - EER: {eer:.2f}%, HTER: {hter:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    try:
        test_dpcnn_model()
    except Exception as e:
        print(f"\\n❌ Testing failed with error: {e}")
        raise
