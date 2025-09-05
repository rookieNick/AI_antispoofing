import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import numpy as np
from model_cnn import OptimizedCNN
from plot_utils import (MetricsLogger, create_results_folder, get_next_index, plot_confusion_matrix,
                       save_metrics_summary, plot_roc_curve, plot_comprehensive_dashboard,
                       plot_performance_analysis, plot_mse_rmse_dashboard, calculate_eer_hter,
                       plot_calibration_curve, plot_precision_recall_curve, plot_eer_hter_analysis) # Added new plot imports
from datetime import datetime
import matplotlib.pyplot as plt

# ======================== CONFIGURATION VARIABLES ========================
# Test Parameters
BATCH_SIZE = 128              # Batch size for testing
IMAGE_SIZE = (112, 112)       # Input image size (height, width)
SAMPLE_LIMIT = -1             # Limit test dataset to this many samples; set to -1 to use all samples

# Data Loading
NUM_WORKERS = 10               # Number of data loading workers
PIN_MEMORY = True             # Pin memory for faster GPU transfer

# Model Loading
MODEL_FILENAME = 'cnn_pytorch1.pth'

# Progress Reporting
PROGRESS_REPORT_INTERVAL = 10 # Report progress every N batches

# Class Information
CLASS_NAMES = ['live', 'spoof']  # Known classes for this dataset

# ======================== END CONFIGURATION ========================

# --- GPU Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Define dataset paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.normpath(os.path.join(script_dir, "..", "dataset", "casia-fasd", "test"))

def test_model(model_path=None):
    # Use default model path if not provided
    if model_path is None:
        model_dir = os.path.join(script_dir, "model")
        model_path = os.path.join(model_dir, MODEL_FILENAME)
    
    print(f"Test directory: {test_dir}")
    print(f"Model path: {model_path}")
    print(f"Using batch size: {BATCH_SIZE}")
    print(f"Using image size: {IMAGE_SIZE}")
    
    # Test transforms (no augmentation for testing)
    test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    # If SAMPLE_LIMIT > 0, limit test dataset; if -1, use full dataset
    if SAMPLE_LIMIT > 0:
        test_indices = torch.randperm(len(test_dataset))[:SAMPLE_LIMIT]
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    
    # Create test data loader
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    # Get class info from the dataset
    num_classes = len(CLASS_NAMES)
    print(f"Class names: {CLASS_NAMES}")
    print(f"Number of classes: {num_classes}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    model = OptimizedCNN(num_classes=num_classes).to(device)
    # Load trained model weights
    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")
    else:
        print(f"Model file not found: {model_path}")
        print("Please run training first or provide a valid model path.")
        return
    # Only use torch.compile if GPU supports CUDA capability >= 7.0
    use_compile = False
    if torch.cuda.is_available():
        cap_major, cap_minor = torch.cuda.get_device_capability()
        if cap_major >= 7:
            use_compile = True
    if use_compile:
        try:
            model = torch.compile(model)
            print("Model compiled for faster inference")
        except Exception as e:
            print(f"torch.compile failed: {e}\nUsing regular model.")
    else:
        print("torch.compile not used: GPU capability too old or not available.")
    
    # Loss function for evaluation
    criterion = nn.CrossEntropyLoss()
    
    # Gradient scaler for mixed precision (if using CUDA)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    print("\nEvaluating model on test set...")
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    test_tp = test_tn = test_fp = test_fn = 0
    
    # For storing predictions and true labels
    all_predictions = []
    all_targets = []
    # For ROC curve - collect prediction scores (probabilities)
    all_y_scores = []
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # Use mixed precision for test evaluation
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(data)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(data)
                loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
            # Store predictions and targets for detailed analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Store prediction scores for ROC curve (softmax probabilities for class 1)
            probabilities = torch.softmax(outputs, dim=1)
            all_y_scores.extend(probabilities[:, 1].cpu().numpy())  # Probability of class 1 (spoof)
            
            # Calculate precision/recall metrics
            for t, p in zip(targets.view(-1), predicted.view(-1)):
                if t.long() == 1 and p.long() == 1:
                    test_tp += 1
                elif t.long() == 1 and p.long() == 0:
                    test_fn += 1
                elif t.long() == 0 and p.long() == 1:
                    test_fp += 1
                else:
                    test_tn += 1
            
            # Print progress
            if batch_idx % PROGRESS_REPORT_INTERVAL == 0:
                print(f"Batch {batch_idx}/{len(test_loader)} processed...")
    
    # Calculate final metrics
    test_acc = 100.0 * test_correct / test_total
    test_loss /= len(test_loader)
    test_precision = test_tp / (test_tp + test_fp) if (test_tp + test_fp) > 0 else 0
    test_recall = test_tp / (test_tp + test_fn) if (test_tp + test_fn) > 0 else 0
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0
    test_specificity = test_tn / (test_tn + test_fp) if (test_tn + test_fp) > 0 else 0
    
    # Print detailed results
    print(f"\n=== Test Results ===")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"Test Specificity: {test_specificity:.4f}")
    
    print(f"\n=== Confusion Matrix (Raw Counts) ===")
    print(f"True Positives (TP): {test_tp}")
    print(f"True Negatives (TN): {test_tn}")
    print(f"False Positives (FP): {test_fp}")
    print(f"False Negatives (FN): {test_fn}")
    
    print(f"\n=== Class-wise Accuracy ===")
    # Calculate class-wise accuracy percentages
    live_total = test_tn + test_fp  # Total actual live samples
    spoof_total = test_fn + test_tp  # Total actual spoof samples
    live_accuracy = (test_tn / live_total * 100) if live_total > 0 else 0
    spoof_accuracy = (test_tp / spoof_total * 100) if spoof_total > 0 else 0
    
    print(f"Live Detection Accuracy: {live_accuracy:.1f}% ({test_tn} correct out of {live_total} live samples)")
    print(f"Spoof Detection Accuracy: {spoof_accuracy:.1f}% ({test_tp} correct out of {spoof_total} spoof samples)")
    print(f"Overall Accuracy: {test_acc:.1f}% ({test_correct} correct out of {test_total} total samples)")
    
    print(f"\n=== Class-wise Performance ===")
    print(f"Live Detection (Class 0):")
    print(f"  - Sensitivity (Recall): {test_tn/(test_tn + test_fp):.4f}")
    print(f"  - Specificity: {test_specificity:.4f}")
    print(f"Spoof Detection (Class 1):")
    print(f"  - Sensitivity (Recall): {test_recall:.4f}")
    print(f"  - Precision: {test_precision:.4f}")
    
    # Calculate EER and HTER
    eer, hter = calculate_eer_hter(all_targets, all_y_scores)
    
    # Determine correct predictions for confidence distribution
    correct_predictions = (np.array(all_predictions) == np.array(all_targets)).tolist()

    results = {
        'model_name': 'CNN', # Added model name
        'accuracy': test_acc,
        'loss': test_loss,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'specificity': test_specificity,
        'confusion_matrix': {
            'tp': test_tp, 'tn': test_tn,
            'fp': test_fp, 'fn': test_fn
        },
        'y_true': all_targets,
        'y_scores': all_y_scores,
        'eer': eer,
        'hter': hter,
        'mse': np.mean((np.array(all_targets) - np.array(all_y_scores)) ** 2),
        'rmse': np.sqrt(np.mean((np.array(all_targets) - np.array(all_y_scores)) ** 2)),
        'correct_predictions': correct_predictions # Added correct predictions
    }
    
    print(f"\nAdditional Metrics:")
    print(f"  - EER (Equal Error Rate): {eer:.2f}%")
    print(f"  - HTER (Half Total Error Rate): {hter:.2f}%")
    print(f"  - MSE (Mean Squared Error): {results['mse']:.6f}")
    print(f"  - RMSE (Root Mean Squared Error): {results['rmse']:.6f}")
    
    # Save test results plots
    print("\nSaving comprehensive test results...")
    results_dir = create_results_folder(folder_type='cnn') # Changed folder_type to 'cnn'
    index = get_next_index(results_dir, "test")
    date_str = datetime.now().strftime('%Y%m%d')
    base_name = f"test_cnn_{index}_{date_str}" # Changed base_name for clarity
    # Create a dedicated folder for this test result set
    result_folder = os.path.join(results_dir, f"{base_name}")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    # Save original plots
    cm_path = os.path.join(result_folder, "confusion_matrix.png")
    plot_confusion_matrix(results['confusion_matrix'], CLASS_NAMES, cm_path)
    print(f"✅ Confusion matrix saved: {cm_path}")
    
    roc_path = os.path.join(result_folder, "roc_curve.png")
    roc_auc = plot_roc_curve(results['y_true'], results['y_scores'], roc_path)
    print(f"✅ ROC curve saved: {roc_path} (AUC: {roc_auc:.4f})")
    
    # Add new plots
    calibration_path = os.path.join(result_folder, "calibration_curve.png")
    plot_calibration_curve(results['y_true'], results['y_scores'], calibration_path, results['model_name'])
    
    pr_curve_path = os.path.join(result_folder, "precision_recall_curve.png")
    plot_precision_recall_curve(results['y_true'], results['y_scores'], pr_curve_path, results['model_name'])

    summary_path = os.path.join(result_folder, "summary.txt")
    save_metrics_summary(results, summary_path)
    print(f"✅ Test summary saved: {summary_path}")
    
    # Save comprehensive analysis dashboards
    print("\n🔥 Generating comprehensive analysis dashboards...")
    save_prefix = os.path.join(result_folder, "advanced")
    
    plot_comprehensive_dashboard(results, save_prefix)
    print(f"✅ EfficientNet+Meta Learning style dashboard saved")
    
    plot_performance_analysis(results, save_prefix)
    print(f"✅ ViT-style performance analysis saved")
    
    plot_mse_rmse_dashboard(results, save_prefix)
    print(f"✅ MSE & RMSE analysis dashboard saved")
    
    # Add EER/HTER analysis
    eer_hter_path = os.path.join(result_folder, "eer_hter_analysis.png")
    plot_eer_hter_analysis(eer, hter, eer_hter_path, results['model_name'])
    
    print(f"\n🎯 All comprehensive test results saved in folder: {result_folder}")
    print(f"📊 Generated {len(os.listdir(result_folder))} analysis files including:")
    print(f"   - Traditional confusion matrix and ROC curves")
    print(f"   - EfficientNet+Meta Learning comprehensive dashboard")
    print(f"   - Performance analysis with MSE/RMSE focus")
    print(f"   - Detailed MSE & RMSE analysis dashboard")
    print(f"   - EER/HTER quality analysis with gauges")
    print(f"   - EER: {eer:.2f}%, HTER: {hter:.2f}%")
    
    return results

if __name__ == '__main__':
    # Enable CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    print("PyTorch version:", torch.__version__)
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # You can also provide a custom model path
    # test_model("path/to/your/model.pth")
    results = test_model()
    print(f"\nTest completed successfully!")
