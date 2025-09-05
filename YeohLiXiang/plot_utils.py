import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import roc_curve, auc, mean_squared_error, precision_recall_curve, average_precision_score # Added for PR curve
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from scipy import stats

def create_results_folder(folder_type=None):
    """Create results folder for patch_cnn, vgg16, dpcnn, or cnn if it doesn't exist"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if folder_type == 'patch_cnn':
        results_dir = os.path.join(script_dir, "results_patch_cnn")
    elif folder_type == 'vgg16':
        results_dir = os.path.join(script_dir, "results_vgg16")
    elif folder_type == 'dpcnn':
        results_dir = os.path.join(script_dir, "results_dpcnn")
    elif folder_type == 'cnn':
        results_dir = os.path.join(script_dir, "results_cnn")
    else:
        results_dir = os.path.join(script_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def get_next_index(results_dir, prefix="result"):
    """Get the next available index for result files with specific prefix"""
    existing_files = [f for f in os.listdir(results_dir) if f.startswith(f'{prefix}_')]
    if not existing_files:
        return 1
    
    indices = []
    for f in existing_files:
        try:
            # Split by underscore and get the index part (second element)
            parts = f.split('_')
            if len(parts) >= 2:
                index = int(parts[1])
                indices.append(index)
        except (IndexError, ValueError):
            continue
    
    return max(indices) + 1 if indices else 1

def plot_training_metrics(train_losses, train_accs, val_losses, val_accs, 
                         val_precisions, val_recalls, save_path):
    """Plot training and validation metrics"""
    return plot_training_metrics_with_model_name(train_losses, train_accs, val_losses, val_accs, 
                                                val_precisions, val_recalls, save_path, "Model")

def plot_training_metrics_with_model_name(train_losses, train_accs, val_losses, val_accs, 
                         val_precisions, val_recalls, save_path, model_name="Model"):
    """Plot training and validation metrics with model name in title"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{model_name} Training Metrics', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Training and Validation Accuracy', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision plot
    axes[0, 2].plot(epochs, val_precisions, 'g-', label='Validation Precision', linewidth=2)
    axes[0, 2].set_title('Validation Precision', fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Recall plot
    axes[1, 0].plot(epochs, val_recalls, 'orange', label='Validation Recall', linewidth=2)
    axes[1, 0].set_title('Validation Recall', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1 Score plot
    f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(val_precisions, val_recalls)]
    axes[1, 1].plot(epochs, f1_scores, 'm-', label='Validation F1 Score', linewidth=2)
    axes[1, 1].set_title('Validation F1 Score', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Learning rate plot (if available)
    axes[1, 2].text(0.5, 0.5, f'Model Architecture:\n{model_name}\n\nLayers:\n- 3 Conv blocks\n- Global Avg Pool\n- 3 FC layers\n\nActivation: SiLU\nOptimizer: AdamW\nScheduler: OneCycleLR', 
                   ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    axes[1, 2].set_title('Model Info', fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_scores, save_path):
    """Plot ROC curve with False Positive Rate vs True Positive Rate"""
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier (AUC = 0.5)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curve - Anti-Spoofing Model', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add text with AUC value
    plt.text(0.6, 0.2, f'AUC = {roc_auc:.4f}', fontsize=14, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_auc

def plot_loss_curve(train_losses, val_losses, save_path):
    """Plot separate training vs validation loss curve"""
    return plot_loss_curve_with_model_name(train_losses, val_losses, save_path, "Model")

def plot_loss_curve_with_model_name(train_losses, val_losses, save_path, model_name="Model"):
    """Plot separate training vs validation loss curve with model name in title"""
    plt.figure(figsize=(12, 8))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    
    plt.title(f'{model_name}: Training vs Validation Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add min loss annotations
    min_train_loss = min(train_losses)
    min_val_loss = min(val_losses)
    min_train_epoch = train_losses.index(min_train_loss) + 1
    min_val_epoch = val_losses.index(min_val_loss) + 1
    
    plt.annotate(f'Min Train Loss: {min_train_loss:.4f}\nEpoch: {min_train_epoch}',
                xy=(min_train_epoch, min_train_loss), xytext=(min_train_epoch + 2, min_train_loss + 0.05),
                arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.annotate(f'Min Val Loss: {min_val_loss:.4f}\nEpoch: {min_val_epoch}',
                xy=(min_val_epoch, min_val_loss), xytext=(min_val_epoch + 2, min_val_loss + 0.05),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_curve(train_accs, val_accs, save_path):
    """Plot separate training vs validation accuracy curve"""
    return plot_accuracy_curve_with_model_name(train_accs, val_accs, save_path, "Model")

def plot_accuracy_curve_with_model_name(train_accs, val_accs, save_path, model_name="Model"):
    """Plot separate training vs validation accuracy curve with model name in title"""
    plt.figure(figsize=(12, 8))
    
    epochs = range(1, len(train_accs) + 1)
    
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
    
    plt.title(f'{model_name}: Training vs Validation Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add max accuracy annotations
    max_train_acc = max(train_accs)
    max_val_acc = max(val_accs)
    max_train_epoch = train_accs.index(max_train_acc) + 1
    max_val_epoch = val_accs.index(max_val_acc) + 1
    
    plt.annotate(f'Max Train Acc: {max_train_acc:.2f}%\nEpoch: {max_train_epoch}',
                xy=(max_train_epoch, max_train_acc), xytext=(max_train_epoch + 2, max_train_acc - 5),
                arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.annotate(f'Max Val Acc: {max_val_acc:.2f}%\nEpoch: {max_val_epoch}',
                xy=(max_val_epoch, max_val_acc), xytext=(max_val_epoch + 2, max_val_acc - 5),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(confusion_matrix, class_names, save_path):
    """Plot confusion matrix as heatmap with counts and class-wise accuracy percentages"""
    plt.figure(figsize=(12, 10))
    
    # Create labels
    labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    values = [confusion_matrix['tn'], confusion_matrix['fp'], 
              confusion_matrix['fn'], confusion_matrix['tp']]
    
    # Calculate total samples
    total_samples = sum(values)
    
    # Reshape for 2x2 matrix
    cm_matrix = np.array([[confusion_matrix['tn'], confusion_matrix['fp']],
                         [confusion_matrix['fn'], confusion_matrix['tp']]])
    
    # Calculate class-wise accuracy percentages
    # For actual live class (row 0): correct = TN, total = TN + FP
    live_total = confusion_matrix['tn'] + confusion_matrix['fp']
    live_accuracy = (confusion_matrix['tn'] / live_total * 100) if live_total > 0 else 0
    
    # For actual spoof class (row 1): correct = TP, total = FN + TP  
    spoof_total = confusion_matrix['fn'] + confusion_matrix['tp']
    spoof_accuracy = (confusion_matrix['tp'] / spoof_total * 100) if spoof_total > 0 else 0
    
    # Create custom annotations with count and class accuracy
    annotations = np.empty_like(cm_matrix, dtype=object)
    # Top row (Actual Live): TN and FP
    annotations[0, 0] = f'{cm_matrix[0, 0]:,}\n({live_accuracy:.1f}%)'  # TN - correct live prediction
    annotations[0, 1] = f'{cm_matrix[0, 1]:,}\n({100-live_accuracy:.1f}%)'  # FP - wrong live prediction
    # Bottom row (Actual Spoof): FN and TP  
    annotations[1, 0] = f'{cm_matrix[1, 0]:,}\n({100-spoof_accuracy:.1f}%)'  # FN - wrong spoof prediction
    annotations[1, 1] = f'{cm_matrix[1, 1]:,}\n({spoof_accuracy:.1f}%)'  # TP - correct spoof prediction
    
    # Create heatmap
    sns.heatmap(cm_matrix, annot=annotations, fmt='', cmap='Blues', 
                xticklabels=['Predicted Live', 'Predicted Spoof'],
                yticklabels=['Actual Live', 'Actual Spoof'],
                cbar_kws={'label': 'Count'},
                annot_kws={'fontsize': 12, 'ha': 'center', 'va': 'center'})
    
    plt.title('Confusion Matrix\n(Count and Class Accuracy %)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    
    # Add text annotations with summary
    overall_accuracy = (cm_matrix[0,0] + cm_matrix[1,1])/total_samples*100
    plt.text(0.5, -0.12, f'Total Samples: {total_samples:,}', 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=11, fontweight='bold')
    plt.text(0.5, -0.16, f'Overall Accuracy: {overall_accuracy:.1f}% | Live Accuracy: {live_accuracy:.1f}% | Spoof Accuracy: {spoof_accuracy:.1f}%', 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_metrics_summary(results, save_path):
    """Save detailed metrics summary to text file"""
    # Use .get with defaults to avoid KeyError when some metrics are missing
    accuracy = results.get('accuracy', None)
    loss = results.get('loss', None)
    precision = results.get('precision', None)
    recall = results.get('recall', None)
    f1_score = results.get('f1_score', None)
    specificity = results.get('specificity', None)
    cm = results.get('confusion_matrix', {})

    tp = cm.get('tp', 0)
    tn = cm.get('tn', 0)
    fp = cm.get('fp', 0)
    fn = cm.get('fn', 0)
    total_samples = tp + tn + fp + fn

    with open(save_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("         MODEL EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: OptimizedCNN\n")
        f.write("\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 30 + "\n")
        if accuracy is not None:
            f.write(f"Test Accuracy:    {accuracy:.2f}%\n")
        else:
            f.write("Test Accuracy:    N/A\n")

        if loss is not None:
            f.write(f"Test Loss:        {loss:.4f}\n")
        else:
            f.write("Test Loss:        N/A\n")

        if precision is not None:
            f.write(f"Test Precision:   {precision:.4f}\n")
        else:
            f.write("Test Precision:   N/A\n")

        if recall is not None:
            f.write(f"Test Recall:      {recall:.4f}\n")
        else:
            f.write("Test Recall:      N/A\n")

        if f1_score is not None:
            f.write(f"Test F1-Score:    {f1_score:.4f}\n")
        else:
            f.write("Test F1-Score:    N/A\n")

        if specificity is not None:
            f.write(f"Test Specificity: {specificity:.4f}\n")
        else:
            f.write("Test Specificity: N/A\n")

        f.write("\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write("-" * 30 + "\n")
        f.write(f"True Positives:   {tp}\n")
        f.write(f"True Negatives:   {tn}\n")
        f.write(f"False Positives:  {fp}\n")
        f.write(f"False Negatives:  {fn}\n")
        f.write(f"Total Samples:    {total_samples}\n")
        f.write("\n")
        
        f.write("CLASS-WISE PERFORMANCE:\n")
        f.write("-" * 30 + "\n")
        # Avoid ZeroDivisionError when computing sensitivities
        live_sensitivity = (tn / (tn + fp)) if (tn + fp) > 0 else None
        if live_sensitivity is not None:
            f.write(f"Live Detection (Class 0):\n")
            f.write(f"  - Sensitivity: {live_sensitivity:.4f}\n")
        else:
            f.write("Live Detection (Class 0):\n")
            f.write("  - Sensitivity: N/A\n")

        if specificity is not None:
            f.write(f"  - Specificity: {specificity:.4f}\n")
        else:
            f.write("  - Specificity: N/A\n")

        f.write(f"Spoof Detection (Class 1):\n")
        if recall is not None:
            f.write(f"  - Sensitivity: {recall:.4f}\n")
        else:
            f.write("  - Sensitivity: N/A\n")

        if precision is not None:
            f.write(f"  - Precision:   {precision:.4f}\n")
        else:
            f.write("  - Precision:   N/A\n")

class MetricsLogger:
    """Class to track training metrics during training"""
    def __init__(self):
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.val_precisions = []
        self.val_recalls = []
    
    def log_epoch(self, train_loss, train_acc, val_loss, val_acc, val_precision, val_recall):
        """Log metrics for one epoch"""
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        self.val_precisions.append(val_precision)
        self.val_recalls.append(val_recall)
    
    def save_all_plots(self, test_results=None, folder_type=None):
        """Save all plots and results in organized folders"""
        return enhanced_save_all_plots(self, test_results, folder_type)

def plot_comprehensive_dashboard(results, save_path_prefix):
    """
    Create comprehensive analysis dashboard similar to EfficientNet+Meta Learning results
    Including: confusion matrices, F1 scores, error distribution, confidence scores, residual analysis
    Now saves each plot as separate image files
    """
    # Extract metrics
    cm = results.get('confusion_matrix', {})
    tp, tn, fp, fn = cm.get('tp', 0), cm.get('tn', 0), cm.get('fp', 0), cm.get('fn', 0)
    total = tp + tn + fp + fn
    
    # Calculate normalized confusion matrix
    cm_normalized = np.array([[tn/(tn+fp) if (tn+fp)>0 else 0, fp/(tn+fp) if (tn+fp)>0 else 0],
                             [fn/(fn+tp) if (fn+tp)>0 else 0, tp/(fn+tp) if (fn+tp)>0 else 0]])
    
    # Calculate F1 scores
    precision_live = tn/(tn+fn) if (tn+fn)>0 else 0
    recall_live = tn/(tn+fp) if (tn+fp)>0 else 0
    f1_live = 2*(precision_live*recall_live)/(precision_live+recall_live) if (precision_live+recall_live)>0 else 0
    
    precision_spoof = tp/(tp+fp) if (tp+fp)>0 else 0
    recall_spoof = tp/(tp+fn) if (tp+fn)>0 else 0
    f1_spoof = 2*(precision_spoof*recall_spoof)/(precision_spoof+recall_spoof) if (precision_spoof+recall_spoof)>0 else 0
    
    macro_f1 = (f1_live + f1_spoof) / 2
    weighted_f1 = results.get('f1_score', macro_f1)
    
    model_name = results.get('model_name', 'Model')
    
    # 1. Raw Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm_raw = np.array([[tn, fp], [fn, tp]])
    im1 = plt.imshow(cm_raw, cmap='Blues', aspect='auto')
    plt.title(f'{model_name}: Confusion Matrix (Raw Counts)', fontweight='bold', fontsize=14)
    plt.xticks([0, 1], ['Live', 'Spoof'])
    plt.yticks([0, 1], ['Live', 'Spoof'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm_raw[i, j]), ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.colorbar(im1, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_confusion_matrix_raw.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Normalized Confusion Matrix
    plt.figure(figsize=(10, 8))
    im2 = plt.imshow(cm_normalized, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    plt.title(f'{model_name}: Normalized Confusion Matrix (Percentages)', fontweight='bold', fontsize=14)
    plt.xticks([0, 1], ['Live', 'Spoof'])
    plt.yticks([0, 1], ['Live', 'Spoof'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar(im2, fraction=0.046, pad=0.04)
    
    # Add percentage annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f'{cm_normalized[i, j]:.2%}', ha='center', va='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_confusion_matrix_normalized.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. F1 Score Comparison
    plt.figure(figsize=(12, 8))
    f1_scores = [f1_spoof, f1_live, macro_f1, weighted_f1]
    f1_labels = ['Spoof F1', 'Live F1', 'Macro F1', 'Weighted F1']
    colors = ['#ff7f7f', '#7fb3d3', '#90EE90', '#FFD700']
    bars = plt.bar(f1_labels, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    plt.title(f'{model_name}: F1 Score Comparison', fontweight='bold', fontsize=14)
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_f1_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Prediction Error Distribution (MSE)
    plt.figure(figsize=(12, 8))
    if 'y_true' in results and 'y_scores' in results:
        y_true = np.array(results['y_true'])
        y_pred = np.array(results['y_scores'])
        mse_errors = (y_true - y_pred) ** 2
        plt.hist(mse_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(mse_errors), color='red', linestyle='--', linewidth=2, label='Mean MSE')
        plt.title(f'{model_name}: Prediction Error Distribution\nMSE: {np.mean(mse_errors):.4f}', fontweight='bold', fontsize=14)
        plt.xlabel('Error (True - Predicted)')
        plt.ylabel('Frequency')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'MSE data not available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f'{model_name}: Prediction Error Distribution\nMSE: N/A', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_mse_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. True vs Predicted Values (RMSE)
    plt.figure(figsize=(12, 8))
    if 'y_true' in results and 'y_scores' in results:
        y_true = np.array(results['y_true'])
        y_pred = np.array(results['y_scores'])
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Scatter plot
        scatter_live = plt.scatter(y_true[y_true==0], y_pred[y_true==0], alpha=0.6, color='blue', label='Live', s=20)
        scatter_spoof = plt.scatter(y_true[y_true==1], y_pred[y_true==1], alpha=0.6, color='red', label='Spoof', s=20)
        
        # Perfect prediction line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.75, linewidth=2, label='Perfect Prediction')
        
        plt.title(f'{model_name}: True vs Predicted Values\nRMSE: {rmse:.4f}', fontweight='bold', fontsize=14)
        plt.xlabel('True Label')
        plt.ylabel('Predicted Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'RMSE data not available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f'{model_name}: True vs Predicted Values\nRMSE: N/A', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_rmse_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Confidence Score Distribution
    plt.figure(figsize=(12, 8))
    if 'y_scores' in results:
        y_scores = np.array(results['y_scores'])
        y_true = np.array(results.get('y_true', []))
        
        if len(y_true) > 0:
            live_scores = y_scores[y_true == 0]
            spoof_scores = y_scores[y_true == 1]
            
            plt.hist(live_scores, bins=30, alpha=0.7, color='blue', label='Live', density=True)
            plt.hist(spoof_scores, bins=30, alpha=0.7, color='red', label='Spoof', density=True)
            plt.title(f'{model_name}: Confidence Score Distribution', fontweight='bold', fontsize=14)
            plt.xlabel('Confidence Score')
            plt.ylabel('Density')
            plt.legend()
        else:
            plt.hist(y_scores, bins=50, alpha=0.7, color='skyblue')
            plt.title(f'{model_name}: Confidence Score Distribution', fontweight='bold', fontsize=14)
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
    else:
        plt.text(0.5, 0.5, 'Confidence data not available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f'{model_name}: Confidence Score Distribution', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_confidence_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Performance Metrics Overview
    plt.figure(figsize=(12, 8))
    metrics_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    metrics_values = [
        results.get('accuracy', 0)/100 if results.get('accuracy', 0) > 1 else results.get('accuracy', 0),
        results.get('f1_score', 0),
        results.get('precision', 0), 
        results.get('recall', 0)
    ]
    
    colors = ['#90EE90', '#87CEEB', '#DDA0DD', '#F0E68C']
    bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='black')
    plt.title(f'{model_name}: Performance Metrics Overview', fontweight='bold', fontsize=14)
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    for bar, value in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_metrics_overview.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Residual Analysis
    plt.figure(figsize=(12, 8))
    if 'y_true' in results and 'y_scores' in results:
        y_true = np.array(results['y_true'])
        y_pred = np.array(results['y_scores'])
        residuals = y_true - y_pred
        
        # Plot residuals vs predicted
        plt.scatter(y_pred, residuals, alpha=0.6, color='purple', s=20)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        
        # Add trend lines
        z = np.polyfit(y_pred, residuals, 1)
        p = np.poly1d(z)
        plt.plot(y_pred, p(y_pred), "r--", alpha=0.8, linewidth=2)
        
        plt.title(f'{model_name}: Residual Analysis', fontweight='bold', fontsize=14)
        plt.xlabel('Predicted Probability Level')
        plt.ylabel('Residual (True - Predicted)')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Residual data not available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f'{model_name}: Residual Analysis', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_residual_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comprehensive dashboard plots saved as separate images with prefix: {save_path_prefix}")

def plot_performance_analysis(results, save_path_prefix):
    """
    Create performance analysis with MSE/RMSE focus
    Now saves each plot as separate image files
    """
    model_name = results.get('model_name', 'Model')
    
    # Extract metrics
    cm = results.get('confusion_matrix', {})
    tp, tn, fp, fn = cm.get('tp', 0), cm.get('tn', 0), cm.get('fp', 0), cm.get('fn', 0)
    total = tp + tn + fp + fn
    
    accuracy = results.get('accuracy', 0)
    if accuracy > 1:
        accuracy = accuracy / 100
    
    mse = results.get('mse', 0)
    rmse = results.get('rmse', 0)
    if rmse == 0 and mse > 0:
        rmse = np.sqrt(mse)
    
    # 1. Comprehensive Model Performance Metrics
    plt.figure(figsize=(14, 8))
    metrics = ['Accuracy', 'Weighted F1', 'Macro F1', 'MSE', 'RMSE']
    values = [accuracy, results.get('f1_score', 0), results.get('f1_score', 0), mse, rmse]
    colors = ['#90EE90', '#87CEEB', '#DDA0DD', '#FFB6C1', '#FFA07A']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    plt.title(f'{model_name}: Comprehensive Model Performance Metrics', fontweight='bold', fontsize=14)
    plt.ylabel('Score/Error Value')
    
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                f'{value:.4f}' if value < 1 else f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_performance_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. MSE vs RMSE Comparison (Pie Chart)
    plt.figure(figsize=(10, 8))
    if mse > 0 and rmse > 0:
        sizes = [mse, rmse]
        labels = [f'MSE\n{mse:.4f}', f'RMSE\n{rmse:.4f}']
        colors_pie = ['#FFA07A', '#87CEEB']
        plt.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
        plt.title(f'{model_name}: MSE vs RMSE Comparison', fontweight='bold', fontsize=14)
    else:
        plt.text(0.5, 0.5, 'MSE/RMSE\nNot Available', ha='center', va='center', transform=plt.gca().transAxes, fontweight='bold')
        plt.title(f'{model_name}: MSE vs RMSE Comparison', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_mse_rmse_pie.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. RMSE Quality Gauge
    plt.figure(figsize=(10, 8))
    if rmse > 0:
        # Create gauge visualization
        theta = np.linspace(0, np.pi, 100)
        r = 1
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        plt.plot(x, y, 'k-', linewidth=3)
        
        # Determine quality based on RMSE
        if rmse <= 0.1:
            quality = "EXCELLENT"
            color = 'green'
            angle = np.pi * 0.8
        elif rmse <= 0.2:
            quality = "GOOD"
            color = 'yellow'
            angle = np.pi * 0.6
        elif rmse <= 0.3:
            quality = "FAIR"
            color = 'orange'
            angle = np.pi * 0.4
        else:
            quality = "POOR"
            color = 'red'
            angle = np.pi * 0.2
        
        # Draw gauge needle
        needle_x = 0.8 * np.cos(angle)
        needle_y = 0.8 * np.sin(angle)
        plt.arrow(0, 0, needle_x, needle_y, head_width=0.1, head_length=0.1, fc=color, ec=color, linewidth=3)
        
        plt.text(0, -0.3, f'{rmse:.4f}', ha='center', va='center', fontsize=16, fontweight='bold')
        plt.text(0, -0.5, quality, ha='center', va='center', fontsize=12, fontweight='bold', color=color)
        plt.xlim(-1.2, 1.2)
        plt.ylim(-0.7, 1.2)
        plt.gca().set_aspect('equal')
        plt.axis('off')
        plt.title(f'{model_name}: RMSE Quality Gauge', fontweight='bold', fontsize=14)
    else:
        plt.text(0.5, 0.5, 'RMSE\nNot Available', ha='center', va='center', transform=plt.gca().transAxes, fontweight='bold')
        plt.title(f'{model_name}: RMSE Quality Gauge', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_vit_rmse_gauge.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Label Distribution
    plt.figure(figsize=(12, 8))
    if 'y_true' in results:
        y_true = np.array(results['y_true'])
        unique, counts = np.unique(y_true, return_counts=True)
        
        spoof_count = counts[0] if len(counts) > 0 else 0
        live_count = counts[1] if len(counts) > 1 else 0
        
        labels = ['Spoof (0)', 'Live (1)']
        counts_data = [spoof_count, live_count]
        colors = ['#FFB6C1', '#87CEEB']
        
        bars = plt.bar(labels, counts_data, color=colors, alpha=0.8, edgecolor='black')
        plt.title(f'{model_name}: Label Distribution (True vs Predicted)', fontweight='bold', fontsize=14)
        plt.ylabel('Count')
        
        for bar, count in zip(bars, counts_data):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts_data)*0.01,
                    str(count), ha='center', va='bottom', fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'Label data\nnot available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f'{model_name}: Label Distribution (True vs Predicted)', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_vit_label_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Prediction Confidence Distribution
    plt.figure(figsize=(12, 8))
    if 'y_scores' in results:
        y_scores = np.array(results['y_scores'])
        correct_mask = np.array(results.get('correct_predictions', [True]*len(y_scores)))
        
        correct_scores = y_scores[correct_mask] if len(correct_mask) > 0 else y_scores
        incorrect_scores = y_scores[~correct_mask] if len(correct_mask) > 0 else []
        
        if len(correct_scores) > 0:
            plt.hist(correct_scores, bins=30, alpha=0.7, color='green', label=f'Correct ({len(correct_scores)})', density=True)
        if len(incorrect_scores) > 0:
            plt.hist(incorrect_scores, bins=30, alpha=0.7, color='red', label=f'Incorrect ({len(incorrect_scores)})', density=True)
        
        plt.title(f'{model_name}: Prediction Confidence Distribution', fontweight='bold', fontsize=14)
        plt.xlabel('Confidence Score')
        plt.ylabel('Density')
        if len(correct_scores) > 0 or len(incorrect_scores) > 0:
            plt.legend()
    else:
        plt.text(0.5, 0.5, 'Confidence data\nnot available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f'{model_name}: Prediction Confidence Distribution', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_vit_confidence_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Detailed Error Analysis (Pie Chart)
    plt.figure(figsize=(10, 8))
    if total > 0:
        sizes = [tn, fp, fn, tp]
        labels = [f'True Negatives\n(Correct Live)\n{tn}', f'False Positives\n(Wrong Spoof)\n{fp}',
                 f'False Negatives\n(Wrong Live)\n{fn}', f'True Positives\n(Correct Spoof)\n{tp}']
        colors = ['#90EE90', '#FFB6C1', '#FFA07A', '#87CEEB']
        
        # Only show non-zero segments
        non_zero_sizes = []
        non_zero_labels = []
        non_zero_colors = []
        for size, label, color in zip(sizes, labels, colors):
            if size > 0:
                non_zero_sizes.append(size)
                non_zero_labels.append(label)
                non_zero_colors.append(color)
        
        if non_zero_sizes:
            plt.pie(non_zero_sizes, labels=non_zero_labels, colors=non_zero_colors, autopct='%1.1f%%', startangle=90)
        plt.title(f'{model_name}: Detailed Error Analysis (Classification Breakdown)', fontweight='bold', fontsize=14)
    else:
        plt.text(0.5, 0.5, 'Error data\nnot available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f'{model_name}: Detailed Error Analysis (Classification Breakdown)', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_vit_error_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ ViT performance analysis plots saved as separate images with prefix: {save_path_prefix}")

def plot_mse_rmse_dashboard(results, save_path_prefix):
    """
    Create detailed MSE & RMSE Analysis Dashboard
    Now saves each plot as separate image files
    """
    mse = results.get('mse', 0)
    rmse = results.get('rmse', 0)
    if rmse == 0 and mse > 0:
        rmse = np.sqrt(mse)
    
    model_name = results.get('model_name', 'Model')
    
    # 1. MSE vs RMSE Values
    plt.figure(figsize=(12, 8))
    metrics = ['MSE', 'RMSE']
    values = [mse, rmse]
    colors = ['#FFB6C1', '#FFA07A']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    plt.title(f'{model_name}: MSE vs RMSE Values', fontweight='bold', fontsize=14)
    plt.ylabel('Error Value')
    
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.05,
                f'{value:.6f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_mse_rmse_values.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. RMSE: Current vs Perfect Model
    plt.figure(figsize=(12, 8))
    models = ['Current Model', 'Perfect Model']
    rmse_values = [rmse, 0.0]
    colors = ['#FFA07A', '#90EE90']
    
    bars = plt.bar(models, rmse_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    plt.title(f'{model_name}: RMSE Current vs Perfect Model', fontweight='bold', fontsize=14)
    plt.ylabel('RMSE Value')
    
    for bar, value in zip(bars, rmse_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values)*0.05,
                f'{value:.6f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_rmse_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Error Severity Assessment (Donut Chart)
    plt.figure(figsize=(10, 8))
    
    # Determine error severity based on RMSE
    if rmse <= 0.1:
        severity = "GOOD"
        severity_color = '#90EE90'
        remaining = 1 - (rmse / 0.1)
    elif rmse <= 0.3:
        severity = "FAIR"
        severity_color = '#FFD700'
        remaining = 1 - ((rmse - 0.1) / 0.2)
    else:
        severity = "POOR"
        severity_color = '#FFB6C1'
        remaining = 0.1
    
    # Create donut chart
    sizes = [rmse if rmse <= 1 else 1, remaining] if remaining > 0 else [1, 0]
    colors = [severity_color, '#E0E0E0']
    
    wedges, texts = plt.pie(sizes, colors=colors, startangle=90, counterclock=False)
    
    # Add center circle for donut effect
    center_circle = plt.Circle((0,0), 0.70, fc='white')
    plt.gca().add_artist(center_circle)
    
    # Add text in center
    plt.text(0, 0.1, severity, ha='center', va='center', fontsize=14, fontweight='bold')
    plt.text(0, -0.1, f'RMSE: {rmse:.4f}', ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.title(f'{model_name}: Error Severity Assessment', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_error_severity.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ MSE/RMSE dashboard plots saved as separate images with prefix: {save_path_prefix}")

def plot_calibration_curve(y_true, y_scores, save_path, model_name="Model", n_bins=10):
    """
    Plot calibration curve (reliability diagram) to assess predicted probability accuracy.
    """
    from sklearn.calibration import calibration_curve
    
    plt.figure(figsize=(10, 8))
    
    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_scores, n_bins=n_bins, strategy='uniform')
    
    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    # Plot model's calibration curve
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=model_name)
    
    plt.title(f'{model_name}: Calibration Plot (Reliability Diagram)', fontsize=16, fontweight='bold')
    plt.xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
    plt.ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Calibration plot saved: {save_path}")

def calculate_eer_hter(y_true, y_scores):
    """
    Calculate Equal Error Rate (EER) and Half Total Error Rate (HTER)
    """
    if len(y_true) == 0 or len(y_scores) == 0:
        return 0.0, 0.0
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Calculate EER (point where FPR = FNR)
    fnr = 1 - tpr
    eer_threshold_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2
    
    # Calculate HTER at EER threshold
    threshold = thresholds[eer_threshold_idx]
    y_pred = (y_scores >= threshold).astype(int)
    
    # Calculate confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate FAR and FRR
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # HTER is the average of FAR and FRR
    hter = (far + frr) / 2
    
    return eer * 100, hter * 100  # Return as percentages

def plot_precision_recall_curve(y_true, y_scores, save_path, model_name="Model"):
    """
    Plot Precision-Recall curve.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title(f'{model_name}: Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Precision-Recall curve saved: {save_path}")

def enhanced_save_all_plots(self, test_results=None, folder_type=None):
    """Enhanced version that creates all comprehensive plots"""
    results_dir = create_results_folder(folder_type)
    index = get_next_index(results_dir, "train")
    date_str = datetime.now().strftime('%Y%m%d')
    
    base_name = f"train_{index}_{date_str}"
    result_folder = os.path.join(results_dir, base_name)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    # Get model name for plot titles
    model_name = test_results.get('model_name', 'Model') if test_results else 'Model'
    
    # Save original training metrics with model name
    metrics_path = os.path.join(result_folder, "training_metrics.png")
    plot_training_metrics_with_model_name(self.train_losses, self.train_accs, self.val_losses,
                        self.val_accs, self.val_precisions, self.val_recalls,
                        metrics_path, model_name)
    
    loss_curve_path = os.path.join(result_folder, "loss_curve.png")
    plot_loss_curve_with_model_name(self.train_losses, self.val_losses, loss_curve_path, model_name)
    
    acc_curve_path = os.path.join(result_folder, "accuracy_curve.png")
    plot_accuracy_curve_with_model_name(self.train_accs, self.val_accs, acc_curve_path, model_name)
    
    # Only create comprehensive test analysis if full test results are provided
    if test_results and 'confusion_matrix' in test_results:
        test_base_name = f"test_{index}_{date_str}"
        test_folder = os.path.join(results_dir, test_base_name)
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)
        
        # Calculate EER and HTER
        if 'y_true' in test_results and 'y_scores' in test_results:
            eer, hter = calculate_eer_hter(test_results['y_true'], test_results['y_scores'])
            test_results['eer'] = eer
            test_results['hter'] = hter
        
        # Calculate MSE and RMSE if not present
        if 'y_true' in test_results and 'y_scores' in test_results and 'mse' not in test_results:
            y_true = np.array(test_results['y_true'])
            y_scores = np.array(test_results['y_scores'])
            mse = mean_squared_error(y_true, y_scores)
            rmse = np.sqrt(mse)
            test_results['mse'] = mse
            test_results['rmse'] = rmse
        
        # Save original plots
        cm_path = os.path.join(test_folder, "confusion_matrix.png")
        plot_confusion_matrix(test_results['confusion_matrix'], ['live', 'spoof'], cm_path)
        
        if 'y_true' in test_results and 'y_scores' in test_results:
            roc_path = os.path.join(test_folder, "roc_curve.png")
            roc_auc = plot_roc_curve(test_results['y_true'], test_results['y_scores'], roc_path)
            
            # New plots
            calibration_path = os.path.join(test_folder, "calibration_curve.png")
            plot_calibration_curve(test_results['y_true'], test_results['y_scores'], calibration_path, model_name)
            
            pr_curve_path = os.path.join(test_folder, "precision_recall_curve.png")
            plot_precision_recall_curve(test_results['y_true'], test_results['y_scores'], pr_curve_path, model_name)

        summary_path = os.path.join(test_folder, "summary.txt")
        save_metrics_summary(test_results, summary_path)
        
        # Save comprehensive dashboards
        save_prefix = os.path.join(test_folder, "advanced")
        plot_comprehensive_dashboard(test_results, save_prefix)
        plot_performance_analysis(test_results, save_prefix)
        plot_mse_rmse_dashboard(test_results, save_prefix)
        
        print(f"✅ All comprehensive analysis plots saved in: {test_folder}")
        return base_name, result_folder, test_base_name, test_folder
    
    print(f"✅ Training plots saved in: {result_folder}")
    return base_name, result_folder

def plot_eer_hter_analysis(eer, hter, save_path, model_name="Model"):
    """
    Create a comprehensive EER and HTER analysis plot with bar chart and gauge visualization.
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Create subplots
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])  # Bar chart
    ax2 = fig.add_subplot(gs[0, 1])  # EER Gauge
    ax3 = fig.add_subplot(gs[1, :])  # HTER Gauge
    
    # 1. Bar Chart for EER and HTER comparison
    metrics = ['EER', 'HTER']
    values = [eer, hter]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax1.bar(metrics, values, color=colors, alpha=0.8, width=0.6)
    ax1.set_title(f'{model_name}: EER vs HTER Comparison', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Error Rate (%)', fontweight='bold')
    ax1.set_ylim(0, max(values) * 1.2 + 5)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 2. EER Gauge Chart
    ax2.set_title(f'{model_name}: EER Quality Gauge', fontweight='bold', fontsize=14)
    
    # Create gauge background
    theta = np.linspace(np.pi, 0, 100)
    r = 1
    x_gauge = r * np.cos(theta)
    y_gauge = r * np.sin(theta)
    ax2.plot(x_gauge, y_gauge, 'k-', linewidth=3)
    
    # Determine EER quality zones
    if eer <= 5:
        quality = "EXCELLENT"
        color = '#2ECC71'
        angle = np.pi * (1 - eer/20)  # Map 0-5% to 180-90 degrees
    elif eer <= 10:
        quality = "GOOD"
        color = '#F1C40F'
        angle = np.pi * (1 - eer/20)
    elif eer <= 15:
        quality = "FAIR"
        color = '#E67E22'
        angle = np.pi * (1 - eer/20)
    else:
        quality = "POOR"
        color = '#E74C3C'
        angle = np.pi * (1 - eer/20)
    
    # Draw gauge needle
    needle_x = [0, 0.8 * np.cos(angle)]
    needle_y = [0, 0.8 * np.sin(angle)]
    ax2.plot(needle_x, needle_y, color=color, linewidth=4)
    
    # Add quality labels
    ax2.text(0, -0.3, quality, ha='center', va='center', fontweight='bold', fontsize=12, color=color)
    ax2.text(0, -0.6, f'EER: {eer:.2f}%', ha='center', va='center', fontweight='bold', fontsize=10)
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 0.2)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # 3. HTER Gauge Chart
    ax3.set_title(f'{model_name}: HTER Quality Gauge', fontweight='bold', fontsize=14)
    
    # Create gauge background
    ax3.plot(x_gauge, y_gauge, 'k-', linewidth=3)
    
    # Determine HTER quality zones
    if hter <= 5:
        quality_hter = "EXCELLENT"
        color_hter = '#2ECC71'
        angle_hter = np.pi * (1 - hter/20)
    elif hter <= 10:
        quality_hter = "GOOD"
        color_hter = '#F1C40F'
        angle_hter = np.pi * (1 - hter/20)
    elif hter <= 15:
        quality_hter = "FAIR"
        color_hter = '#E67E22'
        angle_hter = np.pi * (1 - hter/20)
    else:
        quality_hter = "POOR"
        color_hter = '#E74C3C'
        angle_hter = np.pi * (1 - hter/20)
    
    # Draw gauge needle
    needle_x_hter = [0, 0.8 * np.cos(angle_hter)]
    needle_y_hter = [0, 0.8 * np.sin(angle_hter)]
    ax3.plot(needle_x_hter, needle_y_hter, color=color_hter, linewidth=4)
    
    # Add quality labels
    ax3.text(0, -0.3, quality_hter, ha='center', va='center', fontweight='bold', fontsize=12, color=color_hter)
    ax3.text(0, -0.6, f'HTER: {hter:.2f}%', ha='center', va='center', fontweight='bold', fontsize=10)
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 0.2)
    ax3.set_aspect('equal')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ EER/HTER analysis plot saved: {save_path}")

# Add enhanced method to MetricsLogger class
