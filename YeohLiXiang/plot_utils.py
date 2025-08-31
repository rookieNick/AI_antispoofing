import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import roc_curve, auc

def create_results_folder():
    """Create results folder if it doesn't exist"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
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
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')
    
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
    axes[1, 2].text(0.5, 0.5, 'Model Architecture:\nOptimizedCNN\n\nLayers:\n- 3 Conv blocks\n- Global Avg Pool\n- 3 FC layers\n\nActivation: SiLU\nOptimizer: AdamW\nScheduler: OneCycleLR', 
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
    plt.figure(figsize=(12, 8))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    
    plt.title('Training vs Validation Loss', fontsize=16, fontweight='bold')
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
    plt.figure(figsize=(12, 8))
    
    epochs = range(1, len(train_accs) + 1)
    
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
    
    plt.title('Training vs Validation Accuracy', fontsize=16, fontweight='bold')
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
    with open(save_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("         MODEL EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: OptimizedCNN\n")
        f.write("\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Test Accuracy:    {results['accuracy']:.2f}%\n")
        f.write(f"Test Loss:        {results['loss']:.4f}\n")
        f.write(f"Test Precision:   {results['precision']:.4f}\n")
        f.write(f"Test Recall:      {results['recall']:.4f}\n")
        f.write(f"Test F1-Score:    {results['f1_score']:.4f}\n")
        f.write(f"Test Specificity: {results['specificity']:.4f}\n")
        f.write("\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write("-" * 30 + "\n")
        cm = results['confusion_matrix']
        f.write(f"True Positives:   {cm['tp']}\n")
        f.write(f"True Negatives:   {cm['tn']}\n")
        f.write(f"False Positives:  {cm['fp']}\n")
        f.write(f"False Negatives:  {cm['fn']}\n")
        f.write(f"Total Samples:    {cm['tp'] + cm['tn'] + cm['fp'] + cm['fn']}\n")
        f.write("\n")
        
        f.write("CLASS-WISE PERFORMANCE:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Live Detection (Class 0):\n")
        f.write(f"  - Sensitivity: {cm['tn']/(cm['tn'] + cm['fp']):.4f}\n")
        f.write(f"  - Specificity: {results['specificity']:.4f}\n")
        f.write(f"Spoof Detection (Class 1):\n")
        f.write(f"  - Sensitivity: {results['recall']:.4f}\n")
        f.write(f"  - Precision:   {results['precision']:.4f}\n")

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
    
    def save_all_plots(self, test_results=None):
        """Save all plots and results in organized folders"""
        results_dir = create_results_folder()
        index = get_next_index(results_dir, "train")
        date_str = datetime.now().strftime('%Y%m%d')
        
        base_name = f"train_{index}_{date_str}"
        # Create a dedicated folder for this training result set
        result_folder = os.path.join(results_dir, base_name)
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        
        # Save training metrics plot (original combined plot)
        metrics_path = os.path.join(result_folder, "training_metrics.png")
        plot_training_metrics(self.train_losses, self.train_accs, self.val_losses, 
                            self.val_accs, self.val_precisions, self.val_recalls, 
                            metrics_path)
        print(f"Training metrics plot saved: {metrics_path}")
        
        # Save separate loss curve
        loss_curve_path = os.path.join(result_folder, "loss_curve.png")
        plot_loss_curve(self.train_losses, self.val_losses, loss_curve_path)
        print(f"Loss curve plot saved: {loss_curve_path}")
        
        # Save separate accuracy curve
        acc_curve_path = os.path.join(result_folder, "accuracy_curve.png")
        plot_accuracy_curve(self.train_accs, self.val_accs, acc_curve_path)
        print(f"Accuracy curve plot saved: {acc_curve_path}")
        
        # Save test results if provided
        if test_results:
            test_base_name = f"test_{index}_{date_str}"
            test_folder = os.path.join(results_dir, test_base_name)
            if not os.path.exists(test_folder):
                os.makedirs(test_folder)
            # Confusion matrix
            cm_path = os.path.join(test_folder, "confusion_matrix.png")
            plot_confusion_matrix(test_results['confusion_matrix'], ['live', 'spoof'], cm_path)
            print(f"Confusion matrix saved: {cm_path}")
            # ROC curve (if y_true and y_scores are provided)
            if 'y_true' in test_results and 'y_scores' in test_results:
                roc_path = os.path.join(test_folder, "roc_curve.png")
                roc_auc = plot_roc_curve(test_results['y_true'], test_results['y_scores'], roc_path)
                print(f"ROC curve saved: {roc_path} (AUC: {roc_auc:.4f})")
            # Metrics summary
            summary_path = os.path.join(test_folder, "summary.txt")
            save_metrics_summary(test_results, summary_path)
            print(f"Metrics summary saved: {summary_path}")
            return base_name, result_folder, test_base_name, test_folder
        
        # If no test results, return only training results
        return base_name, result_folder
