# TensorFlow and tf.keras
import tensorflow as tf
import os
import numpy as np

NUM_TEST_IMAGES = 60000  # Number of test images to evaluate (image size must match training, now 112x112)

print("TensorFlow version:", tf.__version__)

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.normpath(os.path.join(script_dir, "..", "casia-fasd", "test"))
best_keras_model_path = os.path.normpath(os.path.join(script_dir, "best_model.keras"))

# Debug: Print the constructed paths
print(f"Script directory: {script_dir}")
print(f"Test directory: {test_dir}")
print(f"Model path: {best_keras_model_path}")
print(f"Model file exists: {os.path.exists(best_keras_model_path)}")
print(f"Test directory exists: {os.path.exists(test_dir)}")

# Additional file checks
if os.path.exists(best_keras_model_path):
    file_size = os.path.getsize(best_keras_model_path)
    print(f"Model file size: {file_size} bytes ({file_size/1024/1024:.2f} MB)")
    print(f"Model file is readable: {os.access(best_keras_model_path, os.R_OK)}")

def calculate_confusion_matrix(y_true, y_pred, num_classes):
    """Calculate confusion matrix manually"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    return cm

def calculate_classification_report(y_true, y_pred, target_names):
    """Calculate classification report manually"""
    num_classes = len(target_names)
    cm = calculate_confusion_matrix(y_true, y_pred, num_classes)
    
    report = []
    report.append("              precision    recall  f1-score   support")
    report.append("")
    
    total_correct = 0
    total_samples = len(y_true)
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(cm[i, :])
        
        total_correct += tp
        
        report.append(f"{target_names[i]:<15} {precision:>8.3f} {recall:>8.3f} {f1:>8.3f} {support:>8}")
    
    # Calculate overall accuracy
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    report.append("")
    report.append(f"accuracy                            {accuracy:.3f} {total_samples}")
    report.append("")
    
    # Calculate macro average
    macro_precision = np.mean([cm[i, i] / np.sum(cm[:, i]) if np.sum(cm[:, i]) > 0 else 0 for i in range(num_classes)])
    macro_recall = np.mean([cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0 for i in range(num_classes)])
    macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0
    
    report.append(f"macro avg        {macro_precision:>8.3f} {macro_recall:>8.3f} {macro_f1:>8.3f} {total_samples:>8}")
    
    return "\n".join(report)

def test_model():
    # Check if model exists
    if not os.path.exists(best_keras_model_path):
        print(f"Error: Model file not found at {best_keras_model_path}")
        print("Please train the model first using train_model.py")
        return
    
    # Load the trained model
    model = tf.keras.models.load_model(best_keras_model_path)
    
    # Load test dataset
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=test_dir,
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        batch_size=16,
        image_size=(112, 112),  # Standardized image size
        shuffle=False
    )
    
    class_names = test_ds.class_names
    # Limit to first NUM_TEST_IMAGES images
    num_batches = (NUM_TEST_IMAGES + 15) // 16  # 16 is the batch size
    limited_test_ds = test_ds.take(num_batches)
    
    # Optimize dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    limited_test_ds = limited_test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    # Get predictions for first NUM_TEST_IMAGES images
    predictions = []
    true_labels = []
    image_count = 0
    for images, labels in limited_test_ds:
        batch_predictions = model.predict(images, verbose=0)
        remaining_images = NUM_TEST_IMAGES - image_count
        if remaining_images <= 0:
            break
        batch_size = min(len(images), remaining_images)
        predictions.extend(batch_predictions[:batch_size])
        true_labels.extend(labels.numpy()[:batch_size])
        image_count += batch_size
        if image_count % 1000 == 0 or image_count == NUM_TEST_IMAGES:
            print(f"Processed {image_count} images...")
        if image_count >= NUM_TEST_IMAGES:
            break
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(true_labels, axis=1)
    
    # Calculate correct and wrong for each class
    results = {}
    for i, class_name in enumerate(class_names):
        class_mask = true_classes == i
        correct = np.sum(predicted_classes[class_mask] == true_classes[class_mask])
        wrong = np.sum(predicted_classes[class_mask] != true_classes[class_mask])
        results[class_name] = (correct, wrong)
    
    print(f"Results on first {NUM_TEST_IMAGES} test images:")
    for class_name in class_names:
        correct, wrong = results[class_name]
        print(f"Class: {class_name}")
        print(f"  Correct: {correct}")
        print(f"  Wrong:   {wrong}\n")

if __name__ == '__main__':
    test_model() 