# TensorFlow and tf.keras
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import Counter

print("TensorFlow version:", tf.__version__)

# --- GPU Optimization Setup ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Found {len(gpus)} GPU(s): {gpus}")
    # Enable memory growth to avoid allocating all GPU memory at once
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Set memory limit to 5GB to leave room for system and avoid OOM
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)]
    )
    print("GPU memory growth enabled, 5GB limit set")
else:
    print("No GPU found, using CPU")

# --- Define dataset paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.normpath(os.path.join(script_dir, "..", "casia-fasd", "train"))
test_dir = os.path.normpath(os.path.join(script_dir, "..", "casia-fasd", "test"))

def train_model():
    # Path to save the best model
    best_keras_model_path = os.path.join(script_dir, 'best_model.keras')
    
    print(f"Training directory: {train_dir}")
    print(f"Test directory: {test_dir}")
    print(f"Model save path: {best_keras_model_path}")

    # --- Training configuration ---
    BATCH_SIZE = 64  # Small batch size to fit in GPU memory
    IMAGE_SIZE = (112, 112)  # Image size for training and testing
    EPOCHS = 30  # Number of training epochs
    
    print(f"Using batch size: {BATCH_SIZE}")
    print(f"Using image size: {IMAGE_SIZE}")

    # --- Load training dataset ---
    print("Loading training dataset...")
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=train_dir,
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=42
    )

    # --- Load test dataset ---
    print("Loading test dataset...")
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=test_dir,
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=False
    )

    # --- Get class names ---
    class_names = train_ds.class_names
    print(f"Class names: {class_names}")
    print(f"Number of classes: {len(class_names)}")
    
    # Print dataset info
    print(f"Training samples: {len(train_ds) * BATCH_SIZE}")
    print(f"Test samples: {len(test_ds) * BATCH_SIZE}")

    # --- Compute class weights for imbalanced data ---
    # This helps the model pay more attention to underrepresented classes (e.g., 'live')
    print("Computing class weights...")
    all_labels = []
    # Unbatch the dataset to get all individual labels
    for _, labels in train_ds.unbatch():
        all_labels.append(np.argmax(labels.numpy()))
    label_counts = Counter(all_labels)
    print(f"Label counts: {label_counts}")
    total = sum(label_counts.values())
    # Inverse frequency weighting: higher weight for less frequent classes
    class_weight = {i: total/(len(class_names)*count) for i, count in label_counts.items()}
    print(f"Class weights: {class_weight}")

    # --- Data pipeline optimization ---
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
    print("Memory-efficient data pipeline enabled")

    # --- Data augmentation for robustness ---
    # These augmentations help the model generalize to new spoofing attacks
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.RandomFlip("horizontal"),
        # tf.keras.layers.RandomRotation(0.1),
        # tf.keras.layers.RandomZoom(0.1),
        # tf.keras.layers.RandomBrightness(0.1),
        # tf.keras.layers.RandomContrast(0.1),
        # tf.keras.layers.GaussianNoise(0.01),  # Add noise for robustness
    ])

    # --- Model architecture ---
    # A compact CNN with batch normalization and dropout for regularization
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(*IMAGE_SIZE, 3)),
        data_augmentation,
        
        # First Conv block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.5),

        # Second Conv block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.5),

        # Third Conv block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.5),

        # Fourth Conv block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.5),

        # Global pooling and dense layers
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        # Output layer
        tf.keras.layers.Dense(len(class_names), activation='softmax')
    ])

    # --- Compile the model ---
    # Use Adam optimizer, categorical crossentropy, and track accuracy/precision/recall
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    # Print model summary
    model.summary()

    # --- Callbacks for training ---
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10,
        restore_best_weights=True, 
        verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=5,
        min_lr=1e-7, 
        verbose=1
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        best_keras_model_path, 
        save_best_only=True, 
        monitor='val_accuracy',
        verbose=1
    )

    # --- Custom callback to print performance every 5 epochs ---
    class PerformanceCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.start_time = None
            
        def on_epoch_begin(self, epoch, logs=None):
            if epoch == 0:
                self.start_time = time.time()
                
        def on_epoch_end(self, epoch, logs=None):

            elapsed = time.time() - self.start_time
            print(f"\nEpoch {epoch}: {elapsed:.1f}s elapsed, "
                    f"Train Acc: {logs['accuracy']:.4f}, "
                    f"Val Acc: {logs['val_accuracy']:.4f}, "
                    f"Val Precision: {logs['val_precision']:.4f}, "
                    f"Val Recall: {logs['val_recall']:.4f}")

    # --- Train the model ---
    print("Starting training...")
    start_time = time.time()
    
    history = model.fit(
        train_ds, 
        validation_data=test_ds, 
        epochs=EPOCHS,
        callbacks=[early_stopping, reduce_lr, checkpoint, PerformanceCallback()],
        verbose=1,
        class_weight=class_weight  # <-- This is the key for handling class imbalance
    )
    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time/60:.1f} minutes")

    # --- Evaluate the model ---
    print("\nEvaluating final model...")
    results = model.evaluate(test_ds, verbose=1)
    
    print(f"\n=== Final Results ===")
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    print(f"Test Precision: {results[2]:.4f}")
    print(f"Test Recall: {results[3]:.4f}")
    print(f"Training Time: {total_time/60:.1f} minutes")


if __name__ == '__main__':
    train_model() 