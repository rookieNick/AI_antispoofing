# Ultra-fast GPU training for GTX 1060 6GB
import tensorflow as tf
import os
import time

print("TensorFlow version:", tf.__version__)

# Aggressive GPU memory management
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Found {len(gpus)} GPU(s): {gpus}")
    # Enable memory growth
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Increased memory limit to 5GB
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)]
    )
    print("GPU memory growth enabled, 5GB limit set")
else:
    print("No GPU found, using CPU")

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.normpath(os.path.join(script_dir, "..", "casia-fasd", "train"))
test_dir = os.path.normpath(os.path.join(script_dir, "..", "casia-fasd", "test"))

def train_model():
    best_keras_model_path = os.path.join(script_dir, 'best_model.keras')
    
    print(f"Training directory: {train_dir}")
    print(f"Test directory: {test_dir}")

    # Ultra-fast settings for GTX 1060 6GB
    BATCH_SIZE = 32  # Very small batch size for maximum speed
    IMAGE_SIZE = (112, 112)  # Standardized image size for speed and consistency
    EPOCHS = 30  # Fewer epochs for faster training
    
    print(f"Using batch size: {BATCH_SIZE}")
    print(f"Using image size: {IMAGE_SIZE}")

    # Load datasets
    print("Loading datasets...")
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

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=test_dir,
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=False
    )

    class_names = train_ds.class_names
    print(f"Class names: {class_names}")
    print(f"Training samples: {len(train_ds) * BATCH_SIZE}")
    print(f"Test samples: {len(test_ds) * BATCH_SIZE}")

    # Minimal data pipeline
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    # Data augmentation for better generalization
    # data_augmentation = tf.keras.Sequential([
    #     tf.keras.layers.Rescaling(1./255),
    #     tf.keras.layers.RandomFlip("horizontal"),
    #     tf.keras.layers.RandomRotation(0.1),
    #     tf.keras.layers.RandomZoom(0.1),
    #     tf.keras.layers.RandomBrightness(0.1),
    #     tf.keras.layers.RandomContrast(0.1),
    # ])

    # Lightweight augmentation - only fast operations
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.RandomFlip("horizontal"),  # Very fast
        # Removed slow operations: RandomRotation, RandomZoom, RandomBrightness, RandomContrast
    ])

    # Ultra-lightweight model for speed with augmentation
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3)),
        data_augmentation,
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.35),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.35),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.35),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(class_names), activation='softmax')
    ])

    # Fast compilation with precision/recall metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    model.summary()

    # Minimal callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5,
        restore_best_weights=True
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        best_keras_model_path, 
        save_best_only=True, 
        monitor='val_accuracy'
    )

    # Performance monitoring with precision/recall
    class SpeedCallback(tf.keras.callbacks.Callback):
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

    # Train with maximum speed
    print("Starting ultra-fast GPU training with augmentation...")
    start_time = time.time()
    
    history = model.fit(
        train_ds, 
        validation_data=test_ds, 
        epochs=EPOCHS,
        callbacks=[early_stopping, checkpoint, SpeedCallback()],
        verbose=1
    )
    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time/60:.1f} minutes")

    # Quick evaluation with all metrics
    print("\nEvaluating model...")
    results = model.evaluate(test_ds, verbose=1)
    
    print(f"\n=== Final Results ===")
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    print(f"Test Precision: {results[2]:.4f}")
    print(f"Test Recall: {results[3]:.4f}")
    print(f"Training Time: {total_time/60:.1f} minutes")
    print(f"Speed: {total_time/60:.1f} minutes total")

if __name__ == '__main__':
    train_model() 