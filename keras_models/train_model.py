# TensorFlow and tf.keras
import tensorflow as tf
import os
# Helper libraries
import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

# Define paths to training and testing directories
train_dir = "casia-fasd/train"
test_dir = "casia-fasd/test"

def train_model():

    # Construct the path to the model file
    best_keras_model_path = os.path.join(os.path.dirname(__file__),'best_model.keras')


    # Load training dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=train_dir,
        labels='inferred',               # Automatically label based on subfolder names ('live' and 'spoof')
        label_mode='int',                # Labels as integers
        color_mode='rgb',                # Load images as RGB
        batch_size=16,                   # Choose an appropriate batch size
        image_size=(128, 128),           # Resize images to a consistent size (e.g., 128x128)
        shuffle=True                     # Shuffle training data
    )

    # Load test dataset
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=test_dir,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=16,
        image_size=(128, 128),
        shuffle=False                     # Typically, test data is not shuffled
    )

    # Class names based on folder structure
    class_names = train_ds.class_names
    print("Class names:", class_names)

    # Normalize images to [0, 1]
    def normalize_img(image, label):
        return image / 255.0, label  # Normalize the image

    train_ds = train_ds.map(normalize_img)
    test_ds = test_ds.map(normalize_img)


    # Limit the training dataset to the first 1000 images
    # test_ds = test_ds.take(100)
    # train_ds = train_ds.take(100)


    # # Display some sample images from the dataset to confirm data loading and labels
    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #     for i in range(16):  # Display first n images
    #         ax = plt.subplot(4, 4, i + 1)
    #         plt.imshow(images[i].numpy())
    #         plt.title(class_names[labels[i]])
    #         plt.axis("off")
    # plt.show()

    # Optimize dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Check if the model already exists
    if os.path.exists(best_keras_model_path):
        print(f"Loading existing model from {best_keras_model_path}")
        model = tf.keras.models.load_model(best_keras_model_path)
    else:
        print("No existing model found. Creating a new one.")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(128, 128, 3)),
            # First Conv layer with regularization
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.3),  # Dropout after first conv layer

            # Second Conv layer with regularization
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', 
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.3),  # Dropout after second conv layer

            # Flatten and dense layers with regularization and dropout
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu', 
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001)),
            tf.keras.layers.Dropout(0.5),  # Dropout after dense layer
            
            # Output layer for 2 classes
            tf.keras.layers.Dense(2)  # Logits layer
        ])
        model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])





    # Set up early stopping and model checkpoint
    # stop early if patience more than n times but still no show improvement
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

    # save everytime if improvement in accuracy
    checkpoint = tf.keras.callbacks.ModelCheckpoint(best_keras_model_path, save_best_only=True, verbose=1)

    # Train the model with early stopping and checkpoint
    history = model.fit(train_ds, validation_data=test_ds, epochs=10, callbacks=[early_stopping, checkpoint])

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    print("\nTest accuracy:", test_acc)