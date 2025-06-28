# facerec.py
import cv2
import numpy as np
import time
import os
# from deepface import DeepFace
import tensorflow as tf
import keras

# Path for the Keras model
keras_model_path = os.path.join(os.path.dirname(__file__), 'best_model.keras')
_keras_model = None # Initialize model variable for lazy loading

# Set the desired image size (same as model input)
keras_img_width, keras_img_height = (128, 128)
# Define the class names
class_names = ["Spoof", "Live"]

def get_keras_model():
    # Lazily loads the Keras model
    global _keras_model
    if _keras_model is None:
        print("Loading Keras model...")
        _keras_model = tf.keras.models.load_model(keras_model_path)
        print("Keras model loaded.")
    return _keras_model

def keras_predict(frame):
    # Preprocess the frame for prediction
    resized_frame = cv2.resize(frame, (keras_img_width, keras_img_height))  # Resize to match model's input size
    normalized_frame = resized_frame / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension

    # Predict using the model
    model = get_keras_model() # Get the model using the lazy loader
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    # Get prediction label and confidence score
    label = class_names[predicted_class]
    confidence = np.max(tf.nn.softmax(predictions[0])) * 100

    return label, confidence
