# Face Anti-Spoofing with TensorFlow/Keras and CASIA-FASD

## Overview
This project implements a deep learning pipeline for face anti-spoofing (presentation attack detection) using the CASIA-FASD dataset. It uses TensorFlow and Keras to train and evaluate a convolutional neural network (CNN) to distinguish between real (live) and spoofed faces.

- **Dataset:** [CASIA-FASD](http://www.cbsr.ia.ac.cn/english/FASDB_database.asp)
- **Framework:** TensorFlow 2.x, Keras
- **Main scripts:**
  - `train_model_accurate.py` — Train a robust CNN with augmentation and class weighting
  - `test_model.py` — Evaluate the trained model on the test set and report per-class results

---

## Requirements
- Python 3.8–3.10 (recommended: use Miniconda)
- TensorFlow 2.10.x (for GPU on Windows)
- CUDA 11.2, cuDNN 8.1 (for GPU support)
- numpy, matplotlib

Install dependencies (if not already):
```
pip install tensorflow==2.10.1 numpy matplotlib
```
Or, for GPU (recommended):
```
conda create -n tf python=3.10
conda activate tf
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1
pip install tensorflow==2.10.1 numpy matplotlib
```

---

## Setup
1. **Download and extract the CASIA-FASD dataset**
   - Organize as:
     ```
     ai/
       casia-fasd/
         train/
           live/
           spoof/
         test/
           live/
           spoof/
     ```
2. **Clone or copy this project into the same `ai/` directory.**
3. **(Optional) Create and activate a virtual environment.**

---

## Training
Run the training script to train a model from scratch:
```
python keras_models/train_model_accurate.py
```
- The script will:
  - Load and augment the training data
  - Compute class weights to handle class imbalance
  - Train a CNN with early stopping and learning rate scheduling
  - Save the best model as `keras_models/best_model.keras`
  - Plot and save training history

**Tips:**
- Training may take 30+ minutes on a GTX 1060 6GB for the full dataset.
- You can adjust `BATCH_SIZE`, `IMAGE_SIZE`, and `EPOCHS` in the script for speed/accuracy tradeoff.

---

## Testing
Evaluate the trained model on the test set:
```
python keras_models/test_model.py
```
- The script will:
  - Load the best trained model
  - Evaluate on the first `NUM_TEST_IMAGES` test images (set at the top of the script)
  - Print, for each class (live/spoof), the number of correct and wrong predictions

**To test on all images:**
- Set `NUM_TEST_IMAGES` in `test_model.py` to the total number of test images (e.g., 65786)

---

## Improving Accuracy
- **Class weights:** Already enabled in `train_model_accurate.py` for better live/spoof balance
- **Augmentation:** The script uses strong augmentation for robustness
- **Model:** You can swap in a pre-trained model (e.g., MobileNetV2) for better results
- **Focal loss:** For highly imbalanced data, consider using focal loss
- **Tune hyperparameters:** Try different batch sizes, learning rates, or more epochs

---

## Troubleshooting
- **GPU not detected:**
  - Ensure you are using TensorFlow 2.10.x and compatible CUDA/cuDNN versions
  - Use `test_gpu.py` to check GPU availability
- **Shape mismatch errors:**
  - Make sure `image_size` in `test_model.py` matches the size used in training
- **Model accuracy is low:**
  - Check for class imbalance, try more augmentation, or use a more powerful model
- **Out of memory:**
  - Lower the batch size in the training script

---

## Contact
For questions or improvements, open an issue or contact the project maintainer. 