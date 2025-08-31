# Face Anti-Spoofing with PyTorch and CASIA-FASD

## Overview
This project implements a deep learning pipeline for face anti-spoofing (presentation attack detection) using the CASIA-FASD dataset. It uses PyTorch to train and evaluate a convolutional neural network (CNN) to distinguish between real (live) and spoofed faces.

**Dataset:** [CASIA-FASD](http://www.cbsr.ia.ac.cn/english/FASDB_database.asp)
**Framework:** PyTorch (CUDA/cuDNN enabled)
**Main scripts:**
  - `train_fine_tuned.py` — Train a robust CNN with advanced augmentation, class balancing, and fine-tuning
  - `test.py` — Evaluate the trained model on the test set and report per-class results
  - `data_check.py` — Analyze dataset quality, check for corrupted, blurry, or low-brightness images

---

## Requirements
* Python 3.8–3.10 (recommended: use Miniconda or venv)
* PyTorch (with CUDA support)
* CUDA 12.x, cuDNN (for GPU support)
* torchvision, numpy, matplotlib, opencv-python

Install dependencies (if not already):
```
pip install torch torchvision numpy matplotlib opencv-python
```
Or, for GPU (recommended):
```
conda create -n ai_antispoofing python=3.10
conda activate ai_antispoofing
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia
pip install numpy matplotlib opencv-python
```

---

## Setup
1. **Download and extract the CASIA-FASD dataset**
   - Organize as:
     ```
     ai asgmt/
       dataset/
         casia-fasd/
           train/
             live/
             spoof/
           test/
             live/
             spoof/
         data_check.py
         balance_class.py
         count.py
         requirements.txt
       YeohLiXiang/
         train_fine_tuned.py
         test.py
         model.py
         plot_utils.py
         README.md
         requirements.txt
     ```
2. **Clone or copy this project into the same `ai asgmt/` directory.**
3. **(Optional) Create and activate a virtual environment.**

---

## Training
Run the fine-tuned PyTorch training script:
```
python train_fine_tuned.py
```
- The script will:
  - Load and augment the training data with advanced techniques
  - Use WeightedRandomSampler for class balancing (no images are deleted)
  - Train a CNN with early stopping, learning rate scheduling, and mixed precision
  - Save the best model as `model/best_model_fine_tuned.pth`
  - Plot and save training history and metrics

**Tips:**
- Training may take 30+ minutes on a modern GPU for the full dataset.
- You can adjust `BATCH_SIZE`, `IMAGE_SIZE`, and `EPOCHS` in the script for speed/accuracy tradeoff.

---

## Testing
Evaluate the trained model on the test set:
```
python test.py
```
- The script will:
  - Load the best trained model
  - Evaluate on all test images
  - Print, for each class (live/spoof), the number of correct and wrong predictions

---

## Improving Accuracy
- **Class balancing:** WeightedRandomSampler is used for balanced training
- **Augmentation:** The script uses strong and advanced augmentation for robustness
- **Model:** You can swap in a different CNN architecture for better results
- **Focal loss:** For highly imbalanced data, focal loss is available
- **Tune hyperparameters:** Try different batch sizes, learning rates, or more epochs

---

## Troubleshooting
- **GPU not detected:**
  - Ensure you are using PyTorch and compatible CUDA/cuDNN versions
  - Use `torch.cuda.is_available()` to check GPU availability
- **Shape mismatch errors:**
  - Make sure `IMAGE_SIZE` in `test.py` matches the size used in training
- **Model accuracy is low:**
  - Check for class imbalance, try more augmentation, or use a more powerful model
- **Out of memory:**
  - Lower the batch size in the training script

---

## Contact
For questions or improvements, open an issue or contact the project maintainer.