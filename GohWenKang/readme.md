## 🛠️ Setup Instructions

To Run the Test Script, please follow these steps:

## Model Weights (If GohWenKang/VIT/vit_models/best_vit_model.pth not found)
Download the required `.pth` files from [https://drive.google.com/file/d/1aZ4dkmbHn1Ns7cKJ7ND7JOK43B4CR_ln/view?usp=sharing] and place them in `GohWenKang/VIT/vit_models/`.

### 1. Deactivate Any Existing Virtual Environment

If you have a virtual environment active, deactivate it first:
```bash
deactivate
```

### 2. Use Python 3.10.8 or Above

Make sure you have Python 3.10.8 or newer installed.

### 3. Create and Activate a New Virtual Environment

Navigate to the `GohWenKang` directory:
```bash
cd GohWenKang
```

Create a new virtual environment:
```bash
py -3.10 -m venv venv_wk
```

Activate the virtual environment:
```bash
venv_wk\Scripts\activate
```

Upgrade pip:
```bash
python -m pip install --upgrade pip
```

### 4. Install Required Packages

Install PyTorch (with CUDA 11.8 support):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install other dependencies:
```bash
pip install transformers timm pillow scikit-learn matplotlib seaborn tqdm opencv-python numpy pandas
```

### 5. (Optional) Verify Installation

Check that PyTorch is installed and CUDA is available:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 6. Run the Test Script

Make sure you are in the `GohWenKang` directory, then run:
```bash
python VIT\test_one_vit.py

python resnet\resnet_gui.py

python efficientNet\efficientnet_meta_gui.py
```

### 7. Run Training Script

To train the model, first navigate to the parent directory:
```bash
cd ../
```
Then run the training script:
```bash
python GohWenKang/VIT/train_vit.py

python GohWenKang\resnet\resnet_train.py

python GohWenKang\efficientNet\Train_EfficientNet_Meta.py
```