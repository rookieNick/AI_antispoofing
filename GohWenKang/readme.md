to train model or use test_one_vit
complete below task first


use python 3.10.8

py -3.10 -m venv vit-antispoofing
vit-antispoofing\Scripts\activate
python -m pip install --upgrade pip

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers timm pillow scikit-learn matplotlib seaborn tqdm opencv-python numpy pandas


verify installation (optional)
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
