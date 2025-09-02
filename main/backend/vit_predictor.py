import torch
import os
import sys
from PIL import Image

# Add the VIT model path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'GohWenKang')))

try:
    from model.VIT_GohWenKang.model import ViTAntiSpoofing
except ImportError:
    # Fallback to import from GohWenKang folder
    from vit_antispoofing import ViTAntiSpoofing

# Model configuration
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'VIT_GohWenKang', 'best_vit_model.pth'))
FALLBACK_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'GohWenKang', 'best_vit_model.pth'))
CLASS_NAMES = ['spoof', 'live']  # VIT uses: 0=spoof, 1=live

def load_vit_model():
    """Load the ViT model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vit_model = ViTAntiSpoofing(device=device)
    
    # Try to load model from main model directory first
    model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_MODEL_PATH
    
    print(f"[DEBUG] Checking for ViT model file at: {model_path}")
    if os.path.exists(model_path):
        try:
            vit_model.model.load_state_dict(torch.load(model_path, map_location=device))
            vit_model.model.eval()
            print(f"[DEBUG] ViT model loaded successfully from: {model_path}")
            return vit_model
        except Exception as e:
            print(f"[ERROR] Failed to load ViT model: {e}")
            return None
    else:
        print(f"[ERROR] ViT model file not found at: {model_path}")
        return None

# Load the model when module is imported
vit_model = load_vit_model()

def predict_image_vit(img: Image.Image):
    """Predict using ViT model"""
    if vit_model is None:
        print("[ERROR] ViT model not loaded!")
        return 0, 0.0
    
    try:
        pred_class, confidence = vit_model.predict(img)
        return pred_class, confidence
    except Exception as e:
        print(f"[ERROR] ViT prediction failed: {e}")
        return 0, 0.0
