import torch
import os
import sys
from PIL import Image

# Add the VIT model path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'GohWenKang')))

try:
    from model.VIT_GohWenKang.model import ViTAntiSpoofing
except ImportError as e:
    print(f"[DEBUG] Could not import from model directory: {e}")
    try:
        # Add GohWenKang directory to path
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'GohWenKang')))
        from vit_antispoofing import ViTAntiSpoofing
    except ImportError as e:
        print(f"[ERROR] Failed to import ViTAntiSpoofing from either location: {e}")
        raise

# Model configuration
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'VIT_GohWenKang', 'best_vit_model.pth'))
FALLBACK_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'GohWenKang', 'best_vit_model.pth'))
CLASS_NAMES = ['spoof', 'live']  # VIT uses: 0=spoof, 1=live

def load_vit_model():
    """Load the ViT model exactly like GohWenKang test_one.py"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vit_model = ViTAntiSpoofing(device=device)
    
    # Try to load model from main model directory first
    model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_MODEL_PATH
    
    print(f"[DEBUG] Checking for ViT model file at: {model_path}")
    if os.path.exists(model_path):
        try:
            # Load exactly like GohWenKang test_one.py - simple and direct
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                vit_model.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                vit_model.model.load_state_dict(checkpoint) # Fallback if it's just the state_dict
            vit_model.model.eval()
            print(f"[DEBUG] ViT model loaded successfully from: {model_path}")
            return vit_model
        except Exception as e:
            print(f"[ERROR] Failed to load ViT model: {e}")
            print("[INFO] ViT model will not be available for predictions")
            return None
    else:
        print(f"[ERROR] ViT model file not found at: {model_path}")
        return None

# Load the model when module is imported
vit_model = load_vit_model()

def predict_image_vit(img: Image.Image):
    """Predict using ViT model - matches test_one.py pattern"""
    if vit_model is None:
        print("[ERROR] ViT model not loaded!")
        return 0, 0.0
    
    try:
        # Ensure image is RGB before transform
        img_rgb = img.convert("RGB")
        transform = vit_model.test_transform
        input_tensor = transform(img_rgb).unsqueeze(0).to(vit_model.device)
        
        with torch.no_grad():
            outputs = vit_model.model(input_tensor)
            probs = torch.softmax(outputs.logits, dim=1)
            print(f"[DEBUG] ViT softmax probabilities: {probs.cpu().numpy().squeeze()}")
            pred_class = probs.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
            print(f"[DEBUG] Predicted class: {CLASS_NAMES[pred_class]}, Confidence: {confidence}")
            return pred_class, confidence
    except Exception as e:
        print(f"[ERROR] VIT prediction failed: {e}")
        return 0, 0.0


if __name__ == "__main__":
    from PIL import Image
    img = Image.open("backend/phone.png").convert("RGB")
    predict_image_vit(img)