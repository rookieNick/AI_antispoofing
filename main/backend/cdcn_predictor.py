import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.CDCN_YeongChingZhou.model import AdvancedCDCN

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'CDCN_YeongChingZhou', 'advanced_cdcn_best.pth'))
CLASS_NAMES = ['live', 'spoof']
IMAGE_SIZE = (224, 224)  # CDCN uses 224x224 input size like test_one.py

def load_checkpoint(model, chk_path, device):
    """Load checkpoint exactly like YeongChingZhou test_one.py"""
    if not os.path.exists(chk_path):
        raise FileNotFoundError(f"Checkpoint not found: {chk_path}")
    ckpt = torch.load(chk_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
    model.load_state_dict(state)

def load_model():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Build model exactly like YeongChingZhou test_one.py
        model = AdvancedCDCN(num_classes=2, theta=0.7, map_size=32, dropout_rate=0.5)
        model = model.to(device)
        
        print(f"[DEBUG] Loading CDCN checkpoint: {MODEL_PATH}")
        load_checkpoint(model, MODEL_PATH, device)
        model.eval()
        print(f"[DEBUG] CDCN model loaded successfully")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load CDCN model: {e}")
        print("[INFO] CDCN model will not be available for predictions")
        return None

model = load_model()

def predict_image(img: Image.Image):
    if model is None:
        print("[ERROR] CDCN model not loaded!")
        return 0, 0.0
    
    # Prepare image exactly like YeongChingZhou test_one.py
    preprocess = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = img.convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0)
    
    with torch.no_grad():
        try:
            # CDCN model returns (classification_output, depth_map)
            cls_output, depth_map = model(input_tensor)
            probs = torch.softmax(cls_output, dim=1).cpu().numpy()[0]
            pred = int(torch.argmax(cls_output, dim=1).cpu().item())
            confidence = probs[pred]
            return pred, confidence
        except Exception as e:
            print(f"[ERROR] CDCN prediction failed: {e}")
            return 0, 0.0
