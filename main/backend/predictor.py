import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'CNN_YeohLiXiang')))
from model.CNN_YeohLiXiang.model import OptimizedCNN

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'CNN_YeohLiXiang', 'cnn_pytorch.pth'))
CLASS_NAMES = ['live', 'spoof']
IMAGE_SIZE = (112, 112)

def load_model():
    device = torch.device('cpu')
    num_classes = len(CLASS_NAMES)
    model = OptimizedCNN(num_classes=num_classes).to(device)
    print(f"[DEBUG] Checking for model file at: {MODEL_PATH}")
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    else:
        print(f"Model file not found: {MODEL_PATH}")
        return None

model = load_model()

def predict_image(img: Image.Image):
    if model is None:
        print("[ERROR] Model not loaded!")
        return 0, 0.0
    
    preprocess = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = img.convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        pred_class = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, pred_class].item()
        return pred_class, confidence
