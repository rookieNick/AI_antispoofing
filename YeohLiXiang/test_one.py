import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from model import OptimizedCNN

# Configuration
MODEL_FILENAME = 'cnn_pytorch.pth'
CLASS_NAMES = ['live', 'spoof']
IMAGE_SIZE = (112, 112)

def predict_image(image_path, model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, "model")
        model_path = os.path.join(model_dir, MODEL_FILENAME)
    
    # Load model
    num_classes = len(CLASS_NAMES)
    model = OptimizedCNN(num_classes=num_classes).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        print(f"Model file not found: {model_path}")
        return None
    
    # Image preprocessing
    test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    input_tensor = test_transform(img).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        pred_class = probabilities.argmax(dim=1).item()
        pred_label = CLASS_NAMES[pred_class]
        confidence = probabilities[0, pred_class].item()
    print(f"Prediction: {pred_label} (confidence: {confidence:.4f})")
    return pred_label, confidence

if __name__ == "__main__":
    # Specify your image path here
    image_path = "../dataset/casia-fasd/test/spoof/s30vHR_4f259.png"  # <-- Change this to your image file
    predict_image(image_path)
