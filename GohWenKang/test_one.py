
import os
from PIL import Image
import torch
from vit_antispoofing import ViTAntiSpoofing

# Set your model and image paths here
model_path = "best_vit_model.pth"  # Example path
image_path = "../dataset/casia-fasd/test/live/s6v1f19.png"  # Example path

def predict_image_with_vit(model_path, image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vit_model = ViTAntiSpoofing(device=device)
    if os.path.exists(model_path):
        vit_model.model.load_state_dict(torch.load(model_path, map_location=device))
        vit_model.model.eval()
    else:
        print(f"Model file not found: {model_path}")
        return

    img = Image.open(image_path).convert('RGB')
    transform = vit_model.test_transform
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = vit_model.model(input_tensor)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
        label = "Live" if pred_class == 1 else "Spoof"
        print(f"Prediction: {label} | Confidence: {confidence:.2f}")

if __name__ == "__main__":
    predict_image_with_vit(model_path, image_path)
