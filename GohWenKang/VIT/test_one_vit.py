# ======================================================================
# Single Image Test Script for ViT Anti-Spoofing Model
# ======================================================================
# This script loads a ViT model and tests it on a single image specified
# by the file path variable in the code.
# ======================================================================

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import os

# PyTorch 2.6 fix for numpy scalar
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])

def load_vit_model(model_path):
    """Load the ViT model from a .pth file."""
    from transformers import ViTForImageClassification
    print(f"Loading model from: {model_path}")
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=2,
        ignore_mismatched_sizes=True,
        cache_dir='VIT/vit_cache'
    )
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Model loaded successfully.")
    return model

def preprocess_image(image_path):
    """Preprocess the image for ViT input."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict(model, image_tensor, device):
    """Run prediction on the image tensor."""
    model.to(device)
    model.eval()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)
        spoof_prob = probabilities[0][0].item()
        live_prob = probabilities[0][1].item()
        prediction = "LIVE" if live_prob > spoof_prob else "SPOOF"
        confidence = max(live_prob, spoof_prob) * 100
    return prediction, confidence, live_prob * 100, spoof_prob * 100

def main():
    # Set the model and image file paths here
    model_path = 'VIT/vit_models/best_vit_model.pth'  # Change as needed
    image_path = '../testCaseImages/side angle face.png'  # Change as needed

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_vit_model(model_path)
    image_tensor = preprocess_image(image_path)
    prediction, confidence, live_prob, spoof_prob = predict(model, image_tensor, device)

    print("\n===== Single Image Test Result =====")
    print(f"Image: {image_path}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Live Probability: {live_prob:.2f}%")
    print(f"Spoof Probability: {spoof_prob:.2f}%")

if __name__ == "__main__":
    main()
