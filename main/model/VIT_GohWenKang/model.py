import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from torchvision import transforms
from PIL import Image
import os

class ViTAntiSpoofing:
    def __init__(self, model_name='google/vit-base-patch16-224', num_classes=2, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Performance optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
        
        # Initialize ViT model
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        self.model.to(self.device)
        
        # Data transforms
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path):
        """Load trained model weights"""
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            return True
        return False
    
    def predict(self, image):
        """Predict single image"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif hasattr(image, 'convert'):
            image = image.convert('RGB')
        
        # Preprocess image
        input_tensor = self.test_transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_class = probs.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
            
        return pred_class, confidence
