from backend.predictor import predict_image as predict_cnn
from backend.vit_predictor import predict_image_vit as predict_vit
from backend.cdcn_predictor import predict_image as predict_cdcn

def get_prediction_label(img, model_type="CNN"):
    """Get prediction label from the specified model.
    
    Args:
        img: PIL Image to predict
        model_type: One of "CNN", "CDCN", or "VIT"
        
    Returns:
        tuple: (class_label, confidence) where class_label is "Live" or "Spoof"
    """
    try:
        if model_type == "CNN":
            pred_class, confidence = predict_cnn(img)
        elif model_type == "CDCN":
            pred_class, confidence = predict_cdcn(img)
        elif model_type == "VIT":
            pred_class, confidence = predict_vit(img)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if model_type in ["CNN", "CDCN"]:
            label = "Live" if pred_class == 0 else "Spoof"
        else:  # VIT
            label = "Live" if pred_class == 1 else "Spoof"
        return label, confidence
    except Exception as e:
        print(f"[ERROR] Prediction failed for {model_type}: {e}")
        # Return default values if prediction fails
        return "Error", 0.0
