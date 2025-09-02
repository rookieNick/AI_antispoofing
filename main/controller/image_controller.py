from backend.predictor import predict_image

def get_prediction_label(img):
    pred = predict_image(img)
    return "Live" if pred == 1 else "Spoof"
