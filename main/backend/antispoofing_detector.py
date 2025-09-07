"""from
Anti-Spoofing Detector Backend
Handles face detection and anti-spoofing prediction logic
"""
import cv2
import numpy as np
from PIL import Image
import os
import sys

# Import predictor functions
from .predictor import predict_image as predict_cnn_image
from .cdcn_predictor import predict_image as predict_cdcn_image  
from .vit_predictor import predict_image_vit

import insightface
from insightface.app import FaceAnalysis

class AntiSpoofingDetector:
    """Main anti-spoofing detector class"""
    
    def __init__(self, detection_size=(640, 640), detection_threshold=0.7, face_confidence_threshold=0.7):
        self.detection_size = detection_size
        self.detection_threshold = detection_threshold
        self.face_confidence_threshold = face_confidence_threshold
        
        # Initialize face detection
        self._face_app = None
    
    def get_face_app(self):
        """Initialize face detection app"""
        if self._face_app is None:
            self._face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self._face_app.prepare(ctx_id=0, det_size=self.detection_size, det_thresh=self.detection_threshold)
        return self._face_app
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        face_app = self.get_face_app()
        faces = face_app.get(frame)
        return faces
    
    def crop_face_with_padding(self, frame, bbox, padding=100):
        """Crop face from frame with padding (minimum 100px)"""
        x1, y1, x2, y2 = bbox.astype(int)
        h, w = frame.shape[:2]
        
        # Ensure minimum padding of 100px
        padding = max(100, padding)
        
        # Calculate face dimensions
        face_width = x2 - x1
        face_height = y2 - y1
        
        # Calculate padding based on face size, but ensure minimum 100px
        dynamic_padding = max(padding, int(max(face_width, face_height) * 0.3))
        
        # Add padding and ensure bounds
        x1_padded = max(0, x1 - dynamic_padding)
        y1_padded = max(0, y1 - dynamic_padding)
        x2_padded = min(w, x2 + dynamic_padding)
        y2_padded = min(h, y2 + dynamic_padding)
        
        # Ensure we don't exceed webcam frame size
        x1_padded = max(0, min(x1_padded, w))
        y1_padded = max(0, min(y1_padded, h))
        x2_padded = max(0, min(x2_padded, w))
        y2_padded = max(0, min(y2_padded, h))
        
        cropped_face = frame[y1_padded:y2_padded, x1_padded:x2_padded]
        return cropped_face, (x1_padded, y1_padded, x2_padded, y2_padded)
    
    def predict_cnn(self, face_img):
        """Predict using CNN model via predictor module"""
        try:
            # Convert to PIL Image if needed
            if isinstance(face_img, np.ndarray):
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_img = Image.fromarray(face_img)
            
            # Use external CNN predictor
            pred_class, confidence = predict_cnn_image(face_img)
            
            # CNN: CLASS_NAMES = ['live', 'spoof'], inverted so 0=spoof, 1=live
            is_real = pred_class == 1  # 0 = spoof, 1 = live
            print(f"[DEBUG CNN] pred_class: {pred_class}, is_real: {is_real}, confidence: {confidence}")
            return is_real, confidence
                
        except Exception as e:
            print(f"CNN prediction error: {e}")
            return False, 0.0
    
    def predict_cdcn(self, face_img):
        """Predict using CDCN model via predictor module"""
        try:
            # Convert to PIL Image if needed
            if isinstance(face_img, np.ndarray):
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_img = Image.fromarray(face_img)
            
            # Use external CDCN predictor
            pred_class, confidence = predict_cdcn_image(face_img)
            
            # CDCN: CLASS_NAMES = ['live', 'spoof'], inverted so 0=spoof, 1=live
            is_real = pred_class == 1  # 0 = spoof, 1 = live
            print(f"[DEBUG CDCN] pred_class: {pred_class}, is_real: {is_real}, confidence: {confidence}")
            return is_real, confidence
                
        except Exception as e:
            print(f"CDCN prediction error: {e}")
            return False, 0.0
    
    def predict_vit(self, face_img):
        """Predict using VIT model via vit_predictor"""
        try:
            # Convert to PIL Image if needed
            if isinstance(face_img, np.ndarray):
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_img = Image.fromarray(face_img)
            
            # Use vit_predictor which matches test_one.py exactly
            pred_class, confidence = predict_image_vit(face_img)
            
            # VIT: 0=spoof, 1=live (matches test_one.py logic)
            is_real = pred_class == 1
            print(f"[DEBUG VIT] pred_class: {pred_class}, is_real: {is_real}, confidence: {confidence}")
            return is_real, confidence
            
        except Exception as e:
            print(f"VIT prediction error: {e}")
            return False, 0.0
    
    def predict(self, face_img, model_type="CNN"):
        """Main prediction method"""
        if model_type == "CNN":
            return self.predict_cnn(face_img)
        elif model_type == "CDCN":
            return self.predict_cdcn(face_img)
        elif model_type == "VIT":
            return self.predict_vit(face_img)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def process_frame(self, frame, model_type="CNN", padding=20, detect_faces=True):
        """Process a single frame for anti-spoofing detection
        Args:
            frame: Input frame/image
            model_type: Model to use for prediction
            padding: Padding around detected faces (only used if detect_faces=True)
            detect_faces: If True, detect faces first. If False, use whole image
        """
        results = []
        
        try:
            if detect_faces:
                # Original behavior - detect faces first
                faces = self.detect_faces(frame)
                
                for face in faces:
                    if face.det_score < self.face_confidence_threshold:
                        continue
                    
                    # Crop face with padding
                    cropped_face, padded_bbox = self.crop_face_with_padding(frame, face.bbox, padding)
                    
                    if cropped_face.size == 0:
                        continue
                    
                    # Predict
                    is_real, confidence = self.predict(cropped_face, model_type)
                    
                    # Original bbox for drawing
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    w, h = x2 - x1, y2 - y1
                    
                    result = {
                        'bbox': (x1, y1, w, h),
                        'padded_bbox': padded_bbox,
                        'recognized_identity': 'Real' if is_real else 'Spoof Detected',
                        'similarity': confidence * 100,
                        'is_spoof': not is_real,
                        'confidence': confidence,
                        'model_type': model_type
                    }
                    results.append(result)
            else:
                # New behavior - use whole image
                h, w = frame.shape[:2]
                
                # Predict on whole image
                is_real, confidence = self.predict(frame, model_type)
                
                result = {
                    'bbox': (0, 0, w, h),  # Full image bbox
                    'padded_bbox': (0, 0, w, h),
                    'recognized_identity': 'Real' if is_real else 'Spoof Detected',
                    'similarity': confidence * 100,
                    'is_spoof': not is_real,
                    'confidence': confidence,
                    'model_type': model_type
                }
                results.append(result)
                
        except Exception as e:
            print(f"Frame processing error: {e}")
        
        return results
    
    def cleanup(self):
        """Clean up resources"""
        self._face_app = None
