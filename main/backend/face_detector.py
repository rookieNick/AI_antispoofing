"""
Face Detection Backend
Handles face detection using InsightFace
"""
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self, detection_size=(640, 640), detection_threshold=0.7):
        self.detection_size = detection_size
        self.detection_threshold = detection_threshold
        self._face_app = None
    
    def get_face_app(self):
        """Lazy load face detection model"""
        if self._face_app is None:
            self._face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self._face_app.prepare(ctx_id=0, det_size=self.detection_size,
                                 det_thresh=self.detection_threshold)
        return self._face_app
    
    def detect_faces(self, frame):
        """
        Detect faces in frame
        Args:
            frame: OpenCV image (BGR format)
        Returns:
            list of face objects with bbox and det_score
        """
        try:
            face_app = self.get_face_app()
            faces = face_app.get(frame)
            return faces
        except Exception as e:
            print(f"Face detection error: {e}")
            return []
    
    def crop_face_with_padding(self, frame, bbox, padding=20):
        """
        Crop face from frame with padding
        Args:
            frame: OpenCV image
            bbox: bounding box [x1, y1, x2, y2]
            padding: padding pixels around face
        Returns:
            cropped face image
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Add padding
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(w, x2 + padding)
        y2_pad = min(h, y2 + padding)
        
        # Crop face with padding
        cropped_face = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        return cropped_face
    
    def unload(self):
        """Unload face detection model"""
        self._face_app = None
