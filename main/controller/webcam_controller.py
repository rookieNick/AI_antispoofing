"""
Webcam Controller
Manages webcam operations and coordinates with anti-spoofing detector
"""
import cv2
import threading
import time
from queue import Queue, Empty
import numpy as np

class WebcamController:
    """Controls webcam operations and frame processing"""
    
    def __init__(self, detector, width=640, height=480):
        self.detector = detector
        self.width = width
        self.height = height
        
        # Webcam state
        self.cap = None
        self.is_active = False
        self.is_processing = False
        
        # Threading and queues
        self.raw_frame_queue = Queue(maxsize=2)  # Reduced size to prevent lag
        self.processed_frame_queue = Queue(maxsize=1)
        self.worker_thread = None
        
        # Current model
        self.current_model = "CNN"
        
        # Frame rate control
        self.last_process_time = 0
        self.process_interval = 0.1  # Process every 100ms to reduce refreshing
    
    def start_webcam(self):
        """Start webcam capture"""
        if self.is_active:
            return True, "Webcam already active"
        
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                return False, "Could not open webcam"
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 10)  # Limit FPS to reduce processing load
            
            self.is_active = True
            return True, "Webcam started successfully"
            
        except Exception as e:
            return False, f"Error starting webcam: {str(e)}"
    
    def stop_webcam(self):
        """Stop webcam capture"""
        if not self.is_active:
            return True, "Webcam already stopped"
        
        try:
            self.is_active = False
            
            if self.cap:
                self.cap.release()
                self.cap = None
            
            return True, "Webcam stopped successfully"
            
        except Exception as e:
            return False, f"Error stopping webcam: {str(e)}"
    
    def start_processing(self, model_type="CNN"):
        """Start frame processing"""
        if self.is_processing:
            return True, "Processing already active"
        
        if not self.is_active:
            return False, "Webcam not active"
        
        try:
            self.current_model = model_type
            self.is_processing = True
            
            # Clear queues
            self._clear_queues()
            
            # Start worker thread
            self.worker_thread = threading.Thread(target=self._processing_worker, daemon=True)
            self.worker_thread.start()
            
            return True, f"Processing started with {model_type} model"
            
        except Exception as e:
            self.is_processing = False
            return False, f"Error starting processing: {str(e)}"
    
    def stop_processing(self):
        """Stop frame processing"""
        if not self.is_processing:
            return True, "Processing already stopped"
        
        try:
            self.is_processing = False
            
            # Wait for worker thread to finish
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=1.0)
            
            # Clear queues
            self._clear_queues()
            
            return True, "Processing stopped successfully"
            
        except Exception as e:
            return False, f"Error stopping processing: {str(e)}"
    
    def switch_model(self, model_type):
        """Switch the current model"""
        if not self.is_processing:
            return False, "Processing not active"
        
        try:
            old_model = self.current_model
            self.current_model = model_type
            return True, f"Model switched from {old_model} to {model_type}"
            
        except Exception as e:
            return False, f"Error switching model: {str(e)}"
    
    def get_frame(self):
        """Get current frame from webcam"""
        if not self.is_active or not self.cap:
            return None, None
        
        try:
            ret, frame = self.cap.read()
            if ret:
                return frame, None
            else:
                return None, "Could not read frame"
                
        except Exception as e:
            return None, f"Error reading frame: {str(e)}"
    
    def get_processed_frame(self):
        """Get the latest processed frame with results"""
        try:
            if not self.processed_frame_queue.empty():
                return self.processed_frame_queue.get_nowait()
            return None
            
        except Empty:
            return None
        except Exception as e:
            print(f"Error getting processed frame: {e}")
            return None
    
    def put_frame_for_processing(self, frame):
        """Put frame in queue for processing"""
        try:
            # Only add frame if enough time has passed (rate limiting)
            current_time = time.time()
            if current_time - self.last_process_time < self.process_interval:
                return
            
            if not self.raw_frame_queue.full():
                self.raw_frame_queue.put_nowait(frame.copy())
                self.last_process_time = current_time
                
        except Exception as e:
            print(f"Error putting frame for processing: {e}")
    
    def _processing_worker(self):
        """Worker thread for processing frames with face detection"""
        print(f"Processing worker started with {self.current_model} model")
        
        while self.is_processing:
            try:
                if not self.raw_frame_queue.empty():
                    frame = self.raw_frame_queue.get_nowait()
                    
                    # Process frame with face detection and padding
                    results = self.detector.process_frame(frame, self.current_model, padding=100, detect_faces=True)
                    
                    # Put processed frame in output queue
                    if not self.processed_frame_queue.full():
                        try:
                            # Remove old processed frame if queue is full
                            while not self.processed_frame_queue.empty():
                                self.processed_frame_queue.get_nowait()
                        except Empty:
                            pass
                        
                        processed_data = {
                            'frame': frame,
                            'results': results,
                            'model_type': self.current_model
                        }
                        self.processed_frame_queue.put_nowait(processed_data)
                else:
                    time.sleep(0.01)  # Small delay when no frames to process
                    
            except Empty:
                time.sleep(0.01)
            except Exception as e:
                print(f"Processing worker error: {e}")
                time.sleep(0.1)
        
        print("Processing worker stopped")
    
    def _clear_queues(self):
        """Clear all queues"""
        try:
            while not self.raw_frame_queue.empty():
                self.raw_frame_queue.get_nowait()
        except Empty:
            pass
        
        try:
            while not self.processed_frame_queue.empty():
                self.processed_frame_queue.get_nowait()
        except Empty:
            pass
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_processing()
        self.stop_webcam()
        self._clear_queues()
        
        if self.detector:
            self.detector.cleanup()
    
    def get_status(self):
        """Get current status"""
        return {
            'webcam_active': self.is_active,
            'processing_active': self.is_processing,
            'current_model': self.current_model,
            'raw_queue_size': self.raw_frame_queue.qsize(),
            'processed_queue_size': self.processed_frame_queue.qsize()
        }
