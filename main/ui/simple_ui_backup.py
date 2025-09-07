# Import the necessary libraries
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import os
import threading
import time
import numpy as np
from queue import Queue
import insightface
from insightface.app import FaceAnalysis
import torch
import torchvision.transforms as transforms
from PIL import Image as PILImage

# Import anti-spoofing models
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the correct model classes based on predictor files
from model.CNN_YeohLiXiang.model import OptimizedCNN
from model.CDCN_YeongChingZhou.optimized_cdcn import OptimizedCDCN
from model.VIT_GohWenKang.model import ViTAntiSpoofing

# Global variables
cap = None
photo = None
recognition_active = False
webcam_active = False
selected_image = None
# Queue for processed frames with results
processed_frame_queue = Queue(maxsize=1)
# Queue for raw frames to be processed by worker thread
raw_frame_queue = Queue(maxsize=1)

# Configuration constants
CONFIG = {
    'model': {
        'detection_size': (640, 640),
        'detection_threshold': 0.7,
        'face_confidence_threshold': 0.7,
    },
    'webcam': {
        'width': 640,
        'height': 480,
    }
}

# Global state variables
state = {
    'webcam_in_use': False,
    'processing_active': True,
    'recognition_results': {},
    'recognition_lock': threading.Lock(),
    'current_model': None # To store the currently selected anti-spoofing model instance
}

# Module-level variables for lazy-loaded models
_face_app = None
_cnn_model = None
_cdcn_model = None
_vit_model = None

# Model paths (following predictor.py pattern)
CNN_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'CNN_YeohLiXiang', 'cnn_pytorch.pth')
CDCN_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'CDCN_YeongChingZhou', 'optimized_cdcn_best.pth')
VIT_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'VIT_GohWenKang', 'best_vit_model.pth')

class CNNModel:
    """Wrapper class for OptimizedCNN model"""
    def __init__(self):
        self.device = torch.device('cpu')
        self.model = OptimizedCNN(num_classes=2).to(self.device)
        self._load_weights()
        self.model.eval()
        
        # Preprocessing transform (following predictor.py)
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _load_weights(self):
        """Load model weights following predictor.py pattern"""
        if os.path.exists(CNN_MODEL_PATH):
            try:
                state_dict = torch.load(CNN_MODEL_PATH, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print("CNN model weights loaded successfully")
            except Exception as e:
                print(f"Failed to load CNN model weights: {e}")
    
    def analyze(self, face_img):
        """Analyze face image and return (is_real, confidence)"""
        try:
            # Convert numpy array to PIL Image
            if isinstance(face_img, np.ndarray):
                pil_img = PILImage.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            else:
                pil_img = face_img
            
            # Preprocess
            input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                pred_class = probabilities.argmax(dim=1).item()
                confidence = probabilities[0, pred_class].item()
                
                # Return (is_real, confidence) - assuming class 0 is spoof, 1 is real
                is_real = pred_class == 1
                return is_real, confidence
                
        except Exception as e:
            print(f"CNN analysis error: {e}")
            return False, 0.0

class CDCNModel:
    """Wrapper class for OptimizedCDCN model"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = OptimizedCDCN(num_classes=2).to(self.device)
        self._load_weights()
        self.model.eval()
        
        # Preprocessing transform (following cdcn_predictor.py)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _load_weights(self):
        """Load model weights following cdcn_predictor.py pattern"""
        if os.path.exists(CDCN_MODEL_PATH):
            try:
                # Handle PyTorch checkpoint loading
                ckpt = torch.load(CDCN_MODEL_PATH, map_location=self.device, weights_only=False)
                if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                    state_dict = ckpt['model_state_dict']
                else:
                    state_dict = ckpt
                self.model.load_state_dict(state_dict)
                print("CDCN model weights loaded successfully")
            except Exception as e:
                print(f"Failed to load CDCN model weights: {e}")
    
    def analyze(self, face_img):
        """Analyze face image and return (is_real, confidence)"""
        try:
            # Convert numpy array to PIL Image
            if isinstance(face_img, np.ndarray):
                pil_img = PILImage.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            else:
                pil_img = face_img
            
            # Preprocess
            input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                cls_output = self.model(input_tensor)
                probs = torch.softmax(cls_output, dim=1).cpu().numpy()[0]
                pred_class = int(torch.argmax(cls_output, dim=1).cpu().item())
                confidence = probs[pred_class]
                
                # Return (is_real, confidence) - assuming class 0 is spoof, 1 is real
                is_real = pred_class == 1
                return is_real, confidence
                
        except Exception as e:
            print(f"CDCN analysis error: {e}")
            return False, 0.0

class VITModel:
    """Wrapper class for ViTAntiSpoofing model"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ViTAntiSpoofing(device=self.device)
        self._load_weights()
    
    def _load_weights(self):
        """Load model weights following vit_predictor.py pattern"""
        if os.path.exists(VIT_MODEL_PATH):
            try:
                checkpoint = torch.load(VIT_MODEL_PATH, map_location=self.device, weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.model.load_state_dict(checkpoint)
                print("VIT model weights loaded successfully")
            except Exception as e:
                print(f"Failed to load VIT model weights: {e}")
    
    def analyze(self, face_img):
        """Analyze face image and return (is_real, confidence)"""
        try:
            # Convert numpy array to PIL Image
            if isinstance(face_img, np.ndarray):
                pil_img = PILImage.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            else:
                pil_img = face_img
            
            # Use the model's predict method
            pred_class, confidence = self.model.predict(pil_img)
            
            # Return (is_real, confidence) - VIT uses: 0=spoof, 1=live
            is_real = pred_class == 1
            return is_real, confidence
            
        except Exception as e:
            print(f"VIT analysis error: {e}")
            return False, 0.0

def get_face_app():
    global _face_app
    if _face_app is None:
        _face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        _face_app.prepare(ctx_id=0, det_size=CONFIG['model']['detection_size'],
                         det_thresh=CONFIG['model']['detection_threshold'])
    return _face_app

def get_cnn_model():
    global _cnn_model
    if _cnn_model is None:
        _cnn_model = CNNModel()
    return _cnn_model

def get_cdcn_model():
    global _cdcn_model
    if _cdcn_model is None:
        _cdcn_model = CDCNModel()
    return _cdcn_model

def get_vit_model():
    global _vit_model
    if _vit_model is None:
        _vit_model = VITModel()
    return _vit_model

def load_anti_spoofing_model(model_type):
    """Loads the specified anti-spoofing model."""
    if model_type == "CNN":
        return get_cnn_model()
    elif model_type == "CDCN":
        return get_cdcn_model()
    elif model_type == "VIT":
        return get_vit_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def draw_recognition_results(display_image, result):
    """
    Draw recognition results on the display image
    Args:
        display_image: Image to draw on
        result: Recognition result dictionary
    """
    x, y, w, h = result['bbox']
    label = result['recognized_identity']
    confidence = result.get('similarity', 0)
    is_spoof = result.get('is_spoof', False)
    
    if is_spoof:
        color = (0, 0, 255)  # Red for spoof attempts
        text = f"Spoof Detected! ({confidence:.1f}%)"
    else:
        color = (0, 255, 0)  # Green for real faces
        text = f"Real ({confidence:.1f}%)"
    
    cv2.rectangle(display_image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(display_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def display_status(display_image):
    """
    Display status information on the image
    Args:
        display_image: Image to draw on
    """
    cv2.putText(display_image, "Press 'ESC' to quit",
                (10, display_image.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def clear_memory():
    """
    Clear unnecessary data from memory
    Also uninitializes models when not in use
    """
    global _face_app, _cnn_model, _cdcn_model, _vit_model
    
    with state['recognition_lock']:
        state['recognition_results'].clear()
    
    while not raw_frame_queue.empty():
        try:
            raw_frame_queue.get_nowait()
        except:
            break
    while not processed_frame_queue.empty():
        try:
            processed_frame_queue.get_nowait()
        except:
            break
    
    if not state['webcam_in_use']:
        if _face_app is not None:
            try:
                _face_app = None
            except Exception as e:
                print(f"Error unloading face app: {e}")
                
        # Unload all anti-spoofing models
        _cnn_model = None
        _cdcn_model = None
        _vit_model = None
    
    print("Memory cleared and models unloaded")
    
    import gc
    gc.collect()
    cv2.destroyAllWindows()
    
    state['webcam_in_use'] = False
    state['processing_active'] = False
    state['recognition_results'].clear()

def start_webcam():
    global cap, webcam_active
    if not webcam_active:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            result_output.insert(tk.END, "Error: Could not open webcam.\n")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['webcam']['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['webcam']['height'])

        webcam_active = True
        state['webcam_in_use'] = True
        update_frame()
        result_output.insert(tk.END, "Webcam started.\n")

def stop_webcam():
    global cap, webcam_active
    if webcam_active:
        cap.release()
        webcam_active = False
        state['webcam_in_use'] = False
        webcam_label.config(image='')
        result_output.insert(tk.END, "Webcam stopped.\n")
        clear_memory()

def update_frame():
    global cap, photo
    if webcam_active and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Put raw frame into queue for processing
            if not raw_frame_queue.full():
                try:
                    raw_frame_queue.put_nowait(frame.copy())
                except:
                    pass

            display_image = frame.copy()

            # Get the latest processed frame with results from the queue
            if not processed_frame_queue.empty():
                processed_frame_data = processed_frame_queue.get_nowait()
                display_image = processed_frame_data['frame']
                results_to_draw = processed_frame_data['results']
                
                for face_key, face_result in results_to_draw:
                    draw_recognition_results(display_image, face_result)
                
                display_status(display_image)

            cv2image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            img.thumbnail((480, 360), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image=img)
            webcam_label.config(image=photo)
            webcam_label.image = photo
            root.after(10, update_frame)
        else:
            result_output.insert(tk.END, "Error: Could not read frame from webcam.\n")
    elif not webcam_active:
        webcam_label.config(image='')

def detection_and_spoofing_worker():
    print("Detection and Spoofing worker thread started")
    face_app = get_face_app()
    
    while state['processing_active']:
        try:
            if not raw_frame_queue.empty():
                frame = raw_frame_queue.get()
                
                faces = face_app.get(frame)
                
                current_frame_results = []
                for i, face in enumerate(faces, 1):
                    if face.det_score < CONFIG['model']['face_confidence_threshold']:
                        continue
                    
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    w, h = x2 - x1, y2 - y1
                    
                    # Crop face for anti-spoofing model
                    face_img_for_spoof = frame[y1:y2, x1:x2]
                    
                    is_real, spoof_confidence = False, 0.0
                    if state['current_model']:
                        # Assuming analyze method takes a cropped face image
                        is_real, spoof_confidence = state['current_model'].analyze(face_img_for_spoof)
                    
                    result = {
                        'bbox': (x1, y1, w, h),
                        'recognized_identity': 'Real' if is_real else 'Spoof Detected',
                        'similarity': spoof_confidence * 100,
                        'is_spoof': not is_real
                    }
                    current_frame_results.append((f"{x1}_{y1}_{w}_{h}", result))

                if not processed_frame_queue.full():
                    processed_frame_queue.put_nowait({
                        'frame': frame,
                        'results': current_frame_results
                    })
                
                raw_frame_queue.task_done()
            else:
                time.sleep(0.01)
        except Exception as e:
            print(f"Detection and Spoofing worker thread error: {str(e)}")
            time.sleep(0.1)

def start_recognition():
    global recognition_active
    if not recognition_active:
        recognition_active = True
        state['processing_active'] = True
        result_output.insert(tk.END, "Anti-Spoofing Recognition started.\n")
        
        # Load the selected anti-spoofing model
        try:
            model_type = model_var.get()
            state['current_model'] = load_anti_spoofing_model(model_type)
            state['current_model_type'] = model_type
            result_output.insert(tk.END, f"Loaded {model_type} model for anti-spoofing.\n")
        except Exception as e:
            result_output.insert(tk.END, f"Error loading anti-spoofing model: {e}\n")
            recognition_active = False
            state['processing_active'] = False
            return

        get_face_app() # Initialize InsightFace
        
        worker_thread = threading.Thread(target=detection_and_spoofing_worker, daemon=True)
        worker_thread.start()
    else:
        result_output.insert(tk.END, "Anti-Spoofing Recognition is already active.\n")

def stop_recognition():
    global recognition_active
    if recognition_active:
        recognition_active = False
        state['processing_active'] = False
        result_output.insert(tk.END, "Anti-Spoofing Recognition stopped.\n")
        
        while not raw_frame_queue.empty():
            try:
                raw_frame_queue.get_nowait()
            except:
                break
        while not processed_frame_queue.empty():
            try:
                processed_frame_queue.get_nowait()
            except:
                break
        state['current_model'] = None # Unload the current model
    else:
        result_output.insert(tk.END, "Anti-Spoofing Recognition is not active.\n")

def import_image():
    file_path = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.gif *.bmp"), ("All Files", "*")]
    )
    if file_path:
        result_output.insert(tk.END, f"Imported image: {file_path}\n")
        try:
            img = Image.open(file_path)
            img.thumbnail((200, 200), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            image_canvas.delete("all")
            image_canvas.create_image(100, 100, image=img_tk)
            image_canvas.image = img_tk
            global selected_image
            selected_image = img
        except Exception as e:
            result_output.insert(tk.END, f"Error loading image: {e}\n")

def predict_image_with_selected_model():
    global selected_image, static_model_var
    if selected_image is None:
        result_output.insert(tk.END, "No image selected. Please import an image first.\n")
        return
    
    try:
        model_type = static_model_var.get()
        # For image prediction, we'll use the selected anti-spoofing model
        model_instance = load_anti_spoofing_model(model_type)
        
        # Convert PIL Image to OpenCV format for model input
        opencv_image = cv2.cvtColor(np.array(selected_image), cv2.COLOR_RGB2BGR)
        
        # Perform face detection to get bounding box
        face_app = get_face_app()
        faces = face_app.get(opencv_image)
        
        if len(faces) == 0:
            result_output.insert(tk.END, "No face detected in the imported image.\n")
            return
        
        # Assuming we only care about the first detected face for static image prediction
        face = faces[0]
        x1, y1, x2, y2 = face.bbox.astype(int)
        cropped_face = opencv_image[y1:y2, x1:x2]
        
        is_real, confidence = model_instance.analyze(cropped_face)
        label = "Real" if is_real else "Spoof Detected"
        result_output.insert(tk.END, f"Static Image Prediction ({model_type}): {label} | Confidence: {confidence:.2f}\n")
    except Exception as e:
        result_output.insert(tk.END, f"Error during prediction with {static_model_var.get()}: {str(e)}\n")


def switch_webcam_model():
    """Switch the model used for webcam prediction during operation"""
    if not recognition_active:
        result_output.insert(tk.END, "Please start webcam recognition first.\n")
        return

    try:
        model_type = model_var.get()
        old_model = state.get('current_model_type', 'None')
        state['current_model'] = load_anti_spoofing_model(model_type)
        state['current_model_type'] = model_type
        result_output.insert(tk.END, f"Switched from {old_model} to {model_type} model for webcam prediction.\n")
    except Exception as e:
        result_output.insert(tk.END, f"Error switching model: {str(e)}\n")


def start_webcam_and_recognition():
    start_webcam()
    start_recognition()

def stop_webcam_and_recognition():
    stop_webcam()
    stop_recognition()


def launch_ui():
    global root, result_output, webcam_label, image_canvas, start_combined_button, import_image_button, model_var, static_model_var, selected_image
    root = tk.Tk()
    root.title("Anti-Spoofing UI")
    root.geometry("1200x700")
    root.resizable(False, False)

    style = ttk.Style()
    style.theme_use('clam')

    # Main frames
    left_frame = ttk.Frame(root, padding="10")
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    right_frame = ttk.Frame(root, padding="10")
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)


    # Webcam Display Frame (Left, larger)
    webcam_display_frame = ttk.LabelFrame(left_frame, text="Webcam Feed", padding="10")
    webcam_display_frame.pack(fill=tk.BOTH, expand=True)

    webcam_label = ttk.Label(webcam_display_frame, background="black")
    webcam_label.pack(fill=tk.BOTH, expand=True)

    # Model Selection for Webcam (moved to left side, above webcam controls)
    webcam_model_frame = ttk.LabelFrame(left_frame, text="Webcam Model Selection", padding="10")
    webcam_model_frame.pack(fill=tk.X, pady=5)

    # Instruction label
    ttk.Label(webcam_model_frame, text="Select model for real-time webcam prediction:",
              font=("Arial", 9)).pack(anchor=tk.W)

    model_var = tk.StringVar(value="CNN")
    model_dropdown = ttk.Combobox(webcam_model_frame, textvariable=model_var, state="readonly",
                                  values=["CNN", "CDCN", "VIT"])
    model_dropdown.pack(fill=tk.X, pady=2)

    # Switch Model Button (for changing model during webcam operation)
    switch_model_button = ttk.Button(webcam_model_frame, text="Switch Model", command=switch_webcam_model)
    switch_model_button.pack(fill=tk.X, pady=2)

    # Combined Start Button below webcam
    start_combined_button = ttk.Button(left_frame, text="Start Webcam & Anti-Spoofing", command=start_webcam_and_recognition)
    start_combined_button.pack(fill=tk.X, pady=5)

    stop_combined_button = ttk.Button(left_frame, text="Stop Webcam & Anti-Spoofing", command=stop_webcam_and_recognition)
    stop_combined_button.pack(fill=tk.X, pady=10)

    # Imported Image Display Frame (Right, fixed size)
    image_display_frame = ttk.LabelFrame(right_frame, text="Imported Image (max 200x200)", padding="10")
    image_display_frame.pack(fill=tk.X, pady=10)

    image_canvas = tk.Canvas(image_display_frame, width=200, height=200, bg="gray", highlightthickness=0)
    image_canvas.pack()

    # Import Image Button above image display
    import_image_button = ttk.Button(right_frame, text="Import Image", command=import_image)
    import_image_button.pack(fill=tk.X, pady=5)

    # Model Selection Frame below image display (for static image prediction)
    static_model_frame = ttk.LabelFrame(right_frame, text="Static Image Model Selection", padding="10")
    static_model_frame.pack(fill=tk.X, pady=5)

    # Instruction label for static images
    ttk.Label(static_model_frame, text="Select model for static image prediction:",
              font=("Arial", 9)).pack(anchor=tk.W)

    static_model_var = tk.StringVar(value="CNN")
    static_model_dropdown = ttk.Combobox(static_model_frame, textvariable=static_model_var, state="readonly",
                                         values=["CNN", "CDCN", "VIT"])
    static_model_dropdown.pack(fill=tk.X, pady=2)

    # Predict Button
    predict_button = ttk.Button(static_model_frame, text="Predict Static Image", command=predict_image_with_selected_model)
    predict_button.pack(fill=tk.X, pady=5)

    # Output Box Frame (Right)
    output_frame = ttk.LabelFrame(right_frame, text="Anti-Spoofing Results", padding="10")
    output_frame.pack(fill=tk.BOTH, expand=True, pady=10)

    result_output = tk.Text(output_frame, height=15, state='normal', wrap='word', font=("Consolas", 10))
    result_output.pack(fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(result_output, command=result_output.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    result_output.config(yscrollcommand=scrollbar.set)

    root.mainloop()

# Only run launch_ui() if this file is executed directly
if __name__ == "__main__":
    launch_ui()
