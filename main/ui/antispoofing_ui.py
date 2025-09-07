"""
Simplified Anti-Spoofing UI
Separated from backend logic
"""
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import time
import os
import sys

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.antispoofing_detector import AntiSpoofingDetector
from controller.webcam_controller import WebcamController

class AntiSpoofingUI:
    """Main UI class for anti-spoofing application"""
    
    def __init__(self):
        # Initialize backend components
        self.detector = AntiSpoofingDetector()
        self.webcam_controller = WebcamController(self.detector)
        
        # UI state
        self.selected_image = None
        self.webcam_photo = None
        self.ui_update_active = False
        
        # Initialize UI
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the main UI"""
        self.root = tk.Tk()
        self.root.title("Anti-Spoofing Detection System")
        self.root.geometry("1200x700")
        self.root.resizable(False, False)
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Create main frames
        self.create_main_frames()
        self.create_webcam_section()
        self.create_image_section()
        self.create_output_section()
        
        # Bind cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_main_frames(self):
        """Create main layout frames"""
        self.left_frame = ttk.Frame(self.root, padding="10")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.right_frame = ttk.Frame(self.root, padding="10")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    def create_webcam_section(self):
        """Create webcam control section"""
        # Webcam display
        webcam_frame = ttk.LabelFrame(self.left_frame, text="Webcam Feed", padding="10")
        webcam_frame.pack(fill=tk.BOTH, expand=True)
        
        self.webcam_label = ttk.Label(webcam_frame, background="black", text="Webcam Feed")
        self.webcam_label.pack(fill=tk.BOTH, expand=True)
        
        # Model selection for webcam
        model_frame = ttk.LabelFrame(self.left_frame, text="Webcam Model Selection", padding="10")
        model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(model_frame, text="Select model for real-time detection:", 
                 font=("Arial", 9)).pack(anchor=tk.W)
        
        self.webcam_model_var = tk.StringVar(value="CNN")
        webcam_model_dropdown = ttk.Combobox(model_frame, textvariable=self.webcam_model_var, 
                                           state="readonly", values=["CNN", "CDCN", "VIT"])
        webcam_model_dropdown.pack(fill=tk.X, pady=2)
        
        # Control buttons
        button_frame = ttk.Frame(model_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.start_button = ttk.Button(button_frame, text="Start Webcam", 
                                      command=self.start_webcam)
        self.start_button.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Webcam", 
                                     command=self.stop_webcam)
        self.stop_button.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        self.switch_model_button = ttk.Button(model_frame, text="Switch Model", 
                                            command=self.switch_webcam_model)
        self.switch_model_button.pack(fill=tk.X, pady=2)
        
        # Status label
        self.status_label = ttk.Label(model_frame, text="Status: Ready", 
                                     font=("Arial", 9), foreground="blue")
        self.status_label.pack(anchor=tk.W, pady=2)
    
    def create_image_section(self):
        """Create static image processing section"""
        # Image display
        image_frame = ttk.LabelFrame(self.right_frame, text="Static Image Analysis", padding="10")
        image_frame.pack(fill=tk.X, pady=10)
        
        self.image_canvas = tk.Canvas(image_frame, width=200, height=200, bg="gray")
        self.image_canvas.pack()
        
        # Import button
        import_button = ttk.Button(image_frame, text="Import Image", 
                                  command=self.import_image)
        import_button.pack(fill=tk.X, pady=5)
        
        # Model selection for static images
        static_model_frame = ttk.LabelFrame(image_frame, text="Model Selection", padding="5")
        static_model_frame.pack(fill=tk.X, pady=5)
        
        self.static_model_var = tk.StringVar(value="CNN")
        static_model_dropdown = ttk.Combobox(static_model_frame, textvariable=self.static_model_var,
                                           state="readonly", values=["CNN", "CDCN", "VIT"])
        static_model_dropdown.pack(fill=tk.X, pady=2)
        
        # Predict button
        predict_button = ttk.Button(static_model_frame, text="Analyze Image", 
                                   command=self.predict_static_image)
        predict_button.pack(fill=tk.X, pady=2)
    
    def create_output_section(self):
        """Create output/results section"""
        output_frame = ttk.LabelFrame(self.right_frame, text="Detection Results", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.result_text = tk.Text(output_frame, height=15, wrap='word', 
                                  font=("Consolas", 10), state='normal')
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(output_frame, command=self.result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)
        
        # Clear button
        clear_button = ttk.Button(output_frame, text="Clear Results", 
                                 command=self.clear_results)
        clear_button.pack(fill=tk.X, pady=2)
    
    def start_webcam(self):
        """Start webcam and detection"""
        try:
            # Start webcam
            success, message = self.webcam_controller.start_webcam()
            if not success:
                self.log_result(f"Error: {message}")
                return
            
            # Start processing
            model_type = self.webcam_model_var.get()
            success, message = self.webcam_controller.start_processing(model_type)
            if not success:
                self.log_result(f"Error: {message}")
                self.webcam_controller.stop_webcam()
                return
            
            self.log_result(f"Webcam started with {model_type} model")
            self.update_status(f"Active - {model_type} Model")
            
            # Start UI update loop
            self.ui_update_active = True
            self.update_webcam_display()
            
        except Exception as e:
            self.log_result(f"Error starting webcam: {str(e)}")
    
    def stop_webcam(self):
        """Stop webcam and detection"""
        try:
            self.ui_update_active = False
            
            # Stop processing
            success, message = self.webcam_controller.stop_processing()
            if success:
                self.log_result(message)
            
            # Stop webcam
            success, message = self.webcam_controller.stop_webcam()
            if success:
                self.log_result(message)
            
            self.update_status("Stopped")
            self.webcam_label.config(image='', text="Webcam Feed")
            
        except Exception as e:
            self.log_result(f"Error stopping webcam: {str(e)}")
    
    def switch_webcam_model(self):
        """Switch the model used for webcam detection"""
        try:
            model_type = self.webcam_model_var.get()
            success, message = self.webcam_controller.switch_model(model_type)
            
            if success:
                self.log_result(message)
                self.update_status(f"Active - {model_type} Model")
            else:
                self.log_result(f"Error: {message}")
                
        except Exception as e:
            self.log_result(f"Error switching model: {str(e)}")
    
    def update_webcam_display(self):
        """Update webcam display with detection results"""
        if not self.ui_update_active:
            return
        
        try:
            # Get current frame
            frame, error = self.webcam_controller.get_frame()
            if frame is None:
                if error:
                    self.log_result(f"Frame error: {error}")
                self.root.after(50, self.update_webcam_display)
                return
            
            display_frame = frame.copy()
            
            # Add frame for processing (rate-limited)
            self.webcam_controller.put_frame_for_processing(frame)
            
            # Get processed results
            processed_data = self.webcam_controller.get_processed_frame()
            if processed_data:
                results = processed_data['results']
                model_type = processed_data['model_type']
                
                # Draw results on frame
                for result in results:
                    self.draw_detection_result(display_frame, result)
                
                # Log results (less frequently)
                if results:
                    for result in results:
                        self.log_result(f"Detection ({model_type}): {result['recognized_identity']} "
                                      f"- Confidence: {result['confidence']:.2f}")
            
            # Convert and display frame
            self.display_frame_on_label(display_frame, self.webcam_label)
            
            # Schedule next update (reduced frequency to prevent constant refreshing)
            self.root.after(100, self.update_webcam_display)  # 10 FPS display rate
            
        except Exception as e:
            print(f"Display update error: {e}")
            self.root.after(100, self.update_webcam_display)
    
    def draw_detection_result(self, frame, result):
        """Draw detection results on frame with enhanced visibility"""
        try:
            bbox = result['bbox']
            is_spoof = result['is_spoof']
            confidence = result['confidence']

            # Validate bbox format and values
            if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
                print(f"[ERROR] Invalid bbox format: {bbox}")
                return

            x, y, w, h = bbox

            # Validate coordinate values (including numpy types)
            try:
                import numpy as np
                is_numeric = lambda val: isinstance(val, (int, float)) or (hasattr(val, 'dtype') and np.issubdtype(val.dtype, np.number))
            except ImportError:
                is_numeric = lambda val: isinstance(val, (int, float))
            
            if not all(is_numeric(val) for val in [x, y, w, h]):
                print(f"[ERROR] Invalid bbox values: {bbox}")
                return

            # Ensure positive dimensions
            if w <= 0 or h <= 0:
                print(f"[ERROR] Invalid bbox dimensions: w={w}, h={h}")
                return

            # Convert to regular Python types for OpenCV compatibility
            x, y, w, h = float(x), float(y), float(w), float(h)

            # Skip if bbox is too small or invalid
            if w < 10 or h < 10:
                print(f"[WARNING] Bbox too small: {w}x{h}")
                return

            # Choose color based on result (BGR format for OpenCV)
            if is_spoof:
                color = (0, 0, 255)  # Red for spoof
                bg_color = (0, 0, 0)  # Black background for text
            else:
                color = (0, 255, 0)  # Green for real
                bg_color = (0, 0, 0)  # Black background for text

            # Draw thicker rectangle for better visibility (thickness based on confidence)
            thickness = max(2, int(confidence * 4))  # 2-4 pixels based on confidence
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)

            # Draw corner markers for better visual identification
            corner_length = min(20, int(w) // 4, int(h) // 4)
            if corner_length > 0:
                # Top-left corner
                cv2.line(frame, (int(x), int(y)), (int(x + corner_length), int(y)), color, 3)
                cv2.line(frame, (int(x), int(y)), (int(x), int(y + corner_length)), color, 3)
                # Top-right corner
                cv2.line(frame, (int(x + w), int(y)), (int(x + w - corner_length), int(y)), color, 3)
                cv2.line(frame, (int(x + w), int(y)), (int(x + w), int(y + corner_length)), color, 3)
                # Bottom-left corner
                cv2.line(frame, (int(x), int(y + h)), (int(x + corner_length), int(y + h)), color, 3)
                cv2.line(frame, (int(x), int(y + h)), (int(x), int(y + h - corner_length)), color, 3)
                # Bottom-right corner
                cv2.line(frame, (int(x + w), int(y + h)), (int(x + w - corner_length), int(y + h)), color, 3)
                cv2.line(frame, (int(x + w), int(y + h)), (int(x + w), int(y + h - corner_length)), color, 3)

            # Create label with more information
            confidence_pct = int(confidence * 100)
            label = f"{'SPOOF' if is_spoof else 'REAL'} {confidence_pct}%"
            model_type = result.get('model_type', 'CNN')

            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

            # Ensure text doesn't go outside frame boundaries
            text_y = max(text_height + 15, int(y) + 5) if int(y) - text_height - 10 < 0 else int(y) - 5
            text_bg_y1 = max(0, text_y - text_height - 5)
            text_bg_y2 = max(text_height + 5, text_y + 5)

            # Draw background rectangle for text
            cv2.rectangle(frame, (int(x), text_bg_y1), (min(int(x) + text_width, frame.shape[1]), text_bg_y2), bg_color, -1)

            # Draw main label
            cv2.putText(frame, label, (int(x), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Draw model type in smaller text (ensure it doesn't go outside frame)
            model_y = min(int(y + h) + 35, frame.shape[0] - 10)
            cv2.putText(frame, f"[{model_type}]", (int(x), model_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        except Exception as e:
            print(f"Error drawing result: {e}")
            print(f"Result data: {result}")
            import traceback
            traceback.print_exc()
    
    def display_frame_on_label(self, frame, label):
        """Display frame on tkinter label"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)
            pil_image.thumbnail((480, 360), Image.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update label
            label.config(image=photo, text="")
            label.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"Error displaying frame: {e}")
    
    def import_image(self):
        """Import static image for analysis"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Image File",
                filetypes=[("Image Files", "*.png *.jpg *.jpeg *.gif *.bmp"), ("All Files", "*")]
            )
            
            if file_path:
                self.log_result(f"Imported: {file_path}")
                
                # Load and display image
                img = Image.open(file_path)
                img.thumbnail((200, 200), Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)
                
                self.image_canvas.delete("all")
                self.image_canvas.create_image(100, 100, image=img_tk)
                self.image_canvas.image = img_tk
                
                # Store original image
                self.selected_image = Image.open(file_path)
                
        except Exception as e:
            self.log_result(f"Error importing image: {str(e)}")
    
    def predict_static_image(self):
        """Predict static image using original predictor approach"""
        if self.selected_image is None:
            self.log_result("No image selected. Please import an image first.")
            return
        
        try:
            model_type = self.static_model_var.get()
            
            # Use original predictor approach - direct image prediction without face detection
            if model_type == "CNN":
                from backend.predictor import predict_image
                pred_class, confidence = predict_image(self.selected_image)
                # predictor.py: 0=live, 1=spoof
                label = "Real" if pred_class == 0 else "Spoof Detected"
                self.log_result(f"Static Image Prediction (CNN): {label} | Confidence: {confidence:.2f}")
            
            elif model_type == "CDCN":
                from backend.cdcn_predictor import predict_image
                pred_class, confidence = predict_image(self.selected_image)
                # cdcn_predictor.py: 0=live, 1=spoof
                label = "Real" if pred_class == 0 else "Spoof Detected"
                self.log_result(f"Static Image Prediction (CDCN): {label} | Confidence: {confidence:.2f}")
            
            elif model_type == "VIT":
                from backend.vit_predictor import predict_image_vit
                pred_class, confidence = predict_image_vit(self.selected_image)
                # vit_predictor.py: 0=spoof, 1=live
                label = "Real" if pred_class == 1 else "Spoof Detected"
                self.log_result(f"Static Image Prediction (VIT): {label} | Confidence: {confidence:.2f}")
            
            else:
                self.log_result(f"Unknown model type: {model_type}")
                
        except Exception as e:
            self.log_result(f"Error analyzing image: {str(e)}")
    
    def log_result(self, message):
        """Log result to output text"""
        try:
            self.result_text.insert(tk.END, f"{message}\n")
            self.result_text.see(tk.END)
            
        except Exception as e:
            print(f"Error logging result: {e}")
    
    def clear_results(self):
        """Clear results text"""
        self.result_text.delete(1.0, tk.END)
    
    def update_status(self, status):
        """Update status label"""
        self.status_label.config(text=f"Status: {status}")
    
    def on_closing(self):
        """Handle window closing"""
        try:
            self.ui_update_active = False
            self.webcam_controller.cleanup()
            self.root.destroy()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
            self.root.destroy()
    
    def run(self):
        """Start the UI"""
        self.root.mainloop()

def launch_ui():
    """Launch the anti-spoofing UI"""
    app = AntiSpoofingUI()
    app.run()

if __name__ == "__main__":
    launch_ui()
