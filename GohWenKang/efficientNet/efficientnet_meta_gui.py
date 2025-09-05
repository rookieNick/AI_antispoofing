import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import cv2
import os
from datetime import datetime

# Import the EfficientNet+Meta model classes (same as in training script)
class MetaAttentionModule(nn.Module):
    """Meta-learning attention module for subject adaptation"""
    def __init__(self, in_channels, reduction=16):
        super(MetaAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        
        # Meta-learning parameters for subject-specific adaptation
        self.meta_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Tanh()
        )
        
    def forward(self, x, meta_features=None):
        b, c, _, _ = x.size()
        
        # Channel attention
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        
        # Combine average and max pooling
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.fc(combined).view(b, c, 1, 1)
        
        # Apply meta-learning adaptation if available
        if meta_features is not None:
            meta_attention = self.meta_fc(meta_features).view(b, c, 1, 1)
            attention = attention * (1 + meta_attention)
        
        return x * attention

class EfficientNetMeta(nn.Module):
    """EfficientNet with Meta-Learning for Face Anti-Spoofing"""
    def __init__(self, num_classes=2, efficientnet_version='b0', dropout_rate=0.4):
        super(EfficientNetMeta, self).__init__()
        
        # Load pre-trained EfficientNet
        if efficientnet_version == 'b0':
            self.backbone = models.efficientnet_b0(pretrained=True)
            feature_dim = 1280
        elif efficientnet_version == 'b1':
            self.backbone = models.efficientnet_b1(pretrained=True)
            feature_dim = 1280
        elif efficientnet_version == 'b2':
            self.backbone = models.efficientnet_b2(pretrained=True)
            feature_dim = 1408
        else:
            raise ValueError(f"Unsupported EfficientNet version: {efficientnet_version}")
        
        # Remove the final classifier
        self.features = self.backbone.features
        
        # Add meta-attention modules at different scales
        self.meta_attention_1 = MetaAttentionModule(40)   # After first few blocks
        self.meta_attention_2 = MetaAttentionModule(112)  # Middle blocks
        self.meta_attention_3 = MetaAttentionModule(feature_dim)  # Final features
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Meta-learning feature extractor
        self.meta_feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256)
        )
        
        # Enhanced classifier with meta-learning
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + 256, 512),  # Concatenate backbone + meta features
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
        
        # Regression head for continuous confidence scores
        self.regression_head = nn.Sequential(
            nn.Linear(feature_dim + 256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def extract_features_at_scale(self, x, scale_idx):
        """Extract features at different scales for meta-attention"""
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == scale_idx:
                return x
        return x
    
    def forward(self, x, meta_features=None, return_confidence=False):
        batch_size = x.size(0)
        
        # Extract features at multiple scales
        features_scale_1 = None
        features_scale_2 = None
        
        # Forward through EfficientNet with meta-attention
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # Apply meta-attention at different scales
            if i == 2:  # Early features
                features_scale_1 = x
                if meta_features is not None:
                    x = self.meta_attention_1(x, meta_features)
            elif i == 4:  # Middle features
                features_scale_2 = x
                if meta_features is not None:
                    x = self.meta_attention_2(x, meta_features)
        
        # Global pooling
        x = self.global_pool(x).view(batch_size, -1)
        
        # Apply final meta-attention
        if meta_features is not None:
            x = x + self.meta_attention_3(x.unsqueeze(-1).unsqueeze(-1), meta_features).view(batch_size, -1)
        
        # Extract meta-learning features
        if meta_features is None:
            # Generate meta features from current input if not provided
            meta_feats = self.meta_feature_extractor(x)
        else:
            meta_feats = meta_features
        
        # Concatenate backbone and meta features
        combined_features = torch.cat([x, meta_feats], dim=1)
        
        # Classification
        logits = self.classifier(combined_features)
        
        if return_confidence:
            # Also return continuous confidence score
            confidence = self.regression_head(combined_features)
            return logits, confidence
        
        return logits

class EfficientNetMetaGUI:
    def __init__(self, model_path=None):
        self.root = tk.Tk()
        self.root.title("EfficientNet+Meta Anti-Spoofing Demo")
        self.root.geometry("1300x850")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = None
        self.model_loaded = False
        self.efficientnet_version = 'b0'  # Default version
        
        # Default model path
        if model_path is None:
            model_path = "GohWenKang/efficientNet/efficientnet_models/best_efficientnet_model.pth"
        
        # Load model if path exists
        if os.path.exists(model_path):
            self.load_model(model_path)
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.class_names = ['Spoof', 'Live']
        self.setup_ui()
        
        # Webcam variables
        self.webcam_active = False
        self.webcam_window = None
        
    def load_model(self, model_path):
        """Load the EfficientNet+Meta model"""
        try:
            print(f"Loading EfficientNet+Meta model from {model_path}")
            
            # Initialize model architecture
            self.model = EfficientNetMeta(
                num_classes=2,
                efficientnet_version=self.efficientnet_version,
                dropout_rate=0.4
            )
            
            # Load state dict
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            self.model_loaded = True
            print("✅ EfficientNet+Meta model loaded successfully!")
            
            # Update status
            if hasattr(self, 'status_label'):
                self.status_label.config(text=f"Model loaded: {os.path.basename(model_path)}")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model_loaded = False
            if hasattr(self, 'status_label'):
                self.status_label.config(text=f"Error: {str(e)}")
            messagebox.showerror("Model Error", f"Failed to load model: {str(e)}")
        
    def setup_ui(self):
        # Create main style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title with EfficientNet+Meta branding
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, columnspan=4, pady=(0, 15), sticky="ew")
        
        title_label = ttk.Label(title_frame, text="EfficientNet+Meta Anti-Spoofing Detection", 
                               font=('Arial', 18, 'bold'))
        title_label.grid(row=0, column=0)
        
        subtitle_label = ttk.Label(title_frame, text="Real-time Face Liveness Detection with Meta-Learning & Compound Scaling", 
                                  font=('Arial', 11), foreground='gray')
        subtitle_label.grid(row=1, column=0, pady=(5, 0))
        
        # Model status indicator
        self.model_status_frame = ttk.LabelFrame(main_frame, text="Model Status", padding="10")
        self.model_status_frame.grid(row=1, column=0, columnspan=4, pady=(0, 10), sticky="ew")
        
        status_color = "green" if self.model_loaded else "red"
        status_text = "✅ EfficientNet+Meta Model Loaded" if self.model_loaded else "❌ Model Not Loaded"
        
        self.model_status_label = ttk.Label(self.model_status_frame, text=status_text, 
                                           font=('Arial', 10, 'bold'))
        self.model_status_label.grid(row=0, column=0)
        
        # Device info
        device_text = f"Device: {self.device.type.upper()}"
        if torch.cuda.is_available():
            device_text += f" ({torch.cuda.get_device_name(0)})"
        
        ttk.Label(self.model_status_frame, text=device_text, 
                 font=('Arial', 9)).grid(row=0, column=1, padx=(20, 0))
        
        # Model version info
        ttk.Label(self.model_status_frame, text=f"EfficientNet-{self.efficientnet_version.upper()}", 
                 font=('Arial', 9)).grid(row=0, column=2, padx=(20, 0))
        
        # Image display frame
        self.image_frame = ttk.LabelFrame(main_frame, text="Input Image", padding="15")
        self.image_frame.grid(row=2, column=0, columnspan=2, padx=(0, 10), pady=10, sticky="nsew")
        
        self.image_label = ttk.Label(self.image_frame, text="No image selected\nClick 'Select Image' to start", 
                                    background="lightgray", anchor="center", 
                                    font=('Arial', 12))
        self.image_label.grid(row=0, column=0, padx=20, pady=20)
        
        # Control frame
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="15")
        control_frame.grid(row=2, column=2, padx=10, pady=10, sticky="nsew")
        
        # Load Model button
        ttk.Button(control_frame, text="📁 Load Model", 
                  command=self.select_model).grid(row=0, column=0, pady=5, sticky="ew")
        
        # EfficientNet version selector
        version_frame = ttk.Frame(control_frame)
        version_frame.grid(row=1, column=0, pady=5, sticky="ew")
        
        ttk.Label(version_frame, text="Version:").grid(row=0, column=0, sticky="w")
        self.version_var = tk.StringVar(value=self.efficientnet_version)
        version_combo = ttk.Combobox(version_frame, textvariable=self.version_var, 
                                   values=['b0', 'b1', 'b2'], state="readonly", width=5)
        version_combo.grid(row=0, column=1, padx=(5, 0))
        version_combo.bind('<<ComboboxSelected>>', self.on_version_change)
        
        # Select Image button
        ttk.Button(control_frame, text="🖼️ Select Image", 
                  command=self.select_image).grid(row=2, column=0, pady=5, sticky="ew")
        
        # Predict button
        self.predict_btn = ttk.Button(control_frame, text="🔍 Predict", 
                                     command=self.predict_image)
        self.predict_btn.grid(row=3, column=0, pady=5, sticky="ew")
        
        # Webcam button
        ttk.Button(control_frame, text="📷 Webcam Demo", 
                  command=self.start_webcam).grid(row=4, column=0, pady=5, sticky="ew")
        
        # Batch Test button
        ttk.Button(control_frame, text="📊 Batch Test", 
                  command=self.batch_test).grid(row=5, column=0, pady=5, sticky="ew")
        
        # Results frame
        self.results_frame = ttk.LabelFrame(main_frame, text="Detection Results", padding="15")
        self.results_frame.grid(row=2, column=3, padx=(10, 0), pady=10, sticky="nsew")
        
        # Result display
        self.result_label = ttk.Label(self.results_frame, text="No prediction yet", 
                                     font=('Arial', 14, 'bold'))
        self.result_label.grid(row=0, column=0, pady=10)
        
        self.confidence_label = ttk.Label(self.results_frame, text="", 
                                         font=('Arial', 12))
        self.confidence_label.grid(row=1, column=0, pady=5)
        
        # Regression confidence score
        self.regression_label = ttk.Label(self.results_frame, text="", 
                                         font=('Arial', 10), foreground='blue')
        self.regression_label.grid(row=2, column=0, pady=5)
        
        # Probability bars frame
        prob_frame = ttk.Frame(self.results_frame)
        prob_frame.grid(row=3, column=0, pady=10, sticky="ew")
        
        # Live probability bar
        ttk.Label(prob_frame, text="Live:").grid(row=0, column=0, sticky="w")
        self.live_var = tk.DoubleVar()
        self.live_bar = ttk.Progressbar(prob_frame, variable=self.live_var, 
                                       maximum=100, length=200)
        self.live_bar.grid(row=0, column=1, padx=(10, 5))
        self.live_percent = ttk.Label(prob_frame, text="0%")
        self.live_percent.grid(row=0, column=2)
        
        # Spoof probability bar
        ttk.Label(prob_frame, text="Spoof:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.spoof_var = tk.DoubleVar()
        self.spoof_bar = ttk.Progressbar(prob_frame, variable=self.spoof_var, 
                                        maximum=100, length=200)
        self.spoof_bar.grid(row=1, column=1, padx=(10, 5), pady=(5, 0))
        self.spoof_percent = ttk.Label(prob_frame, text="0%")
        self.spoof_percent.grid(row=1, column=2, pady=(5, 0))
        
        # Processing time and meta-learning info
        self.time_label = ttk.Label(self.results_frame, text="", 
                                   font=('Arial', 10), foreground='gray')
        self.time_label.grid(row=4, column=0, pady=(10, 0))
        
        self.meta_label = ttk.Label(self.results_frame, text="", 
                                   font=('Arial', 9), foreground='purple')
        self.meta_label.grid(row=5, column=0, pady=(5, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=3, column=0, columnspan=4, pady=15, sticky="ew")
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready" if self.model_loaded else "Load model to start", 
                                     font=('Arial', 10))
        self.status_label.grid(row=4, column=0, columnspan=4, pady=(5, 0))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=2)
        main_frame.columnconfigure(2, weight=1)
        main_frame.columnconfigure(3, weight=2)
        main_frame.rowconfigure(2, weight=1)
        
        self.current_image_path = None
        
        # Disable predict button if model not loaded
        if not self.model_loaded:
            self.predict_btn.config(state='disabled')
    
    def on_version_change(self, event):
        """Handle EfficientNet version change"""
        new_version = self.version_var.get()
        if self.model_loaded and new_version != self.efficientnet_version:
            messagebox.showinfo("Version Change", 
                               f"Version changed to EfficientNet-{new_version.upper()}.\n"
                               "Please reload the model for this version.")
    
    def select_model(self):
        """Select and load model file"""
        model_path = filedialog.askopenfilename(
            title="Select EfficientNet+Meta Model File",
            filetypes=[
                ("PyTorch Model", "*.pth"),
                ("All files", "*.*")
            ],
            initialdir="efficientnet_models"
        )
        
        if model_path:
            self.load_model(model_path)
            self.predict_btn.config(state='normal' if self.model_loaded else 'disabled')
            
            # Update model status
            status_text = "✅ EfficientNet+Meta Model Loaded" if self.model_loaded else "❌ Failed to Load Model"
            self.model_status_label.config(text=status_text)
    
    def select_image(self):
        """Select image for prediction"""
        file_path = filedialog.askopenfilename(
            title="Select Face Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.status_label.config(text=f"Image loaded: {os.path.basename(file_path)}")
            
            # Reset results
            self.result_label.config(text="Click 'Predict' to analyze")
            self.confidence_label.config(text="")
            self.regression_label.config(text="")
            self.time_label.config(text="")
            self.meta_label.config(text="")
            self.live_var.set(0)
            self.spoof_var.set(0)
            self.live_percent.config(text="0%")
            self.spoof_percent.config(text="0%")
    
    def display_image(self, image_path):
        """Display selected image"""
        try:
            # Load and resize image for display
            image = Image.open(image_path)
            
            # Calculate size to maintain aspect ratio
            display_size = (350, 350)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep reference
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")
    
    def predict_image(self):
        """Predict if image is live or spoof"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
            
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
        
        # Start progress bar
        self.progress.start()
        self.status_label.config(text="Analyzing image with EfficientNet+Meta...")
        
        # Run prediction in separate thread
        thread = threading.Thread(target=self._predict_thread)
        thread.daemon = True
        thread.start()
    
    def _predict_thread(self):
        """Thread function for prediction"""
        try:
            start_time = datetime.now()
            
            # Load and preprocess image
            image = Image.open(self.current_image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Predict with EfficientNet+Meta model
            with torch.no_grad():
                # Get both classification and regression outputs
                logits, regression_confidence = self.model(input_tensor, return_confidence=True)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            prediction = self.class_names[predicted.item()]
            confidence_score = confidence.item()
            regression_score = regression_confidence.item()
            probs = probabilities.cpu().numpy()[0]
            
            # Update UI in main thread
            self.root.after(0, self._update_results, prediction, confidence_score, 
                          probs, processing_time, regression_score)
            
        except Exception as e:
            self.root.after(0, self._show_error, str(e))
    
    def _update_results(self, prediction, confidence, probabilities, processing_time, regression_score):
        """Update UI with prediction results"""
        # Stop progress bar
        self.progress.stop()
        
        # Update result labels
        color = "#2e7d32" if prediction == "Live" else "#d32f2f"  # Green for Live, Red for Spoof
        self.result_label.config(text=f"🎯 {prediction}", foreground=color)
        self.confidence_label.config(text=f"Confidence: {confidence:.1%}")
        
        # Show regression confidence (continuous score)
        self.regression_label.config(text=f"Regression Score: {regression_score:.3f}")
        
        # Update probability bars
        live_prob = probabilities[1] * 100
        spoof_prob = probabilities[0] * 100
        
        self.live_var.set(live_prob)
        self.spoof_var.set(spoof_prob)
        self.live_percent.config(text=f"{live_prob:.1f}%")
        self.spoof_percent.config(text=f"{spoof_prob:.1f}%")
        
        # Update processing time
        self.time_label.config(text=f"Processing time: {processing_time:.1f}ms")
        
        # Meta-learning info
        self.meta_label.config(text=f"EfficientNet-{self.efficientnet_version.upper()} + Meta-Attention")
        
        self.status_label.config(text="Analysis completed")
    
    def _show_error(self, error_msg):
        """Show error message"""
        self.progress.stop()
        self.status_label.config(text="Error occurred")
        messagebox.showerror("Prediction Error", f"Analysis failed: {error_msg}")
    
    def batch_test(self):
        """Test multiple images from a directory"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
            
        test_dir = filedialog.askdirectory(title="Select Directory with Test Images")
        if test_dir:
            self.status_label.config(text="Running batch test...")
            messagebox.showinfo("Batch Test", f"Testing images in: {test_dir}\nCheck console for results.")
            
            # Run batch test in thread
            thread = threading.Thread(target=self._batch_test_thread, args=(test_dir,))
            thread.daemon = True
            thread.start()
    
    def _batch_test_thread(self, test_dir):
        """Thread function for batch testing"""
        try:
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
            image_files = [f for f in os.listdir(test_dir) 
                          if os.path.splitext(f.lower())[1] in image_extensions]
            
            if not image_files:
                self.root.after(0, lambda: messagebox.showwarning("Warning", "No image files found!"))
                return
            
            print(f"\n🧪 EFFICIENTNET+META BATCH TESTING {len(image_files)} IMAGES")
            print("=" * 60)
            
            results = []
            
            for i, filename in enumerate(image_files):
                filepath = os.path.join(test_dir, filename)
                
                try:
                    # Load and predict
                    image = Image.open(filepath).convert('RGB')
                    input_tensor = self.transform(image).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        logits, regression_confidence = self.model(input_tensor, return_confidence=True)
                        probabilities = torch.nn.functional.softmax(logits, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                    
                    prediction = self.class_names[predicted.item()]
                    conf_score = confidence.item()
                    reg_score = regression_confidence.item()
                    
                    results.append({
                        'filename': filename,
                        'prediction': prediction,
                        'confidence': conf_score,
                        'regression_score': reg_score,
                        'live_prob': probabilities[0][1].item(),
                        'spoof_prob': probabilities[0][0].item()
                    })
                    
                    print(f"{i+1:3d}. {filename:<30} → {prediction:<5} ({conf_score:.1%}) [Reg: {reg_score:.3f}]")
                    
                except Exception as e:
                    print(f"{i+1:3d}. {filename:<30} → ERROR: {str(e)}")
            
            # Summary
            live_count = sum(1 for r in results if r['prediction'] == 'Live')
            spoof_count = len(results) - live_count
            avg_confidence = np.mean([r['confidence'] for r in results])
            avg_regression = np.mean([r['regression_score'] for r in results])
            
            print(f"\n📊 EFFICIENTNET+META BATCH TEST SUMMARY:")
            print(f"   Total Images: {len(results)}")
            print(f"   Live Detected: {live_count}")
            print(f"   Spoof Detected: {spoof_count}")
            print(f"   Average Confidence: {avg_confidence:.1%}")
            print(f"   Average Regression Score: {avg_regression:.3f}")
            print(f"   Model Version: EfficientNet-{self.efficientnet_version.upper()}")
            print("=" * 60)
            
            self.root.after(0, lambda: self.status_label.config(text="Batch test completed"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Batch test failed: {str(e)}"))
    
    def start_webcam(self):
        """Start webcam demo"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
            
        # Create webcam window
        self.webcam_window = tk.Toplevel(self.root)
        self.webcam_window.title("EfficientNet+Meta Webcam Demo")
        self.webcam_window.geometry("850x650")
        self.webcam_window.configure(bg='#2b2b2b')
        
        # Webcam display frame
        webcam_frame = ttk.Frame(self.webcam_window, padding="10")
        webcam_frame.pack(expand=True, fill='both')
        
        # Title
        ttk.Label(webcam_frame, text="Live Face Anti-Spoofing Detection", 
                 font=('Arial', 14, 'bold')).pack(pady=(0, 10))
        
        # Model info
        ttk.Label(webcam_frame, text=f"EfficientNet-{self.efficientnet_version.upper()} + Meta-Learning", 
                 font=('Arial', 10), foreground='blue').pack()
        
        # Video display
        self.webcam_label = ttk.Label(webcam_frame, text="Initializing camera...", 
                                     background="black", foreground="white")
        self.webcam_label.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Control buttons
        button_frame = ttk.Frame(webcam_frame)
        button_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Button(button_frame, text="Stop Camera", 
                  command=self._stop_webcam).pack(side='left')
        
        self.webcam_status_label = ttk.Label(button_frame, text="Initializing...", 
                                           font=('Arial', 10))
        self.webcam_status_label.pack(side='right')
        
        # Start webcam thread
        self.webcam_active = True
        thread = threading.Thread(target=self._webcam_thread)
        thread.daemon = True
        thread.start()
        
        # Handle window close
        self.webcam_window.protocol("WM_DELETE_WINDOW", self._stop_webcam)
    
    def _webcam_thread(self):
        """Webcam processing thread"""
        cap = cv2.VideoCapture(0)
        frame_count = 0
        
        if not cap.isOpened():
            self.root.after(0, lambda: self.webcam_status_label.config(text="Camera not found!"))
            return
        
        while self.webcam_active and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 10th frame for real-time performance
            if frame_count % 10 == 0 and self.model_loaded:
                try:
                    # Convert and predict
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    
                    input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        logits, regression_confidence = self.model(input_tensor, return_confidence=True)
                        probabilities = torch.nn.functional.softmax(logits, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                    
                    prediction = self.class_names[predicted.item()]
                    confidence_score = confidence.item()
                    regression_score = regression_confidence.item()
                    
                    # Add prediction overlay to frame
                    color = (0, 255, 0) if prediction == "Live" else (0, 0, 255)
                    text = f"{prediction}: {confidence_score:.1%}"
                    reg_text = f"Reg: {regression_score:.3f}"
                    
                    # Add background rectangle for text
                    cv2.rectangle(frame, (10, 10), (450, 80), (0, 0, 0), -1)
                    cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, color, 2)
                    cv2.putText(frame, reg_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (255, 255, 255), 1)
                    
                    # Add model info
                    cv2.putText(frame, f"EfficientNet-{self.efficientnet_version.upper()}+Meta", (300, 35), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    # Update status
                    if hasattr(self, 'webcam_status_label'):
                        status_text = f"Detecting: {prediction} ({confidence_score:.1%}, R:{regression_score:.3f})"
                        self.root.after(0, lambda: self.webcam_status_label.config(text=status_text))
                
                except Exception as e:
                    print(f"Webcam prediction error: {e}")
            
            # Convert frame for tkinter display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Resize to fit display
            pil_image = pil_image.resize((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update webcam display
            if hasattr(self, 'webcam_label') and self.webcam_active:
                self.webcam_label.config(image=photo, text="")
                self.webcam_label.image = photo
        
        cap.release()
    
    def _stop_webcam(self):
        """Stop webcam and close window"""
        self.webcam_active = False
        if self.webcam_window:
            self.webcam_window.destroy()
            self.webcam_window = None
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

def main():
    """Main function to start the GUI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="EfficientNet+Meta Anti-Spoofing GUI")
    parser.add_argument("--model", "-m", type=str, 
                       help="Path to model file (.pth)",
                       default="GohWenKang/efficientNet/efficientnet_models/best_efficientnet_model.pth")
    parser.add_argument("--version", "-v", type=str, choices=['b0', 'b1', 'b2'],
                       help="EfficientNet version", default="b0")
    
    args = parser.parse_args()
    
    print("🚀 Starting EfficientNet+Meta Anti-Spoofing GUI...")
    print(f"📁 Model path: {args.model}")
    print(f"🏗️ EfficientNet version: {args.version}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("💻 Using CPU")
    
    try:
        app = EfficientNetMetaGUI(model_path=args.model)
        app.efficientnet_version = args.version  # Set version from args
        app.run()
    except Exception as e:
        print(f"❌ Error starting GUI: {e}")
        print("\n📋 Troubleshooting:")
        print("1. Make sure you have trained EfficientNet+Meta model (.pth file)")
        print("2. Check if the model path is correct")
        print("3. Ensure required packages are installed:")
        print("   pip install torch torchvision pillow opencv-python matplotlib")
        print("4. For EfficientNet versions:")
        print("   - b0: Best for RTX 3050 (default)")
        print("   - b1: More accurate, requires more memory") 
        print("   - b2: Highest accuracy, highest memory requirements")

if __name__ == "__main__":
    main()