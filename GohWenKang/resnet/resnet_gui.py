import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import cv2
import os
from datetime import datetime

# Import the ResNet+Attention model classes (same as in training script)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(attention_input))
        return x * attention_map

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction_ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class CBAM(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class AttentionResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, 
                 use_attention=True, reduction_ratio=16):
        super(AttentionResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.use_attention = use_attention
        
        if use_attention:
            self.attention = CBAM(out_channels, reduction_ratio)
            
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_attention:
            out = self.attention(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        
        return out

class ResNetAttention(nn.Module):
    def __init__(self, num_classes=2, layers=[2, 2, 2, 2], base_channels=64, 
                 use_attention=True, dropout_rate=0.3):
        super(ResNetAttention, self).__init__()
        
        self.in_channels = base_channels
        self.use_attention = use_attention
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=7, stride=2, 
                              padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers with attention
        self.layer1 = self._make_layer(base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 8, layers[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final attention mechanism on global features
        if use_attention:
            self.global_attention = nn.Sequential(
                nn.Linear(base_channels * 8, base_channels * 8 // 4),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(base_channels * 8 // 4, base_channels * 8),
                nn.Sigmoid()
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(base_channels * 8, base_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(base_channels * 2, num_classes)
        )
        
    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(AttentionResidualBlock(self.in_channels, out_channels, 
                                           stride, downsample, self.use_attention))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(AttentionResidualBlock(out_channels, out_channels, 
                                               use_attention=self.use_attention))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual layers with attention
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Apply global attention to features
        if self.use_attention:
            attention_weights = self.global_attention(x)
            x = x * attention_weights
        
        # Classification
        x = self.classifier(x)
        
        return x

class ResNetAttentionGUI:
    def __init__(self, model_path=None):
        self.root = tk.Tk()
        self.root.title("ResNet+Attention Anti-Spoofing Demo")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = None
        self.model_loaded = False
        
        # Default model path
        if model_path is None:
            model_path = "resnet/resnet_attention_models/best_resnet_attention_model.pth"
        
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
        """Load the ResNet+Attention model"""
        try:
            print(f"Loading ResNet+Attention model from {model_path}")
            
            # Initialize model architecture
            self.model = ResNetAttention(
                num_classes=2,
                layers=[3, 4, 6, 3],
                base_channels=64,
                use_attention=True,
                dropout_rate=0.3
            )
            
            # Load state dict
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            self.model_loaded = True
            print("✅ ResNet+Attention model loaded successfully!")
            
            # Update status
            if hasattr(self, 'status_label'):
                self.status_label.config(text=f"Model loaded: {model_path}")
                
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
        
        # Title with ResNet+Attention branding
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, columnspan=4, pady=(0, 15), sticky="ew")
        
        title_label = ttk.Label(title_frame, text="ResNet+Attention Anti-Spoofing Detection", 
                               font=('Arial', 18, 'bold'))
        title_label.grid(row=0, column=0)
        
        subtitle_label = ttk.Label(title_frame, text="Real-time Face Liveness Detection with Attention Mechanism", 
                                  font=('Arial', 11), foreground='gray')
        subtitle_label.grid(row=1, column=0, pady=(5, 0))
        
        # Model status indicator
        self.model_status_frame = ttk.LabelFrame(main_frame, text="Model Status", padding="10")
        self.model_status_frame.grid(row=1, column=0, columnspan=4, pady=(0, 10), sticky="ew")
        
        status_color = "green" if self.model_loaded else "red"
        status_text = "✅ ResNet+Attention Model Loaded" if self.model_loaded else "❌ Model Not Loaded"
        
        self.model_status_label = ttk.Label(self.model_status_frame, text=status_text, 
                                           font=('Arial', 10, 'bold'))
        self.model_status_label.grid(row=0, column=0)
        
        # Device info
        device_text = f"Device: {self.device.type.upper()}"
        if torch.cuda.is_available():
            device_text += f" ({torch.cuda.get_device_name(0)})"
        
        ttk.Label(self.model_status_frame, text=device_text, 
                 font=('Arial', 9)).grid(row=0, column=1, padx=(20, 0))
        
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
        
        # Select Image button
        ttk.Button(control_frame, text="🖼️ Select Image", 
                  command=self.select_image).grid(row=1, column=0, pady=5, sticky="ew")
        
        # Predict button
        self.predict_btn = ttk.Button(control_frame, text="🔍 Predict", 
                                     command=self.predict_image)
        self.predict_btn.grid(row=2, column=0, pady=5, sticky="ew")
        
        # Webcam button
        ttk.Button(control_frame, text="📷 Webcam Demo", 
                  command=self.start_webcam).grid(row=3, column=0, pady=5, sticky="ew")
        
        # Batch Test button
        ttk.Button(control_frame, text="📊 Batch Test", 
                  command=self.batch_test).grid(row=4, column=0, pady=5, sticky="ew")
        
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
        
        # Probability bars frame
        prob_frame = ttk.Frame(self.results_frame)
        prob_frame.grid(row=2, column=0, pady=10, sticky="ew")
        
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
        
        # Processing time
        self.time_label = ttk.Label(self.results_frame, text="", 
                                   font=('Arial', 10), foreground='gray')
        self.time_label.grid(row=3, column=0, pady=(10, 0))
        
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
    
    def select_model(self):
        """Select and load model file"""
        model_path = filedialog.askopenfilename(
            title="Select ResNet+Attention Model File",
            filetypes=[
                ("PyTorch Model", "*.pth"),
                ("All files", "*.*")
            ],
            initialdir="resnet_attention_models"
        )
        
        if model_path:
            self.load_model(model_path)
            self.predict_btn.config(state='normal' if self.model_loaded else 'disabled')
            
            # Update model status
            status_text = "✅ ResNet+Attention Model Loaded" if self.model_loaded else "❌ Failed to Load Model"
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
            self.time_label.config(text="")
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
        self.status_label.config(text="Analyzing image with ResNet+Attention...")
        
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
            
            # Predict with ResNet+Attention model
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            prediction = self.class_names[predicted.item()]
            confidence_score = confidence.item()
            probs = probabilities.cpu().numpy()[0]
            
            # Update UI in main thread
            self.root.after(0, self._update_results, prediction, confidence_score, 
                          probs, processing_time)
            
        except Exception as e:
            self.root.after(0, self._show_error, str(e))
    
    def _update_results(self, prediction, confidence, probabilities, processing_time):
        """Update UI with prediction results"""
        # Stop progress bar
        self.progress.stop()
        
        # Update result labels
        color = "#2e7d32" if prediction == "Live" else "#d32f2f"  # Green for Live, Red for Spoof
        self.result_label.config(text=f"🎯 {prediction}", foreground=color)
        self.confidence_label.config(text=f"Confidence: {confidence:.1%}")
        
        # Update probability bars
        live_prob = probabilities[1] * 100
        spoof_prob = probabilities[0] * 100
        
        self.live_var.set(live_prob)
        self.spoof_var.set(spoof_prob)
        self.live_percent.config(text=f"{live_prob:.1f}%")
        self.spoof_percent.config(text=f"{spoof_prob:.1f}%")
        
        # Update processing time
        self.time_label.config(text=f"Processing time: {processing_time:.1f}ms")
        
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
            
            print(f"\\n🧪 BATCH TESTING {len(image_files)} IMAGES")
            print("=" * 50)
            
            results = []
            
            for i, filename in enumerate(image_files):
                filepath = os.path.join(test_dir, filename)
                
                try:
                    # Load and predict
                    image = Image.open(filepath).convert('RGB')
                    input_tensor = self.transform(image).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                    
                    prediction = self.class_names[predicted.item()]
                    conf_score = confidence.item()
                    
                    results.append({
                        'filename': filename,
                        'prediction': prediction,
                        'confidence': conf_score,
                        'live_prob': probabilities[0][1].item(),
                        'spoof_prob': probabilities[0][0].item()
                    })
                    
                    print(f"{i+1:3d}. {filename:<30} → {prediction:<5} ({conf_score:.1%})")
                    
                except Exception as e:
                    print(f"{i+1:3d}. {filename:<30} → ERROR: {str(e)}")
            
            # Summary
            live_count = sum(1 for r in results if r['prediction'] == 'Live')
            spoof_count = len(results) - live_count
            avg_confidence = np.mean([r['confidence'] for r in results])
            
            print("\\n📊 BATCH TEST SUMMARY:")
            print(f"   Total Images: {len(results)}")
            print(f"   Live Detected: {live_count}")
            print(f"   Spoof Detected: {spoof_count}")
            print(f"   Average Confidence: {avg_confidence:.1%}")
            print("=" * 50)
            
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
        self.webcam_window.title("ResNet+Attention Webcam Demo")
        self.webcam_window.geometry("800x600")
        self.webcam_window.configure(bg='#2b2b2b')
        
        # Webcam display frame
        webcam_frame = ttk.Frame(self.webcam_window, padding="10")
        webcam_frame.pack(expand=True, fill='both')
        
        # Title
        ttk.Label(webcam_frame, text="Live Face Anti-Spoofing Detection", 
                 font=('Arial', 14, 'bold')).pack(pady=(0, 10))
        
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
                        outputs = self.model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                    
                    prediction = self.class_names[predicted.item()]
                    confidence_score = confidence.item()
                    
                    # Add prediction overlay to frame
                    color = (0, 255, 0) if prediction == "Live" else (0, 0, 255)
                    text = f"{prediction}: {confidence_score:.1%}"
                    
                    # Add background rectangle for text
                    cv2.rectangle(frame, (10, 10), (400, 60), (0, 0, 0), -1)
                    cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, color, 2)
                    
                    # Add model info
                    cv2.putText(frame, "ResNet+Attention", (20, 55), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Update status
                    if hasattr(self, 'webcam_status_label'):
                        status_text = f"Detecting: {prediction} ({confidence_score:.1%})"
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
    
    parser = argparse.ArgumentParser(description="ResNet+Attention Anti-Spoofing GUI")
    parser.add_argument("--model", "-m", type=str, 
                       help="Path to model file (.pth)",
                       default="resnet/resnet_attention_models/best_resnet_attention_model.pth")
    
    args = parser.parse_args()
    
    print("🚀 Starting ResNet+Attention Anti-Spoofing GUI...")
    print(f"📁 Model path: {args.model}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("💻 Using CPU")
    
    try:
        app = ResNetAttentionGUI(model_path=args.model)
        app.run()
    except Exception as e:
        print(f"❌ Error starting GUI: {e}")
        print("\n📋 Troubleshooting:")
        print("1. Make sure you have trained ResNet+Attention model (.pth file)")
        print("2. Check if the model path is correct")
        print("3. Ensure required packages are installed:")
        print("   pip install torch torchvision pillow opencv-python matplotlib")

if __name__ == "__main__":
    main()