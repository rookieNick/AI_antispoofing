"""
🎯 CDCN Face Anti-Spoofing GUI Demo
================================================================================
Real-time demonstration GUI for trained CDCN model
Features: Webcam testing, Image testing, Batch processing, Results visualization
================================================================================
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import numpy as np
import threading
import time
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from datetime import datetime
import hashlib

# Import the CDCN model architecture
from cdcn_clean import CDCN_RTX4050, OptimizedCDCConv2d

class CDCNDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 CDCN Face Anti-Spoofing Demo - RTX 4050 Optimized")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Model configuration
        self.model_path = r"C:\Users\User\AndroidStudioProjects\AI_antispoofing\YeongChingZhou\CDCN\best_algo\rtx4050_cdcn_results_20250906_165750\rtx4050_cdcn_best.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.is_model_loaded = False
        
        # Camera variables
        self.cap = None
        self.is_camera_on = False
        self.camera_thread = None
        
        # Transform for preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Results storage
        self.test_results = []
        
        # Create GUI
        self.create_gui()
        self.load_model()
        
    def create_gui(self):
        """Create the main GUI interface"""
        # Main title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x', padx=5, pady=5)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="🎯 CDCN Face Anti-Spoofing Demo", 
            font=('Arial', 24, 'bold'),
            bg='#2c3e50', 
            fg='white'
        )
        title_label.pack(expand=True)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_realtime_tab()
        self.create_image_test_tab()
        self.create_batch_test_tab()
        self.create_results_tab()
        self.create_model_info_tab()
        
    def create_realtime_tab(self):
        """Create real-time webcam testing tab"""
        realtime_frame = ttk.Frame(self.notebook)
        self.notebook.add(realtime_frame, text="📹 Real-time Testing")
        
        # Left panel - Camera feed
        left_frame = tk.Frame(realtime_frame, bg='white', relief='raised', bd=2)
        left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        # Camera controls
        controls_frame = tk.Frame(left_frame, bg='white')
        controls_frame.pack(fill='x', padx=10, pady=10)
        
        self.camera_btn = tk.Button(
            controls_frame,
            text="🎥 Start Camera",
            command=self.toggle_camera,
            font=('Arial', 12, 'bold'),
            bg='#27ae60',
            fg='white',
            relief='raised',
            bd=3
        )
        self.camera_btn.pack(side='left', padx=10)
        
        # Camera status
        self.camera_status = tk.Label(
            controls_frame,
            text="📴 Camera Off",
            font=('Arial', 10),
            bg='white',
            fg='red'
        )
        self.camera_status.pack(side='left', padx=20)
        
        # Video display
        self.video_label = tk.Label(left_frame, bg='black', text="Camera feed will appear here")
        self.video_label.pack(padx=10, pady=10, fill='both', expand=True)
        
        # Right panel - Real-time results
        right_frame = tk.Frame(realtime_frame, bg='white', relief='raised', bd=2)
        right_frame.pack(side='right', fill='y', padx=10, pady=10)
        right_frame.configure(width=400)
        right_frame.pack_propagate(False)
        
        # Results title
        results_title = tk.Label(
            right_frame,
            text="🔍 Real-time Analysis",
            font=('Arial', 16, 'bold'),
            bg='white'
        )
        results_title.pack(pady=10)
        
        # Prediction display
        self.prediction_frame = tk.Frame(right_frame, bg='white')
        self.prediction_frame.pack(fill='x', padx=10, pady=10)
        
        self.prediction_label = tk.Label(
            self.prediction_frame,
            text="Prediction: --",
            font=('Arial', 14, 'bold'),
            bg='white'
        )
        self.prediction_label.pack()
        
        self.confidence_label = tk.Label(
            self.prediction_frame,
            text="Confidence: --%",
            font=('Arial', 12),
            bg='white'
        )
        self.confidence_label.pack()
        
        # Confidence bar
        self.confidence_var = tk.DoubleVar()
        self.confidence_progress = ttk.Progressbar(
            self.prediction_frame,
            variable=self.confidence_var,
            maximum=100,
            length=300
        )
        self.confidence_progress.pack(pady=10)
        
        # Statistics
        stats_frame = tk.LabelFrame(right_frame, text="📊 Session Statistics", bg='white')
        stats_frame.pack(fill='x', padx=10, pady=10)
        
        self.stats_text = tk.Text(stats_frame, height=8, width=40, font=('Courier', 10))
        self.stats_text.pack(padx=5, pady=5)
        
        # Reset button
        reset_btn = tk.Button(
            right_frame,
            text="🔄 Reset Statistics",
            command=self.reset_stats,
            font=('Arial', 10),
            bg='#f39c12',
            fg='white'
        )
        reset_btn.pack(pady=10)
        
    def create_image_test_tab(self):
        """Create single image testing tab"""
        image_frame = ttk.Frame(self.notebook)
        self.notebook.add(image_frame, text="🖼️ Image Testing")
        
        # Left panel - Image display
        left_frame = tk.Frame(image_frame, bg='white', relief='raised', bd=2)
        left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        # Image controls
        controls_frame = tk.Frame(left_frame, bg='white')
        controls_frame.pack(fill='x', padx=10, pady=10)
        
        load_btn = tk.Button(
            controls_frame,
            text="📁 Load Image",
            command=self.load_image,
            font=('Arial', 12, 'bold'),
            bg='#3498db',
            fg='white',
            relief='raised',
            bd=3
        )
        load_btn.pack(side='left', padx=10)
        
        test_btn = tk.Button(
            controls_frame,
            text="🔍 Test Image",
            command=self.test_image,
            font=('Arial', 12, 'bold'),
            bg='#e74c3c',
            fg='white',
            relief='raised',
            bd=3
        )
        test_btn.pack(side='left', padx=10)
        
        # Image display
        self.image_label = tk.Label(left_frame, bg='lightgray', text="Load an image to test")
        self.image_label.pack(padx=10, pady=10, fill='both', expand=True)
        
        # Right panel - Results
        right_frame = tk.Frame(image_frame, bg='white', relief='raised', bd=2)
        right_frame.pack(side='right', fill='y', padx=10, pady=10)
        right_frame.configure(width=400)
        right_frame.pack_propagate(False)
        
        # Results display
        results_title = tk.Label(
            right_frame,
            text="📋 Test Results",
            font=('Arial', 16, 'bold'),
            bg='white'
        )
        results_title.pack(pady=10)
        
        self.image_results = tk.Text(right_frame, height=20, width=45, font=('Courier', 10))
        scrollbar = tk.Scrollbar(right_frame, orient='vertical', command=self.image_results.yview)
        self.image_results.configure(yscrollcommand=scrollbar.set)
        self.image_results.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        scrollbar.pack(side='right', fill='y')
        
    def create_batch_test_tab(self):
        """Create batch testing tab"""
        batch_frame = ttk.Frame(self.notebook)
        self.notebook.add(batch_frame, text="📂 Batch Testing")
        
        # Controls
        controls_frame = tk.Frame(batch_frame, bg='white', relief='raised', bd=2)
        controls_frame.pack(fill='x', padx=10, pady=10)
        
        folder_btn = tk.Button(
            controls_frame,
            text="📁 Select Folder",
            command=self.select_batch_folder,
            font=('Arial', 12, 'bold'),
            bg='#9b59b6',
            fg='white'
        )
        folder_btn.pack(side='left', padx=10, pady=10)
        
        batch_test_btn = tk.Button(
            controls_frame,
            text="🚀 Start Batch Test",
            command=self.start_batch_test,
            font=('Arial', 12, 'bold'),
            bg='#e67e22',
            fg='white'
        )
        batch_test_btn.pack(side='left', padx=10, pady=10)
        
        # Progress bar
        self.batch_progress_var = tk.DoubleVar()
        self.batch_progress = ttk.Progressbar(
            controls_frame,
            variable=self.batch_progress_var,
            maximum=100,
            length=300
        )
        self.batch_progress.pack(side='right', padx=10, pady=10)
        
        # Results display
        self.batch_results = tk.Text(batch_frame, font=('Courier', 10))
        batch_scrollbar = tk.Scrollbar(batch_frame, orient='vertical', command=self.batch_results.yview)
        self.batch_results.configure(yscrollcommand=batch_scrollbar.set)
        self.batch_results.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        batch_scrollbar.pack(side='right', fill='y', padx=(0, 10), pady=10)
        
    def create_results_tab(self):
        """Create results visualization tab"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="📊 Results")
        
        # Matplotlib figure
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('CDCN Model Performance Analysis', fontsize=16, fontweight='bold')
        
        self.canvas = FigureCanvasTkAgg(self.fig, results_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
        # Controls
        viz_controls = tk.Frame(results_frame, bg='white')
        viz_controls.pack(fill='x', padx=10, pady=5)
        
        update_viz_btn = tk.Button(
            viz_controls,
            text="🔄 Update Visualizations",
            command=self.update_visualizations,
            font=('Arial', 10, 'bold'),
            bg='#1abc9c',
            fg='white'
        )
        update_viz_btn.pack(side='left', padx=10)
        
        save_results_btn = tk.Button(
            viz_controls,
            text="💾 Save Results",
            command=self.save_results,
            font=('Arial', 10, 'bold'),
            bg='#34495e',
            fg='white'
        )
        save_results_btn.pack(side='left', padx=10)
        
    def create_model_info_tab(self):
        """Create model information tab"""
        info_frame = ttk.Frame(self.notebook)
        self.notebook.add(info_frame, text="ℹ️ Model Info")
        
        # Model information
        info_text = tk.Text(info_frame, font=('Courier', 11), wrap='word')
        info_scrollbar = tk.Scrollbar(info_frame, orient='vertical', command=info_text.yview)
        info_text.configure(yscrollcommand=info_scrollbar.set)
        info_text.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        info_scrollbar.pack(side='right', fill='y', padx=(0, 10), pady=10)
        
        # Insert model information
        model_info = self.get_model_info()
        info_text.insert('1.0', model_info)
        info_text.configure(state='disabled')
        
    def load_model(self):
        """Load the trained CDCN model"""
        try:
            print("Loading CDCN model...")
            
            # Initialize model
            self.model = CDCN_RTX4050(num_classes=2)
            
            # Load checkpoint with PyTorch 2.6 compatibility
            try:
                import torch.serialization
                torch.serialization.add_safe_globals([
                    'numpy._core.multiarray.scalar',
                    'numpy.core.multiarray.scalar',
                    'numpy.dtype',
                    'numpy.ndarray'
                ])
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            except Exception:
                # Fallback to weights_only=False for trusted model
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            self.is_model_loaded = True
            
            print(f"✅ Model loaded successfully on {self.device}")
            print(f"Model file: {os.path.basename(self.model_path)}")
            
            # Update GUI status
            self.update_model_status("✅ Model Loaded", "green")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            messagebox.showerror("Model Error", f"Failed to load model:\n{e}")
            self.is_model_loaded = False
            self.update_model_status("❌ Model Error", "red")
    
    def update_model_status(self, text, color):
        """Update model status in GUI"""
        # Add status to title or create status bar
        pass
    
    def predict_image(self, image):
        """Predict whether an image is live or spoof"""
        if not self.is_model_loaded:
            return None, 0.0
        
        try:
            # Preprocess image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Convert to numpy
                pred_class = predicted.cpu().numpy()[0]
                conf_score = confidence.cpu().numpy()[0]
                
                # Classes: 0=Live, 1=Spoof
                prediction = "LIVE" if pred_class == 0 else "SPOOF"
                
                return prediction, conf_score
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if not self.is_camera_on:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start camera feed"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Could not open camera")
                return
            
            self.is_camera_on = True
            self.camera_btn.config(text="🛑 Stop Camera", bg='#e74c3c')
            self.camera_status.config(text="📹 Camera On", fg='green')
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to start camera: {e}")
    
    def stop_camera(self):
        """Stop camera feed"""
        self.is_camera_on = False
        if self.cap:
            self.cap.release()
        
        self.camera_btn.config(text="🎥 Start Camera", bg='#27ae60')
        self.camera_status.config(text="📴 Camera Off", fg='red')
        self.video_label.config(image='', text="Camera feed stopped")
    
    def camera_loop(self):
        """Main camera loop"""
        frame_count = 0
        live_count = 0
        spoof_count = 0
        
        while self.is_camera_on:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Resize for display
            display_frame = cv2.resize(frame, (640, 480))
            
            # Predict every 10 frames for performance
            if frame_count % 10 == 0 and self.is_model_loaded:
                prediction, confidence = self.predict_image(frame)
                
                if prediction:
                    # Update counters
                    if prediction == "LIVE":
                        live_count += 1
                        color = (0, 255, 0)  # Green for live
                    else:
                        spoof_count += 1
                        color = (0, 0, 255)  # Red for spoof
                    
                    # Draw prediction on frame
                    cv2.putText(display_frame, f"{prediction}: {confidence:.2f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Update GUI
                    self.root.after(0, self.update_realtime_gui, prediction, confidence, live_count, spoof_count, frame_count)
            
            # Convert frame for tkinter
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            tk_image = ImageTk.PhotoImage(pil_image)
            
            # Update video label
            self.root.after(0, self.update_video_label, tk_image)
            
            time.sleep(0.03)  # ~30 FPS
    
    def update_video_label(self, tk_image):
        """Update video display in GUI"""
        self.video_label.config(image=tk_image)
        self.video_label.image = tk_image  # Keep reference
    
    def update_realtime_gui(self, prediction, confidence, live_count, spoof_count, frame_count):
        """Update real-time results GUI"""
        # Update prediction
        color = 'green' if prediction == "LIVE" else 'red'
        self.prediction_label.config(text=f"Prediction: {prediction}", fg=color)
        self.confidence_label.config(text=f"Confidence: {confidence:.1%}")
        
        # Update confidence bar
        self.confidence_var.set(confidence * 100)
        
        # Update statistics
        stats_text = f"""Frame Count: {frame_count}
Live Predictions: {live_count}
Spoof Predictions: {spoof_count}
Live Rate: {live_count/(live_count+spoof_count)*100:.1f}%
Spoof Rate: {spoof_count/(live_count+spoof_count)*100:.1f}%
Last Update: {datetime.now().strftime('%H:%M:%S')}
"""
        self.stats_text.delete('1.0', 'end')
        self.stats_text.insert('1.0', stats_text)
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats_text.delete('1.0', 'end')
        self.stats_text.insert('1.0', "Statistics reset\nStart camera to begin testing")
    
    def load_image(self):
        """Load image for testing"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                # Load and display image
                image = Image.open(file_path)
                
                # Resize for display
                display_size = (400, 400)
                display_image = image.copy()
                display_image.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                tk_image = ImageTk.PhotoImage(display_image)
                self.image_label.config(image=tk_image, text="")
                self.image_label.image = tk_image  # Keep reference
                
                # Store original image for testing
                self.current_image = image
                self.current_image_path = file_path
                
            except Exception as e:
                messagebox.showerror("Image Error", f"Failed to load image: {e}")
    
    def test_image(self):
        """Test loaded image"""
        if not hasattr(self, 'current_image'):
            messagebox.showwarning("No Image", "Please load an image first")
            return
        
        if not self.is_model_loaded:
            messagebox.showerror("Model Error", "Model not loaded")
            return
        
        try:
            # Get prediction
            prediction, confidence = self.predict_image(self.current_image)
            
            if prediction:
                # Store result
                result = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'file': os.path.basename(self.current_image_path),
                    'prediction': prediction,
                    'confidence': confidence
                }
                self.test_results.append(result)
                
                # Display result
                result_text = f"""
=== IMAGE TEST RESULT ===
File: {result['file']}
Time: {result['timestamp']}
Prediction: {result['prediction']}
Confidence: {result['confidence']:.4f} ({result['confidence']:.1%})
Status: {'✅ GENUINE' if prediction == 'LIVE' else '❌ SPOOFING DETECTED'}

File Path: {self.current_image_path}
Image Size: {self.current_image.size}
Device: {self.device}
Model: CDCN_RTX4050

{'='*50}
"""
                
                self.image_results.insert('1.0', result_text)
                
                # Show popup result
                status = "✅ GENUINE FACE" if prediction == "LIVE" else "❌ SPOOFING DETECTED"
                messagebox.showinfo("Test Result", f"{status}\nConfidence: {confidence:.1%}")
                
            else:
                messagebox.showerror("Prediction Error", "Failed to predict image")
                
        except Exception as e:
            messagebox.showerror("Test Error", f"Failed to test image: {e}")
    
    def select_batch_folder(self):
        """Select folder for batch testing"""
        folder_path = filedialog.askdirectory(title="Select Folder with Images")
        if folder_path:
            self.batch_folder = folder_path
            self.batch_results.insert('end', f"Selected folder: {folder_path}\n")
    
    def start_batch_test(self):
        """Start batch testing"""
        if not hasattr(self, 'batch_folder'):
            messagebox.showwarning("No Folder", "Please select a folder first")
            return
        
        if not self.is_model_loaded:
            messagebox.showerror("Model Error", "Model not loaded")
            return
        
        # Start batch testing in thread
        threading.Thread(target=self.batch_test_worker, daemon=True).start()
    
    def batch_test_worker(self):
        """Worker function for batch testing"""
        try:
            # Get all image files
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            image_files = []
            
            for file in os.listdir(self.batch_folder):
                if file.lower().endswith(image_extensions):
                    image_files.append(os.path.join(self.batch_folder, file))
            
            total_files = len(image_files)
            if total_files == 0:
                self.root.after(0, lambda: messagebox.showinfo("No Images", "No image files found in folder"))
                return
            
            # Process images
            live_count = 0
            spoof_count = 0
            results = []
            
            for i, file_path in enumerate(image_files):
                try:
                    # Load image
                    image = Image.open(file_path)
                    prediction, confidence = self.predict_image(image)
                    
                    if prediction:
                        if prediction == "LIVE":
                            live_count += 1
                        else:
                            spoof_count += 1
                        
                        result = {
                            'file': os.path.basename(file_path),
                            'prediction': prediction,
                            'confidence': confidence
                        }
                        results.append(result)
                        
                        # Update progress
                        progress = ((i + 1) / total_files) * 100
                        self.root.after(0, self.update_batch_progress, progress, file_path, prediction, confidence)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
            
            # Show final results
            final_text = f"""
=== BATCH TEST COMPLETED ===
Total Images: {total_files}
Live Predictions: {live_count}
Spoof Predictions: {spoof_count}
Live Rate: {live_count/total_files*100:.1f}%
Spoof Rate: {spoof_count/total_files*100:.1f}%

Detailed Results:
"""
            
            for result in results:
                final_text += f"{result['file']}: {result['prediction']} ({result['confidence']:.2f})\n"
            
            self.root.after(0, lambda: self.batch_results.insert('end', final_text))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Batch Error", f"Batch testing failed: {e}"))
    
    def update_batch_progress(self, progress, file_path, prediction, confidence):
        """Update batch testing progress"""
        self.batch_progress_var.set(progress)
        status = f"Processing: {os.path.basename(file_path)} -> {prediction} ({confidence:.2f})\n"
        self.batch_results.insert('end', status)
        self.batch_results.see('end')
    
    def update_visualizations(self):
        """Update result visualizations"""
        if not self.test_results:
            messagebox.showinfo("No Data", "No test results to visualize")
            return
        
        # Clear previous plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
        
        # Extract data
        predictions = [r['prediction'] for r in self.test_results]
        confidences = [r['confidence'] for r in self.test_results]
        
        # Plot 1: Prediction distribution
        pred_counts = {pred: predictions.count(pred) for pred in set(predictions)}
        self.ax1.pie(pred_counts.values(), labels=pred_counts.keys(), autopct='%1.1f%%')
        self.ax1.set_title('Prediction Distribution')
        
        # Plot 2: Confidence histogram
        self.ax2.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        self.ax2.set_xlabel('Confidence Score')
        self.ax2.set_ylabel('Frequency')
        self.ax2.set_title('Confidence Score Distribution')
        
        # Plot 3: Confidence by prediction
        live_conf = [r['confidence'] for r in self.test_results if r['prediction'] == 'LIVE']
        spoof_conf = [r['confidence'] for r in self.test_results if r['prediction'] == 'SPOOF']
        
        if live_conf and spoof_conf:
            self.ax3.boxplot([live_conf, spoof_conf], labels=['LIVE', 'SPOOF'])
            self.ax3.set_ylabel('Confidence Score')
            self.ax3.set_title('Confidence by Prediction')
        
        # Plot 4: Timeline
        times = [i for i in range(len(self.test_results))]
        colors = ['green' if p == 'LIVE' else 'red' for p in predictions]
        self.ax4.scatter(times, confidences, c=colors, alpha=0.6)
        self.ax4.set_xlabel('Test Number')
        self.ax4.set_ylabel('Confidence Score')
        self.ax4.set_title('Test Results Timeline')
        
        self.canvas.draw()
    
    def save_results(self):
        """Save test results to file"""
        if not self.test_results:
            messagebox.showinfo("No Data", "No test results to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("CDCN Face Anti-Spoofing Test Results\n")
                    f.write("="*50 + "\n")
                    for result in self.test_results:
                        f.write(f"{result['timestamp']}, {result['file']}, {result['prediction']}, {result['confidence']:.4f}\n")
                
                messagebox.showinfo("Saved", f"Results saved to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save results: {e}")
    
    def get_model_info(self):
        """Get model information text"""
        # Calculate model file hash for integrity check
        try:
            with open(self.model_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            file_size = os.path.getsize(self.model_path) / (1024*1024)  # MB
        except:
            file_hash = "Unable to calculate"
            file_size = 0
        
        info = f"""
🎯 CDCN Face Anti-Spoofing Model Information
{'='*60}

Model Architecture: Central Difference Convolutional Network (CDCN)
Optimization: RTX 4050 Optimized
Input Size: 224x224x3
Output Classes: 2 (Live=0, Spoof=1)

Model File Details:
├─ Path: {self.model_path}
├─ Size: {file_size:.2f} MB
├─ SHA256: {file_hash[:32]}...
└─ Device: {self.device}

Architecture Details:
{'─'*30}
Layer 1: OptimizedCDCConv2d(3→32) + BatchNorm + MaxPool + Dropout(0.1)
Layer 2: OptimizedCDCConv2d(32→64) + BatchNorm + MaxPool + Dropout(0.15)
Layer 3: OptimizedCDCConv2d(64→128) + BatchNorm + MaxPool + Dropout(0.2)
Layer 4: OptimizedCDCConv2d(128→256) + BatchNorm + MaxPool + Dropout(0.25)

Classifier:
├─ Global Average Pooling
├─ FC1: 256→128 + BatchNorm + ReLU + Dropout(0.3)
├─ FC2: 128→64 + BatchNorm + ReLU + Dropout(0.2)
└─ FC3: 64→2 (Output)

Central Difference Convolution Features:
• Enhanced edge detection for spoofing artifacts
• Lightweight attention mechanism
• Optimized for real-time inference
• Micro-batching support for memory efficiency

Training Configuration:
├─ Batch Size: 64 (effective via micro-batching)
├─ Epochs: 20
├─ Optimizer: AdamW
├─ Scheduler: OneCycleLR
├─ Data Augmentation: RandomCrop, Flip, Rotation, ColorJitter
└─ Label Smoothing: 0.1

Performance Optimizations:
• RTX 4050 6GB memory optimized
• CUDA acceleration enabled
• Batch normalization for stability
• Dropout for regularization
• Gradient clipping for training stability

Usage Instructions:
1. Real-time Testing: Use webcam for live detection
2. Image Testing: Load single images for analysis
3. Batch Testing: Process multiple images from folder
4. Results: View statistics and confidence scores

Model Status: {'✅ Loaded' if self.is_model_loaded else '❌ Not Loaded'}
{'='*60}
        """
        return info
    
    def __del__(self):
        """Cleanup when closing"""
        if self.is_camera_on:
            self.stop_camera()

# Main execution
if __name__ == "__main__":
    # Check if model file exists
    model_path = r"C:\Users\User\AndroidStudioProjects\AI_antispoofing\YeongChingZhou\CDCN\best_algo\rtx4050_cdcn_results_20250906_165750\rtx4050_cdcn_best.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please check the model path and try again.")
    else:
        # Create and run GUI
        root = tk.Tk()
        app = CDCNDemo(root)
        
        try:
            root.mainloop()
        except KeyboardInterrupt:
            print("\n🛑 Application closed by user")
        except Exception as e:
            print(f"❌ Application error: {e}")
            import traceback
            traceback.print_exc()
