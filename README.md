# AI Anti-Spoofing Project

A comprehensive face anti-spoofing system that detects whether a face image is live (real person) or spoofed (fake/photo attack). This project includes multiple CNN implementations, model training scripts, and a user-friendly GUI application.

## 📁 Project Structure

```
ai asgmt/
├── README.md                    # This file
├── dataset/                     # Dataset files and utilities
│   ├── casia-fasd.zip          # Original dataset archive
│   ├── casia-fasd/             # Extracted dataset
│   │   ├── train/              # Training data
│   │   │   ├── live/           # Live face images
│   │   │   └── spoof/          # Spoofed face images
│   │   └── test/               # Testing data
│   │       ├── live/           # Live face test images
│   │       └── spoof/          # Spoofed face test images
│   ├── count.py                # Dataset counting utility
│   ├── data_check.py           # Dataset validation script
│   ├── readme.md               # Dataset documentation
│   └── requirements.txt        # Dataset processing dependencies
├── main/                       # Main application (GUI + Backend)
│   ├── main.py                 # Application entry point
│   ├── requirements.txt        # Main app dependencies
│   ├── backend/                # Model inference backend
│   │   └── predictor.py        # CNN model prediction logic
│   ├── controller/             # Application controllers
│   │   └── image_controller.py # Image processing controller
│   ├── model/                  # Trained models storage
│   │   └── CNN_YeohLiXiang/    # CNN model files
│   │       ├── model.py        # CNN architecture definition
│   │       └── cnn_pytorch.pth # Trained model weights
│   └── ui/                     # User interface
│       └── simple_ui.py        # Tkinter GUI application
├── YeohLiXiang/                # CNN training implementation
│   ├── model.py                # CNN model architecture
│   ├── train.py                # Training script
│   ├── test.py                 # Testing script
│   ├── test_one.py             # Single image testing
│   ├── train_fine_tuned.py     # Fine-tuning script
│   ├── plot_utils.py           # Visualization utilities
│   ├── requirements.txt        # Training dependencies
│   ├── model/                  # Model checkpoints
│   │   └── cnn_pytorch.pth     # Trained model weights
│   └── results/                # Training/testing results
│       ├── test_3_20250901/    # Test results
│       └── train_3_20250901/   # Training metrics
├── GohWenKang/                 # Alternative implementation
│   └── readme.md               # Implementation documentation
└── YeongChingZhou/             # YOLO implementation
    ├── readme.md               # YOLO documentation
    └── yolo_from_scratch.py    # YOLO implementation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12 or higher
- Git (for cloning the repository)
- Webcam (optional, for real-time detection)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rookieNick/AI_antispoofing.git
   cd AI_antispoofing
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   # Install main application dependencies
   cd main
   pip install -r requirements.txt
   ```

### Running the Application

1. **Ensure virtual environment is activated:**
   ```bash
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Navigate to main directory and run:**
   ```bash
   cd main
   python main.py
   ```

3. **Using the GUI:**
   - **Import Image**: Click "Import Image" to select a face image for analysis
   - **Select Model**: Choose "CNN" from the dropdown (other models not yet implemented)
   - **Predict**: Click "Predict" to analyze the imported image
   - **Webcam**: Click "Start Webcam & Recognition" for real-time detection
   - **Results**: View predictions with confidence scores in the results panel

## 🎯 Features

### Current Features
- **CNN-based Anti-Spoofing**: Optimized CNN architecture for face anti-spoofing
- **GUI Application**: User-friendly Tkinter interface
- **Image Analysis**: Import and analyze individual images
- **Webcam Support**: Real-time face anti-spoofing detection
- **Confidence Scores**: Detailed prediction confidence levels
- **Multiple Model Support**: Framework ready for CNN, CDCN, and ViT models

### Model Performance
- **Architecture**: Optimized CNN with batch normalization and dropout
- **Input Size**: 112x112 pixels
- **Classes**: Live (real face) vs Spoof (fake/photo attack)
- **Confidence**: Returns probability scores for predictions

## 🔧 Model Training

### Training Your Own Model

1. **Prepare dataset:**
   ```bash
   cd dataset
   python data_check.py  # Validate dataset structure
   python count.py       # Count images in dataset
   ```

2. **Train the model:**
   ```bash
   cd YeohLiXiang
   pip install -r requirements.txt
   ## 🏗️ Environment Setup: Per-Folder Virtual Environments

   To avoid dependency conflicts and keep each part of the project isolated, create and activate a separate virtual environment in each major folder (`main`, `YeohLiXiang`, and `dataset`).

   ### 1. Main Application
   ```powershell
   cd main
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

   ### 2. YeohLiXiang (CNN Training & Testing)
   ```powershell
   cd YeohLiXiang
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

   ### 3. Dataset Utilities
   ```powershell
   cd dataset
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

   **Note:**
   - Always activate the correct environment before running scripts in each folder.
   - If you use macOS/Linux, replace `venv\Scripts\activate` with `source venv/bin/activate`.
   - This setup ensures each folder has its own dependencies and avoids version conflicts.
   python train.py
   ```

3. **Test the model:**
   ```bash
   python test.py        # Full test set evaluation
   python test_one.py    # Single image testing
   ```

### Model Architecture
The CNN model (`YeohLiXiang/model.py`) features:
- **Convolutional Layers**: Multiple conv layers with batch normalization
- **Pooling**: Max pooling for dimension reduction
- **Dropout**: Regularization to prevent overfitting
- **Dense Layers**: Fully connected layers for classification
- **Optimization**: Adam optimizer with learning rate scheduling

## 📊 Results and Metrics

Training results are automatically saved in:
- `YeohLiXiang/results/train_*/`: Training metrics and plots
- `YeohLiXiang/results/test_*/`: Test results and confusion matrices

Key metrics include:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance metrics
- **ROC Curve**: Receiver Operating Characteristic analysis
- **Confusion Matrix**: Detailed classification breakdown

## 🛠️ Development

### Adding New Models

1. **Create model file** in `main/model/YourModel/`
2. **Update predictor.py** to support your model
3. **Add to UI dropdown** in `simple_ui.py`
4. **Implement prediction logic** in the UI controller

### File Descriptions

#### Core Application Files
- `main/main.py`: Application entry point and initialization
- `main/ui/simple_ui.py`: Main GUI interface using Tkinter
- `main/backend/predictor.py`: Model loading and inference logic
- `main/controller/image_controller.py`: Image processing controller

#### Model Files
- `YeohLiXiang/model.py`: CNN architecture definition
- `YeohLiXiang/train.py`: Model training with advanced features
- `YeohLiXiang/test.py`: Comprehensive model evaluation
- `main/model/CNN_YeohLiXiang/cnn_pytorch.pth`: Trained model weights

#### Utilities
- `dataset/data_check.py`: Dataset validation and structure verification
- `dataset/count.py`: Dataset statistics and counting
- `YeohLiXiang/plot_utils.py`: Training visualization utilities

## 🐛 Troubleshooting

### Common Issues

1. **"Module not found" errors:**
   - Ensure virtual environment is activated
   - Install dependencies: `pip install -r requirements.txt`

2. **Model not loading:**
   - Check if model file exists at `main/model/CNN_YeohLiXiang/cnn_pytorch.pth`
   - Retrain model if needed: `cd YeohLiXiang && python train.py`

3. **Webcam not working:**
   - Check webcam permissions
   - Ensure no other applications are using the webcam
   - Try different webcam index in code (change from 0 to 1)

4. **Poor predictions:**
   - Ensure input images are clear face photos
   - Check image preprocessing in `predictor.py`
   - Consider retraining with more data

### Error Messages

- **"Error: Could not open webcam"**: Check webcam connection and permissions
- **"No image selected"**: Import an image before prediction
- **"Model file not found"**: Run training script to generate model weights

## 📝 License

This project is part of an AI coursework assignment. Please refer to your institution's academic policies regarding code sharing and collaboration.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the code comments for implementation details
3. Consult the individual README files in subdirectories
4. Check the results folder for training metrics and examples

## 🔮 Future Enhancements

- **CDCN Model**: Central Difference Convolutional Network implementation
- **Vision Transformer**: ViT-based anti-spoofing model
- **Real-time Performance**: Optimized inference for webcam streams
- **Mobile Deployment**: Export models for mobile applications
- **Advanced Augmentation**: More sophisticated data augmentation techniques
- **Ensemble Methods**: Combine multiple models for better accuracy
