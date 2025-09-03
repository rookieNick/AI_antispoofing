
import torch
import torch.nn as nn

# OptimizedCNN: A simple but modern convolutional neural network for binary classification
# This model is designed for anti-spoofing tasks (e.g., live vs spoof face detection)
# It uses SiLU activations, batch normalization, dropout, and global average pooling for regularization and stability.

class OptimizedCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(OptimizedCNN, self).__init__()
        # Feature extraction layers
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 2D convolution: extracts local features from input image
            nn.BatchNorm2d(32),                           # BatchNorm: normalizes activations, speeds up training
            nn.SiLU(inplace=True),                        # SiLU (Swish): smooth, non-linear activation, helps gradients
            nn.MaxPool2d(2, 2),                           # MaxPool2d: reduces spatial size by taking max in 2x2 window
            nn.Dropout2d(0.3),                            # Dropout2d: randomly zeroes channels, prevents overfitting

            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # More filters, deeper features
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.4),

            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # Even deeper features
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.5),

            # Global Average Pooling: reduces each feature map to a single value (spatial average)
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Classifier: Fully connected layers for final prediction
        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # Flattens 4D tensor to 2D for linear layers
            nn.Linear(128, 256),                          # Linear: dense layer, learns global patterns
            nn.BatchNorm1d(256),                          # BatchNorm: normalizes activations
            nn.SiLU(inplace=True),                        # SiLU activation
            nn.Dropout(0.6),                              # Dropout: regularization
            nn.Linear(256, 64),                           # Linear: reduces to 64 features
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)                    # Final output: logits for each class
        )

        # Note: L2 regularization is applied via optimizer's weight_decay parameter

    def forward(self, x):
        # Forward pass: extract features, then classify
        x = self.features(x)
        x = self.classifier(x)
        return x
