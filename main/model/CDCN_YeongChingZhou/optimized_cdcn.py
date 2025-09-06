"""
Optimized CDCN Model - Clean Version for Main Application
=========================================================
This is an optimized CDCN model architecture for face anti-spoofing
but without training dependencies like pandas, sklearn, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class OptimizedCDCConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, dilation=1, bias=True, theta=0.7):
        super(OptimizedCDCConv2d, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                             padding, dilation, bias=bias)
        
        # Fixed theta for speed (no learnable parameter)
        self.theta = theta
        
        # Simplified edge convolution
        self.edge_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                  stride=stride, padding=0, bias=False)
        
        # Lightweight attention
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, max(1, out_channels // 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, out_channels // 8), out_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Standard convolution
        conv_out = self.conv(x)
        
        # Simplified central difference (vertical only for speed)
        diff_v = F.conv2d(x, self._get_diff_kernel_v(x), padding=1, groups=1)
        
        # Edge enhancement
        edge_out = self.edge_conv(torch.abs(diff_v))
        
        # Fast combination
        combined = self.theta * conv_out + (1 - self.theta) * edge_out
        
        # Apply lightweight attention
        attention_weights = self.attention(combined)
        output = combined * attention_weights
        
        return output
    
    def _get_diff_kernel_v(self, x):
        if not hasattr(self, '_cached_kernel_v') or self._cached_kernel_v.device != x.device:
            kernel = torch.zeros(x.size(1), x.size(1), 3, 3, device=x.device)
            for i in range(x.size(1)):
                kernel[i, i, 0, 1] = 1.0
                kernel[i, i, 2, 1] = -1.0
            self._cached_kernel_v = kernel
        return self._cached_kernel_v

class OptimizedCDCN(nn.Module):
    """
    Optimized CDCN model architecture for face anti-spoofing.
    Clean version without training dependencies.
    """
    def __init__(self, num_classes=2, input_channels=3):
        super(OptimizedCDCN, self).__init__()
        
        # Optimized CDC blocks for RTX 4050 (batch size 64)
        self.cdc1 = OptimizedCDCConv2d(input_channels, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.1)
        
        self.cdc2 = OptimizedCDCConv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.15)
        
        self.cdc3 = OptimizedCDCConv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.2)
        
        self.cdc4 = OptimizedCDCConv2d(128, 256, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout2d(0.25)
        
        # Optimized classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # CDC feature extraction
        x = F.relu(self.bn1(self.cdc1(x)), inplace=True)
        x = self.pool1(self.dropout1(x))
        
        x = F.relu(self.bn2(self.cdc2(x)), inplace=True)
        x = self.pool2(self.dropout2(x))
        
        x = F.relu(self.bn3(self.cdc3(x)), inplace=True)
        x = self.pool3(self.dropout3(x))
        
        x = F.relu(self.bn4(self.cdc4(x)), inplace=True)
        x = self.pool4(self.dropout4(x))
        
        # Classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

def load_pretrained_optimized_cdcn(model_path=None):
    """Load a pretrained Optimized CDCN model."""
    model = OptimizedCDCN(num_classes=2)
    if model_path is None:
        # Use default path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'optimized_cdcn_best.pth')
    
    if os.path.exists(model_path):
        try:
            # Load checkpoint with weights_only=False for compatibility
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # Full training checkpoint format
                    state_dict = checkpoint['model_state_dict']
                    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                    if 'best_accuracy' in checkpoint:
                        print(f"Model accuracy: {checkpoint['best_accuracy']:.4f}")
                elif 'state_dict' in checkpoint:
                    # Alternative checkpoint format
                    state_dict = checkpoint['state_dict']
                else:
                    # Direct state dict
                    state_dict = checkpoint
            else:
                # Assume it's a direct state dict
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            print(f"✅ Loaded pretrained Optimized CDCN model from {model_path}")
            
        except Exception as e:
            print(f"❌ Warning: Could not load CDCN model weights: {str(e)}")
            print("Using randomly initialized CDCN model")
    else:
        print(f"❌ Warning: No pretrained model found at {model_path}")
        print("Using randomly initialized CDCN model")
    
    return model
