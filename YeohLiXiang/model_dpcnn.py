# ==============================================================================
# DPCNN Model for Face Anti-Spoofing
# ==============================================================================
# Deep Pyramid Convolutional Neural Network implementation
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class DPCNNBlock(nn.Module):
    """Basic DPCNN block with downsampling"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DPCNNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.3)
        
        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=False)
        out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)), inplace=False)
        # Add skip connection (no in-place)
        out = out + F.interpolate(identity, size=out.shape[2:], mode='bilinear', align_corners=False)
        out = self.pool(out)
        return out

class DPCNN(nn.Module):
    """Deep Pyramid CNN for Face Anti-Spoofing"""
    def __init__(self, num_classes=2, input_channels=3, dropout_rate=0.5):
        super(DPCNN, self).__init__()
        
        # Initial feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2)
        )
        
        # Pyramid blocks
        self.block1 = DPCNNBlock(64, 128)
        self.block2 = DPCNNBlock(128, 256)
        self.block3 = DPCNNBlock(256, 512)
        self.block4 = DPCNNBlock(512, 512)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

class LightweightDPCNN(nn.Module):
    """Lightweight version of DPCNN for faster training"""
    def __init__(self, num_classes=2, input_channels=3, dropout_rate=0.5):
        super(LightweightDPCNN, self).__init__()
        
        # Initial feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2)
        )
        
        # Lightweight pyramid blocks
        self.block1 = DPCNNBlock(32, 64)
        self.block2 = DPCNNBlock(64, 128)
        self.block3 = DPCNNBlock(128, 256)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

def create_dpcnn_model(model_type='standard', num_classes=2, dropout_rate=0.5):
    """
    Create DPCNN model
    
    Args:
        model_type: 'standard' or 'lightweight'
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
    
    Returns:
        DPCNN model
    """
    if model_type == 'lightweight':
        return LightweightDPCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    else:
        return DPCNN(num_classes=num_classes, dropout_rate=dropout_rate)

if __name__ == "__main__":
    # Test the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test standard DPCNN
    model = create_dpcnn_model('standard', num_classes=2)
    model = model.to(device)
    
    # Test input
    x = torch.randn(4, 3, 112, 112).to(device)
    output = model(x)
    
    print(f"Standard DPCNN:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test lightweight DPCNN
    model_light = create_dpcnn_model('lightweight', num_classes=2)
    model_light = model_light.to(device)
    
    output_light = model_light(x)
    
    print(f"\nLightweight DPCNN:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_light.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_light.parameters()):,}")
