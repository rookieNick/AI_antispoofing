import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# --- Adaptive Central Difference Convolution ---
class AdaptiveCDCConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, dilation=1, bias=True, theta=0.7):
        super(AdaptiveCDCConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                             padding, dilation, bias=bias)
        self.theta = nn.Parameter(torch.tensor(theta, dtype=torch.float32))
        self.edge_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                  stride=stride, padding=0, bias=False)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        out_normal = self.conv(x)
        if abs(self.theta) > 1e-8:
            kernel_diff = self.conv.weight.sum(dim=(2, 3), keepdim=True)
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=None, 
                               stride=self.conv.stride, padding=0)
            if out_diff.size() != out_normal.size():
                out_diff = F.interpolate(out_diff, size=out_normal.shape[2:], 
                                       mode='bilinear', align_corners=False)
            cdc_out = out_normal - self.theta * out_diff
        else:
            cdc_out = out_normal
        edge_out = self.edge_conv(x)
        combined_out = cdc_out + 0.3 * edge_out
        attention_weights = self.attention(combined_out)
        final_out = combined_out * attention_weights
        return final_out

# --- Enhanced CDC Block with Multi-Scale Feature Fusion ---
class EnhancedCDCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, theta=0.7, use_se=True):
        super(EnhancedCDCBlock, self).__init__()
        self.conv1 = AdaptiveCDCConv2d(in_channels, out_channels//2, 3, stride, 1, theta=theta)
        self.conv1_alt = nn.Conv2d(in_channels, out_channels//2, 1, stride, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = AdaptiveCDCConv2d(out_channels, out_channels, 3, 1, 1, theta=theta)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Squeeze-and-Excitation module
        self.use_se = use_se
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 16, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 16, out_channels, 1),
                nn.Sigmoid()
            )
        
        # Shortcut connection
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        # Multi-scale feature extraction
        out1 = self.conv1(x)
        out1_alt = self.conv1_alt(x)
        out = torch.cat([out1, out1_alt], dim=1)
        out = F.relu(self.bn1(out))
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply SE attention if enabled
        if self.use_se:
            se_weight = self.se(out)
            out = out * se_weight
        
        # Add shortcut connection
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        
        out += shortcut
        out = F.relu(out)
        return out

# --- Advanced CDCN Model with Enhanced Feature Fusion ---
class AdvancedCDCN(nn.Module):
    def __init__(self, num_classes=2, theta=0.7, map_size=32, dropout_rate=0.5):
        super(AdvancedCDCN, self).__init__()
        self.map_size = map_size
        
        # Enhanced stem network (matches checkpoint structure)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        # Progressive CDC layers with different theta values (matches checkpoint)
        self.layer1 = self._make_layer(64, 64, 2, 1, theta * 1.0)
        self.layer2 = self._make_layer(64, 128, 2, 2, theta * 0.8)
        self.layer3 = self._make_layer(128, 256, 3, 2, theta * 0.6)
        self.layer4 = self._make_layer(256, 512, 2, 2, theta * 0.4)
        
        # Multi-scale feature aggregation (matches checkpoint)
        self.feature_aggregation = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Enhanced classification head (matches checkpoint)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        # Enhanced depth prediction network (matches checkpoint)
        self.depth_predictor = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, blocks, stride, theta):
        layers = []
        layers.append(EnhancedCDCBlock(in_channels, out_channels, stride, theta))
        for _ in range(1, blocks):
            layers.append(EnhancedCDCBlock(out_channels, out_channels, 1, theta))
        return nn.Sequential(*layers)
    
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature extraction
        x = self.stem(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature_map = self.layer4(x)
        
        # Feature aggregation for depth prediction
        aggregated_features = self.feature_aggregation(feature_map)
        
        # Classification pathway
        avg_pool_features = self.global_pool(feature_map)
        max_pool_features = self.max_pool(feature_map)
        
        # Combine average and max pooling
        combined_features = torch.cat([avg_pool_features, max_pool_features], dim=1)
        combined_features = torch.flatten(combined_features, 1)
        
        cls_output = self.classifier(combined_features)
        
        # Depth prediction pathway
        depth_map = self.depth_predictor(aggregated_features)
        depth_map = F.interpolate(depth_map, size=(self.map_size, self.map_size), 
                                 mode='bilinear', align_corners=False)
        
        return cls_output, depth_map

def load_pretrained_cdcn(model_path=None):
    """Load a pretrained CDCN model."""
    model = AdvancedCDCN(num_classes=2)
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
                    print(f"Model accuracy: {checkpoint.get('accuracy', 'unknown'):.4f}")
                elif 'state_dict' in checkpoint:
                    # Alternative checkpoint format
                    state_dict = checkpoint['state_dict']
                else:
                    # Direct state dict
                    state_dict = checkpoint
            else:
                # Assume it's a direct state dict
                state_dict = checkpoint
            
            model.load_state_dict(state_dict)
            model.eval()
            print(f"Loaded pretrained CDCN model from {model_path}")
            
        except Exception as e:
            print(f"Warning: Could not load CDCN model weights: {str(e)}")
            print("Using randomly initialized CDCN model")
    else:
        print(f"Warning: No pretrained model found at {model_path}")
        print("Using randomly initialized CDCN model")
    
    return model
