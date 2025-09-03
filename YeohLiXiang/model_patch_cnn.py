# ==============================================================================
# Patch-based CNN for Face Anti-Spoofing
# ==============================================================================
# This model divides input images into patches and processes each patch separately
# before combining the results. This approach helps capture local texture details
# that are important for detecting spoofing attacks.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchBasedCNN(nn.Module):
    """
    Patch-based CNN that divides images into patches and processes them separately.
    Each patch is processed by a shared CNN backbone, then results are combined.
    """
    
    def __init__(self, num_classes=2, patch_size=32, num_patches=16, dropout_rate=0.5):
        super(PatchBasedCNN, self).__init__()
        
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patches_per_dim = int(math.sqrt(num_patches))  # Assuming square grid
        
        # Shared CNN backbone for processing individual patches
        self.patch_cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # Calculate the feature size after convolutions
        # For patch_size=32: 32 -> 16 -> 8 -> 4, so 4*4*128 = 2048
        conv_output_size = (patch_size // 8) * (patch_size // 8) * 128
        
        # Patch feature extractor
        self.patch_fc = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        
        # Patch aggregation layer
        self.aggregation = nn.Sequential(
            nn.Linear(128 * num_patches, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        
        # Final classifier
        self.classifier = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
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
    
    def extract_patches(self, x):
        """
        Extract patches from input images
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        Returns:
            patches: Tensor of shape (batch_size, num_patches, channels, patch_size, patch_size)
        """
        batch_size, channels, height, width = x.shape
        
        # Calculate step size for patch extraction
        step_h = (height - self.patch_size) // (self.patches_per_dim - 1) if self.patches_per_dim > 1 else 0
        step_w = (width - self.patch_size) // (self.patches_per_dim - 1) if self.patches_per_dim > 1 else 0
        
        patches = []
        for i in range(self.patches_per_dim):
            for j in range(self.patches_per_dim):
                start_h = i * step_h
                start_w = j * step_w
                end_h = start_h + self.patch_size
                end_w = start_w + self.patch_size
                
                # Ensure we don't go out of bounds
                if end_h > height:
                    start_h = height - self.patch_size
                    end_h = height
                if end_w > width:
                    start_w = width - self.patch_size
                    end_w = width
                
                patch = x[:, :, start_h:end_h, start_w:end_w]
                patches.append(patch)
        
        return torch.stack(patches, dim=1)  # Shape: (batch_size, num_patches, channels, patch_size, patch_size)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # Extract patches
        patches = self.extract_patches(x)  # (batch_size, num_patches, channels, patch_size, patch_size)
        
        # Process each patch through the shared CNN
        patch_features = []
        for i in range(self.num_patches):
            patch = patches[:, i]  # (batch_size, channels, patch_size, patch_size)
            
            # Pass through CNN
            features = self.patch_cnn(patch)  # (batch_size, 128, patch_size//8, patch_size//8)
            features = features.view(batch_size, -1)  # Flatten
            
            # Extract patch-level features
            patch_feat = self.patch_fc(features)  # (batch_size, 128)
            patch_features.append(patch_feat)
        
        # Concatenate all patch features
        combined_features = torch.cat(patch_features, dim=1)  # (batch_size, 128 * num_patches)
        
        # Aggregate features
        aggregated = self.aggregation(combined_features)  # (batch_size, 256)
        
        # Final classification
        logits = self.classifier(aggregated)  # (batch_size, num_classes)
        
        return logits


class PatchDepthCNN(nn.Module):
    """
    Enhanced Patch-based CNN that also considers depth information
    and uses attention mechanism to weight patch importance
    """
    
    def __init__(self, num_classes=2, patch_size=32, num_patches=16, dropout_rate=0.5):
        super(PatchDepthCNN, self).__init__()
        
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patches_per_dim = int(math.sqrt(num_patches))
        
        # Enhanced patch CNN with depth-aware features
        self.patch_cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )
        
        # Calculate feature size
        conv_output_size = (patch_size // 8) * (patch_size // 8) * 256
        
        # Patch feature extractor
        self.patch_fc = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        
        # Attention mechanism for patch weighting
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Final aggregation and classification
        self.aggregation = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        
        self.classifier = nn.Linear(256, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
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
    
    def extract_patches(self, x):
        """Extract patches from input images"""
        batch_size, channels, height, width = x.shape
        
        step_h = (height - self.patch_size) // (self.patches_per_dim - 1) if self.patches_per_dim > 1 else 0
        step_w = (width - self.patch_size) // (self.patches_per_dim - 1) if self.patches_per_dim > 1 else 0
        
        patches = []
        for i in range(self.patches_per_dim):
            for j in range(self.patches_per_dim):
                start_h = i * step_h
                start_w = j * step_w
                end_h = start_h + self.patch_size
                end_w = start_w + self.patch_size
                
                if end_h > height:
                    start_h = height - self.patch_size
                    end_h = height
                if end_w > width:
                    start_w = width - self.patch_size
                    end_w = width
                
                patch = x[:, :, start_h:end_h, start_w:end_w]
                patches.append(patch)
        
        return torch.stack(patches, dim=1)
    
    def forward(self, x):
        """Forward pass with attention-weighted patch aggregation"""
        batch_size = x.size(0)
        
        # Extract patches
        patches = self.extract_patches(x)
        
        # Process patches and compute attention weights
        patch_features = []
        attention_weights = []
        
        for i in range(self.num_patches):
            patch = patches[:, i]
            
            # Extract features
            features = self.patch_cnn(patch)
            features = features.view(batch_size, -1)
            patch_feat = self.patch_fc(features)
            
            # Compute attention weight
            attention_weight = self.attention(patch_feat)
            
            patch_features.append(patch_feat)
            attention_weights.append(attention_weight)
        
        # Stack features and weights
        patch_features = torch.stack(patch_features, dim=1)  # (batch_size, num_patches, 256)
        attention_weights = torch.stack(attention_weights, dim=1)  # (batch_size, num_patches, 1)
        
        # Apply attention weights
        weighted_features = patch_features * attention_weights  # (batch_size, num_patches, 256)
        
        # Global average pooling with attention
        aggregated = torch.mean(weighted_features, dim=1)  # (batch_size, 256)
        
        # Final processing
        aggregated = self.aggregation(aggregated)
        logits = self.classifier(aggregated)
        
        return logits


def create_patch_cnn(model_type='patch', num_classes=2, patch_size=32, num_patches=16, dropout_rate=0.5):
    """
    Factory function to create patch-based CNN models
    
    Args:
        model_type: 'patch' for basic patch CNN, 'patch_depth' for enhanced version
        num_classes: Number of output classes
        patch_size: Size of each patch
        num_patches: Number of patches to extract
        dropout_rate: Dropout rate for regularization
    
    Returns:
        model: The created model
    """
    if model_type == 'patch':
        return PatchBasedCNN(num_classes, patch_size, num_patches, dropout_rate)
    elif model_type == 'patch_depth':
        return PatchDepthCNN(num_classes, patch_size, num_patches, dropout_rate)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 112, 112).to(device)
    
    # Test basic patch CNN
    print("Testing Patch-based CNN...")
    patch_model = create_patch_cnn('patch').to(device)
    output = patch_model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in patch_model.parameters()):,}")
    
    # Test enhanced patch depth CNN
    print("\nTesting Patch-Depth CNN...")
    patch_depth_model = create_patch_cnn('patch_depth').to(device)
    output = patch_depth_model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in patch_depth_model.parameters()):,}")
