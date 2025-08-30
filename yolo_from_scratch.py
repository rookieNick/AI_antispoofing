import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
from datetime import datetime, timedelta
import pandas as pd # For saving results to CSV
import time # For tracking execution time
from tqdm import tqdm # For progress bar
import matplotlib.pyplot as plt

# --- ANCHORS (define some reasonable anchors for 224x224 images) ---
# Format: [(w, h), ...] in pixels, normalized to 224x224
ANCHORS = [
    (0.10, 0.10),  # small
    (0.20, 0.20),  # medium
]

# --- Intersection over Union (IoU) function ---
def intersection_over_union(boxes_preds, boxes_labels):
    """
    Calculate intersection over union for bounding boxes.
    boxes_preds: (N, 4) [x, y, w, h] (center format, normalized 0-1)
    boxes_labels: (N, 4) [x, y, w, h] (center format, normalized 0-1)
    Returns: IoU (N,)
    """
    box1_x1 = boxes_preds[..., 0] - boxes_preds[..., 2] / 2
    box1_y1 = boxes_preds[..., 1] - boxes_preds[..., 3] / 2
    box1_x2 = boxes_preds[..., 0] + boxes_preds[..., 2] / 2
    box1_y2 = boxes_preds[..., 1] + boxes_preds[..., 3] / 2
    box2_x1 = boxes_labels[..., 0] - boxes_labels[..., 2] / 2
    box2_y1 = boxes_labels[..., 1] - boxes_labels[..., 3] / 2
    box2_x2 = boxes_labels[..., 0] + boxes_labels[..., 2] / 2
    box2_y2 = boxes_labels[..., 1] + boxes_labels[..., 3] / 2
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union = box1_area + box2_area - intersection + 1e-6
    return intersection / union

# --- Placeholder for Non-Maximum Suppression (NMS) ---

# --- Helper to compute grid size based on input image size and number of poolings ---
def compute_grid_size(input_size, num_poolings=5):
    if isinstance(input_size, int):
        h = w = input_size
    else:
        h, w = input_size
    return (h // (2 ** num_poolings), w // (2 ** num_poolings))

# --- Custom Dataset for YOLO Anti-Spoofing ---
class AntiSpoofingYOLODataset(Dataset):
    def __init__(self, root_dir, S, B, C, anchors, transform=None):
        self.root_dir = root_dir
        self.S = S  # grid size (tuple)
        self.B = B  # number of boxes per cell
        self.C = C  # number of classes
        self.anchors = torch.tensor(anchors, dtype=torch.float32)
        self.transform = transform
        self.class_map = {'live': 0, 'spoof': 1}
        self.samples = []
        for label in ['live', 'spoof']:
            class_dir = os.path.join(root_dir, label)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(class_dir, fname), self.class_map[label]))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Ground truth bounding box (x, y, w, h) relative to image (0-1)
        # For anti-spoofing, we assume the "object" (face) is the whole image.
        gt_x_img, gt_y_img, gt_w_img, gt_h_img = 0.5, 0.5, 1.0, 1.0

        # Determine the responsible grid cell (center cell)
        # Note: int() truncates, which is equivalent to floor for positive numbers.
        cell_x_idx = int(self.S[1] * gt_x_img)
        cell_y_idx = int(self.S[0] * gt_y_img)

        # Calculate (tx, ty) for the center of the cell
        # We want sigmoid(tx) to be 0.5 (relative to cell), so tx = 0.0
        tx = torch.tensor(0.0)
        ty = torch.tensor(0.0)

        # Calculate (tw, th) for each anchor box
        # We need to find the best anchor for the ground truth box (1.0, 1.0)
        # For simplicity, and since all B boxes in the cell get the same target,
        # we calculate tw, th for each anchor.
        true_bbox_per_anchor = torch.zeros((self.B, 4))
        for b_idx, (anchor_w, anchor_h) in enumerate(self.anchors):
            # tw = log(gt_w_img / anchor_w)
            # th = log(gt_h_img / anchor_h)
            # Clamp anchors to prevent log(0) or division by zero
            tw = torch.log(gt_w_img / anchor_w.clamp(min=1e-6))
            th = torch.log(gt_h_img / anchor_h.clamp(min=1e-6))
            true_bbox_per_anchor[b_idx, :] = torch.tensor([tx, ty, tw, th])

        # Prepare YOLO target tensor: (S, S, B, 4) for bbox, (S, S, B, 1) for obj, (S, S, B, C) for cls
        true_bbox = torch.zeros((self.S[0], self.S[1], self.B, 4))
        true_obj = torch.zeros((self.S[0], self.S[1], self.B, 1))
        true_cls = torch.zeros((self.S[0], self.S[1], self.B, self.C))
        for b in range(self.B): # Assign the calculated tx,ty,tw,th for each anchor
            true_bbox[cell_y_idx, cell_x_idx, b] = true_bbox_per_anchor[b]
            true_obj[cell_y_idx, cell_x_idx, b] = torch.tensor([1.0]) # Object is present
            true_cls[cell_y_idx, cell_x_idx, b, label] = 1.0 # One-hot encode class
        return image, (true_bbox, true_obj, true_cls)

class ConvBlock(nn.Module):
    """
    Helper function for a Convolutional block: Conv2d -> BatchNorm2d -> LeakyReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))
##this is a simple cnn, need ask lx
class DarknetLikeBackbone(nn.Module):
    """
    A simplified Darknet-like backbone network.
    This aims to be more robust than SimpleCNNBackbone, providing a better feature extractor.
    It will reduce spatial dimensions by a factor of 32 (5 pooling layers).
    Input image size 224x224 -> 7x7 feature map.
    """
    def __init__(self, in_channels=3):
        super(DarknetLikeBackbone, self).__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2), # 112x112

            ConvBlock(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2), # 56x56

            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 64, kernel_size=1, stride=1, padding=0),
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2), # 28x28

            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 128, kernel_size=1, stride=1, padding=0),
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14x14

            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            ConvBlock(512, 256, kernel_size=1, stride=1, padding=0),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            # No pooling here, as the detection head expects a feature map
            # that can be mapped to the grid size.
            # The final spatial dimensions will be 7x7 if input is 224x224
        )

    def forward(self, x):
        return self.features(x)

class SimpleYOLODetectionHead(nn.Module):
    """
    A very simplified YOLO-like detection head.
    It predicts bounding box coordinates (x, y, w, h), objectness score, and class probabilities
    for each cell in a grid.
    """
    def __init__(self, num_classes, anchors, grid_size=(7, 7)):
        super(SimpleYOLODetectionHead, self).__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_bboxes_per_cell = len(anchors) # Number of anchor boxes
        self.anchors = torch.tensor(anchors, dtype=torch.float32) # (B, 2)

        # Output features per bounding box:
        # 4 (bbox coords: x, y, w, h) + 1 (objectness score) + num_classes (class probabilities)
        self.output_dim_per_bbox = 4 + 1 + num_classes
        self.total_output_dim = self.num_bboxes_per_cell * self.output_dim_per_bbox

        # This detection head now takes the output from the backbone.
        # The input channels should match the output channels of the backbone.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, self.total_output_dim, kernel_size=1) # 1x1 conv to get final predictions
        )

    def forward(self, x):
        # x is the feature map from the backbone
        x = self.conv_layers(x) # Output shape: (batch_size, total_output_dim, H_grid, W_grid)

        # Reshape to (batch_size, H_grid, W_grid, num_bboxes_per_cell * output_dim_per_bbox)
        x = x.permute(0, 2, 3, 1) # (batch_size, H_grid, W_grid, total_output_dim)

        # Reshape to separate predictions per cell and bbox
        # (batch_size, H_grid, W_grid, num_bboxes_per_cell, output_dim_per_bbox)
        x = x.view(x.size(0), self.grid_size[0], self.grid_size[1], self.num_bboxes_per_cell, self.output_dim_per_bbox)

        # Separate predictions:
        # bbox_coords: (batch_size, H, W, B, 4) - raw x, y, w, h
        # objectness: (batch_size, H, W, B, 1) - raw objectness score
        # class_probs: (batch_size, H, W, B, num_classes) - raw class scores
        bbox_coords = x[..., 0:4]
        objectness = x[..., 4:5]
        class_probs = x[..., 5:]

        # Apply activation functions
        # For x, y, objectness, and w, h (relative to cell and anchor), sigmoid is typically used.
        # For class probabilities, softmax is used.
        # Note: For w, h, YOLO typically predicts log-space offsets, which are then exponentiated.
        # For simplicity here, we apply sigmoid directly to w,h as well, assuming they are normalized.
        
        # Bounding box coordinates (x, y, w, h)
        # x, y are relative to the grid cell, w, h are relative to anchor boxes
        # Apply sigmoid to x, y to constrain them within the cell [0, 1]
        bbox_x_y = torch.sigmoid(bbox_coords[..., 0:2])
        
        # w, h are predicted as log-space offsets, then exponentiated and multiplied by anchor dimensions
        # Reshape anchors to match the prediction tensor's dimensions for broadcasting
        # anchors: (1, 1, 1, num_bboxes_per_cell, 2)
        anchors_reshaped = self.anchors.view(1, 1, 1, self.num_bboxes_per_cell, 2).to(bbox_coords.device)
        
        # Apply exponential to predicted w, h and multiply by anchor w, h
        # This ensures w, h are positive and scaled by anchors
        bbox_w_h = torch.exp(bbox_coords[..., 2:4]) * anchors_reshaped
        
        bbox_coords = torch.cat((bbox_x_y, bbox_w_h), dim=-1)
        
        # Objectness score - apply sigmoid
        objectness = torch.sigmoid(objectness)
        
        # Class probabilities - apply softmax
        class_probs = F.softmax(class_probs, dim=-1)

        return bbox_coords, objectness, class_probs

# Conceptual YOLO Loss Function (highly simplified)
# A real YOLO loss is much more complex, involving:
# - IoU-based loss for bounding boxes (e.g., CIoU, GIoU)
# - Binary Cross-Entropy for objectness (presence of an object)
# - Cross-Entropy or BCE for class probabilities
# - Handling of "no object" cells and responsible bounding boxes


class YOLOLoss(nn.Module):
    """
    A more realistic YOLO loss function.
    This loss combines:
    - Bounding box regression loss (using IoU)
    - Objectness loss (whether an object is present in a cell)
    - Classification loss (what class the object belongs to)
    """
    def __init__(self, lambda_coord=5, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.bce = nn.BCEWithLogitsLoss(reduction="sum") # For objectness and class
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, target):
        # predictions: (batch_size, S, S, B, 5 + C)
        # target: (batch_size, S, S, B, 5 + C) - where 5 is (x, y, w, h, obj_conf)
        
        # predictions: (pred_bbox_coords_raw, pred_obj_raw, pred_cls_raw)
        # pred_bbox_coords_raw: (batch_size, H, W, B, 4) - raw tx, ty, tw, th
        # pred_obj_raw: (batch_size, H, W, B, 1) - raw objectness score
        # pred_cls_raw: (batch_size, H, W, B, num_classes) - raw class scores
        
        # target: (true_bbox_raw, true_obj, true_cls)
        # true_bbox_raw: (batch_size, H, W, B, 4) - raw tx, ty, tw, th from build_targets
        
        pred_bbox_raw, pred_obj_raw, pred_cls_raw = predictions
        true_bbox_raw, true_obj, true_cls = target

        # Identify cells with objects (responsible for detection)
        # true_obj is (batch_size, H, W, B, 1)
        # We need to flatten it to (batch_size, H*W*B) and then select based on 1s
        # For simplicity, we assume true_obj indicates the presence of an object
        # and that the target bounding box is the one we care about.
        # In a real YOLO, you'd find the best anchor box for each ground truth.
        
        # For now, let's assume true_obj is 1 for cells with objects and 0 otherwise.
        # And we only calculate loss for the "responsible" bounding box.
        # This simplified version assumes the target already specifies the responsible box.
        
        # Reshape predictions and targets for easier indexing
        # (N, S, S, B, 4), (N, S, S, B, 1), (N, S, S, B, C)
        
        # 1. Bounding Box Loss
        # Only calculate for cells that contain an object (where true_obj == 1)
        # We need to ensure the shapes match for IoU calculation.
        
        # Find the best IoU for each predicted box with the ground truth box in that cell
        # This is a simplification. A real YOLO would compare with all ground truths
        # and assign responsibility based on the highest IoU.
        
        # For simplicity, we'll assume true_obj is 1 for the responsible box.
        # We need to mask the predictions and targets.
        
        # Create masks for cells with objects
        obj_mask = true_obj.squeeze(-1).bool() # (N, H, W, B)
        
        # Bounding box coordinates loss (x, y, w, h)
        # Only for cells that contain an object
        # Note: YOLO applies sqrt to w, h for loss, but here we use direct values after sigmoid.
        # A more advanced loss would use CIoU/GIoU.
        
        # For x, y, w, h, we need to ensure they are in the same scale.
        # Since we applied sigmoid, they are [0, 1].
        
        # Calculate IoU for the responsible boxes
        # pred_bbox and true_bbox are (N, H, W, B, 4)
        # We need to flatten them to (N*H*W*B, 4) for IoU calculation, then mask.
        
        # Flatten and apply mask
        # Apply activations to raw predictions for loss calculation where needed
        # For bbox, we need to transform both predicted and true raw values to (x,y,w,h)
        # to calculate IoU and then MSE on (x,y) and (sqrt(w), sqrt(h))
        
        # Get anchors from the detection head (assuming it's passed or accessible)
        # For now, we'll use the global ANCHORS. In a full model, anchors would be passed.
        anchors = torch.tensor(ANCHORS, dtype=torch.float32).to(pred_bbox_raw.device)
        anchors_reshaped = anchors.view(1, 1, 1, len(ANCHORS), 2)

        # Transform predicted raw bbox coords (tx, ty, tw, th) to (x, y, w, h)
        pred_bbox_x_y = torch.sigmoid(pred_bbox_raw[..., 0:2])
        pred_bbox_w_h = torch.exp(pred_bbox_raw[..., 2:4]) * anchors_reshaped
        pred_bbox_transformed = torch.cat((pred_bbox_x_y, pred_bbox_w_h), dim=-1)

        # Transform true raw bbox coords (tx, ty, tw, th) to (x, y, w, h)
        true_bbox_x_y = torch.sigmoid(true_bbox_raw[..., 0:2])
        true_bbox_w_h = torch.exp(true_bbox_raw[..., 2:4]) * anchors_reshaped
        true_bbox_transformed = torch.cat((true_bbox_x_y, true_bbox_w_h), dim=-1)

        # Flatten and apply mask for bbox loss
        pred_bbox_flat = pred_bbox_transformed[obj_mask] # (num_objects, 4)
        true_bbox_flat = true_bbox_transformed[obj_mask] # (num_objects, 4)
        
        # If no objects, bbox_loss is 0
        if pred_bbox_flat.numel() == 0:
            bbox_loss = torch.tensor(0.0, device=predictions[0].device)
        else:
            # Calculate IoU for the responsible boxes
            ious = intersection_over_union(pred_bbox_flat, true_bbox_flat)
            
            # Coordinate loss: MSE on (x, y) and (sqrt(w), sqrt(h))
            # For simplicity, we'll use MSE on (x, y, w, h) directly after sigmoid.
            # A real YOLO would use sqrt(w), sqrt(h) and potentially CIoU/GIoU.
            
            # The original YOLO paper used MSE for (x,y) and (sqrt(w), sqrt(h))
            # Let's try to mimic that for w,h, but keep x,y as is.
            
            # For w, h, we need to ensure they are positive before sqrt.
            # Since sigmoid output is [0,1], this is fine.
            
            # Coordinate loss for x, y
            xy_loss = self.mse(pred_bbox_flat[..., :2], true_bbox_flat[..., :2])
            
            # Coordinate loss for w, h (after sqrt)
            wh_loss = self.mse(torch.sqrt(pred_bbox_flat[..., 2:]), torch.sqrt(true_bbox_flat[..., 2:]))
            
            bbox_loss = self.lambda_coord * (xy_loss + wh_loss)
            
        # 2. Objectness Loss
        # For cells with objects (obj_mask), target is 1.
        # For cells without objects (~obj_mask), target is 0.
        
        # Objectness loss for cells with objects
        # Objectness loss for cells with objects
        obj_loss = self.bce(pred_obj_raw[obj_mask], true_obj[obj_mask])
        
        # Objectness loss for cells without objects (noobj_mask)
        noobj_mask = ~obj_mask # (N, H, W, B)
        noobj_loss = self.lambda_noobj * self.bce(pred_obj_raw[noobj_mask], true_obj[noobj_mask])
        
        objectness_loss = obj_loss + noobj_loss
        
        # 3. Classification Loss
        # Only for cells that contain an object
        # true_cls is (N, H, W, B, C)
        
        # Flatten and apply mask
        pred_cls_flat = pred_cls_raw[obj_mask] # (num_objects, C)
        true_cls_flat = true_cls[obj_mask] # (num_objects, C)
        
        if pred_cls_flat.numel() == 0:
            cls_loss = torch.tensor(0.0, device=predictions[0].device)
        else:
            # Use BCEWithLogitsLoss for multi-label classification (if classes are not mutually exclusive)
            # Or CrossEntropyLoss if classes are mutually exclusive (one-hot encoded target)
            # Assuming one-hot encoded target for simplicity, so BCEWithLogitsLoss is fine.
            cls_loss = self.bce(pred_cls_flat, true_cls_flat)
        
        total_loss = bbox_loss + objectness_loss + cls_loss
        
        return total_loss

if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a unique run directory for saving results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join("runs", "yolo_from_scratch", f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    # List to store results for each epoch
    results_data = []

    # Define checkpoint path
    CHECKPOINT_DIR = os.path.join("runs", "yolo_from_scratch", "checkpoints")
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "yolo_anti_spoofing_model.pth")

    # Hyperparameters
    num_classes = 2 # live, spoof
    num_bboxes_per_cell = 2
    learning_rate = 1e-4
    batch_size = 128 # Increased batch size for better GPU utilization
    num_epochs = 5 # Small number of epochs for demonstration

    # Data transformations
    input_size = (113, 113) # Set to your image size
    transform = transforms.Compose([
        transforms.Resize(input_size), # Resize images to expected input size for backbone
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
    ])

    # Compute grid size dynamically
    grid_size = compute_grid_size(input_size, num_poolings=4)
    print(f"Grid size: {grid_size}")

    # Load datasets
    try:
        train_dataset = AntiSpoofingYOLODataset(root_dir='train', S=grid_size, B=num_bboxes_per_cell, C=num_classes, anchors=ANCHORS, transform=transform)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() // 2 if os.cpu_count() > 1 else 0, pin_memory=True)

        test_dataset = AntiSpoofingYOLODataset(root_dir='test', S=grid_size, B=num_bboxes_per_cell, C=num_classes, anchors=ANCHORS, transform=transform)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() // 2 if os.cpu_count() > 1 else 0, pin_memory=True)

        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")

    except ValueError as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure 'train' and 'test' directories exist and contain 'live' and 'spoof' subfolders with images.")
        exit()
    except IndexError:
        print("Dataset is empty or index out of bounds during loading. Check dataset logic.")
        exit()

    # Initialize the backbone and detection head
    backbone = DarknetLikeBackbone(in_channels=3).to(device)
    yolo_head = SimpleYOLODetectionHead(num_classes, ANCHORS, grid_size).to(device)

    # Combine backbone and head into a single model for training
    class YOLOModel(nn.Module):
        def __init__(self, backbone, head):
            super(YOLOModel, self).__init__()
            self.backbone = backbone
            self.head = head
        
        def forward(self, x):
            features = self.backbone(x)
            # print('Backbone output shape:', features.shape)  # Debug print
            return self.head(features)

    model = YOLOModel(backbone, yolo_head).to(device)

    # Check for existing model checkpoint
    os.makedirs(CHECKPOINT_DIR, exist_ok=True) # Ensure checkpoint directory exists
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading previous model from: {CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print("Previous model loaded successfully.")
    else:
        print("No previous model found. Initializing a new model for training.")

    # Loss function and optimizer
    criterion = YOLOLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    print("\n--- Starting Training ---")
    start_total_time = time.time()
    total_batches_overall = num_epochs * len(train_loader)

    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        total_train_loss = 0.0
        start_epoch_time = time.time()
        
        num_batches = len(train_loader)
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False)
        for batch_idx, (images, targets) in enumerate(train_loader_tqdm):
            images = images.to(device)
            
            # targets is a tuple (true_bbox, true_obj, true_cls)
            # Move each component of the tuple to the device
            targets = tuple(t.to(device) for t in targets)

            # Forward pass
            predictions = model(images) # predictions will be (pred_bbox_raw, pred_obj_raw, pred_cls_raw)
            
            # Calculate loss
            loss = criterion(predictions, targets)
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # Calculate and display progress
            current_time = time.time()
            elapsed_epoch_time = current_time - start_epoch_time
            
            # Calculate estimated time remaining for current epoch
            batches_processed_in_epoch = batch_idx + 1
            if batches_processed_in_epoch > 0:
                time_per_batch_epoch = elapsed_epoch_time / batches_processed_in_epoch
                estimated_epoch_time_remaining = time_per_batch_epoch * (num_batches - batches_processed_in_epoch)
                estimated_epoch_end_time = datetime.now() + timedelta(seconds=estimated_epoch_time_remaining)
            else:
                estimated_epoch_end_time = datetime.now() + timedelta(seconds=0) # No estimate yet

            # Calculate estimated total time remaining
            total_batches_processed = epoch * num_batches + batches_processed_in_epoch
            elapsed_total_time = current_time - start_total_time
            if total_batches_processed > 0:
                time_per_batch_overall = elapsed_total_time / total_batches_processed
                estimated_total_time_remaining = time_per_batch_overall * (total_batches_overall - total_batches_processed)
                estimated_total_end_time = datetime.now() + timedelta(seconds=estimated_total_time_remaining)
            else:
                estimated_total_end_time = datetime.now() + timedelta(seconds=0) # No estimate yet

            train_loader_tqdm.set_postfix(loss=loss.item(),
                                          eta_epoch=estimated_epoch_end_time.strftime('%H:%M:%S'),
                                          eta_total=estimated_total_end_time.strftime('%H:%M:%S'))
        
        avg_train_loss = total_train_loss / num_batches
        end_epoch_time = time.time()
        epoch_duration = end_epoch_time - start_epoch_time
        print(f"\nEpoch [{epoch+1}/{num_epochs}] finished. Avg Training Loss: {avg_train_loss:.4f}, Duration: {epoch_duration:.2f}s")

        # --- Evaluation Loop ---
        model.eval() # Set model to evaluation mode
        total_eval_loss = 0.0
        all_preds = []
        all_targets = []
        
        # Determine the responsible grid cell (center cell) for evaluation
        eval_cell_x_idx = int(grid_size[1] * 0.5)
        eval_cell_y_idx = int(grid_size[0] * 0.5)

        with torch.no_grad():
            test_loader_tqdm = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} Evaluation", leave=False)
            for images, targets in test_loader_tqdm:
                images = images.to(device)
                targets = tuple(t.to(device) for t in targets)

                predictions = model(images)
                loss = criterion(predictions, targets)
                total_eval_loss += loss.item()
                test_loader_tqdm.set_postfix(loss=loss.item())

                # Unpack predictions and targets for metric calculation
                pred_bbox_raw, pred_obj_raw, pred_cls_raw = predictions
                true_bbox_raw, true_obj, true_cls = targets

                # For each image in the batch, get the predicted and true class from the responsible cell
                for i in range(images.size(0)):
                    # Get predictions from the responsible cell (center cell) and first anchor
                    # Assuming the first anchor is representative for this simplified setup
                    predicted_class_scores = pred_cls_raw[i, eval_cell_y_idx, eval_cell_x_idx, 0, :]
                    predicted_class_label = torch.argmax(predicted_class_scores).item()
                    
                    # Get true class label from the responsible cell
                    true_class_label = torch.argmax(true_cls[i, eval_cell_y_idx, eval_cell_x_idx, 0, :]).item()
                    
                    all_preds.append(predicted_class_label)
                    all_targets.append(true_class_label)
            
            avg_eval_loss = total_eval_loss / len(test_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}] Evaluation Loss: {avg_eval_loss:.4f}")

            # Calculate and print classification metrics
            accuracy = accuracy_score(all_targets, all_preds)
            precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

            print(f"Epoch [{epoch+1}/{num_epochs}] Metrics:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")

            # Calculate and print Confusion Matrix
            cm = confusion_matrix(all_targets, all_preds)
            print("\n  Confusion Matrix:")
            print(cm)

            # Store results for saving
            results_data.append({
                'epoch': epoch + 1,
                'avg_train_loss': avg_train_loss,
                'avg_eval_loss': avg_eval_loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm.tolist() # Store as list for CSV compatibility
            })

    end_total_time = time.time()
    total_duration = end_total_time - start_total_time
    print(f"\n--- Training Complete --- Total Duration: {total_duration:.2f}s")

    # --- Generate Final Confusion Matrix ---
    print("\nGenerating Final Confusion Matrix...")
    # Get class names from the dataset
    class_names = list(train_dataset.class_map.keys()) # Assuming train_dataset and test_dataset have same class_map

    # Plot and save Confusion Matrix
    final_cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=final_cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Final Confusion Matrix')
    final_cm_filename = os.path.join(results_dir, 'final_confusion_matrix.png')
    plt.savefig(final_cm_filename)
    plt.close() # Close the plot to free memory
    print(f"Final Confusion matrix saved as {final_cm_filename}")

    # Plot and save Normalized Confusion Matrix
    final_cm_normalized = confusion_matrix(all_targets, all_preds, normalize='true')
    disp_normalized = ConfusionMatrixDisplay(confusion_matrix=final_cm_normalized, display_labels=class_names)
    disp_normalized.plot(cmap=plt.cm.Blues)
    plt.title('Final Normalized Confusion Matrix')
    final_cm_normalized_filename = os.path.join(results_dir, 'final_normalized_confusion_matrix.png')
    plt.savefig(final_cm_normalized_filename)
    plt.close() # Close the plot to free memory
    print(f"Final Normalized confusion matrix saved as {final_cm_normalized_filename}")

    # Save results to CSV
    results_df = pd.DataFrame(results_data)
    results_csv_path = os.path.join(results_dir, "training_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nTraining results saved to: {results_csv_path}")

    # Save the trained model to the checkpoint path
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Trained model saved to: {CHECKPOINT_PATH}")

    print("\n--- Training Complete ---")
    print("This is a conceptual implementation. For real-world use, further refinements are needed:")
    print("- More sophisticated data augmentation.")
    print("- Advanced loss functions (e.g., CIoU/DIoU/GIoU for bbox regression).")
    print("- Proper Non-Maximum Suppression (NMS) implementation for inference.")
    print("- A more robust backbone and neck architecture.")
    print("- Learning rate scheduling and early stopping.")
    print("- Metrics calculation (mAP, precision, recall).")