# ==============================================================================
# Debug and Fix Model Loading Script
# ==============================================================================
# This script will help identify what's in your saved model file and load it correctly
# ==============================================================================

import torch
import numpy as np
import os

# Apply PyTorch 2.6 fix
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

def inspect_model_file(model_path):
    """Inspect what's inside the model file to understand its structure."""
    print(f"Inspecting model file: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        return None
    
    try:
        # Try loading with weights_only=False to see the structure
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict):
            print("Model file contains a dictionary with keys:")
            for key in checkpoint.keys():
                print(f"  - {key}: {type(checkpoint[key])}")
            return checkpoint
        else:
            print(f"Model file contains: {type(checkpoint)}")
            return checkpoint
            
    except Exception as e:
        print(f"Error loading model file: {e}")
        return None

def find_available_models():
    """Find all available model files in common directories."""
    search_dirs = ['vit_models', '.', 'models', 'checkpoints']
    model_files = []
    
    for dir_name in search_dirs:
        if os.path.exists(dir_name):
            for file in os.listdir(dir_name):
                if file.endswith('.pth'):
                    full_path = os.path.join(dir_name, file)
                    model_files.append(full_path)
    
    return model_files

def load_model_correctly():
    """Load the model with the correct method based on file inspection."""
    
    # First, find available model files
    print("Searching for model files...")
    model_files = find_available_models()
    
    if not model_files:
        print("No .pth files found in common directories")
        return None, None
    
    print(f"Found {len(model_files)} model file(s):")
    for i, file in enumerate(model_files):
        file_size = os.path.getsize(file) / (1024*1024)  # MB
        print(f"  {i+1}. {file} ({file_size:.1f} MB)")
    
    # Use the first model file or let user choose
    if len(model_files) == 1:
        chosen_model = model_files[0]
        print(f"\nUsing: {chosen_model}")
    else:
        print(f"\nUsing the first model file: {model_files[0]}")
        chosen_model = model_files[0]
    
    # Inspect the model file
    checkpoint_data = inspect_model_file(chosen_model)
    if checkpoint_data is None:
        return None, None
    
    # Import required modules
    from transformers import ViTForImageClassification
    
    # Initialize the model
    print("\nInitializing ViT model...")
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=2,
        ignore_mismatched_sizes=True,
        cache_dir='./vit_cache'
    )
    
    # Load the weights based on what we found
    try:
        if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
            print("Loading from checkpoint format (with model_state_dict key)")
            model.load_state_dict(checkpoint_data['model_state_dict'])
            
            # Show additional checkpoint info
            if 'epoch' in checkpoint_data:
                print(f"Checkpoint from epoch: {checkpoint_data['epoch']}")
            if 'best_val_acc' in checkpoint_data:
                print(f"Best validation accuracy: {checkpoint_data['best_val_acc']:.2f}%")
                
        elif isinstance(checkpoint_data, dict):
            print("Loading state dict directly (dictionary format)")
            model.load_state_dict(checkpoint_data)
        else:
            print("Unknown model format - attempting direct load")
            model.load_state_dict(checkpoint_data)
        
        print("Model loaded successfully!")
        return model, chosen_model
        
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("\nTrying alternative loading methods...")
        
        # Try loading with strict=False (allows missing/unexpected keys)
        try:
            if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
                missing_keys, unexpected_keys = model.load_state_dict(
                    checkpoint_data['model_state_dict'], strict=False
                )
            else:
                missing_keys, unexpected_keys = model.load_state_dict(
                    checkpoint_data, strict=False
                )
            
            print(f"Loaded with strict=False")
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
            
            return model, chosen_model
            
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
            return None, None

def run_quick_test(model, model_path):
    """Run a quick test to verify the model works."""
    from torchvision import transforms
    from PIL import Image
    import torch.nn.functional as F
    
    print("\nRunning quick model test...")
    
    # Create test transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create a dummy image
    dummy_image = Image.new('RGB', (224, 224), color='red')
    input_tensor = transform(dummy_image).unsqueeze(0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    input_tensor = input_tensor.to(device)
    
    try:
        with torch.no_grad():
            outputs = model(input_tensor)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)
            
        print(f"Model test successful!")
        print(f"Output shape: {logits.shape}")
        print(f"Probabilities: {probabilities[0].cpu().numpy()}")
        
        # The model is working - now we can run full testing
        return True
        
    except Exception as e:
        print(f"Model test failed: {e}")
        return False

def run_full_testing(model, model_path):
    """Run the full testing pipeline."""
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Dataset class (simplified version)
    class CASIAFASDDataset:
        def __init__(self, data_dir, transform=None):
            self.data_dir = data_dir
            self.transform = transform
            self.images = []
            self.labels = []
            
            if not os.path.exists(data_dir):
                raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
            
            # Load images
            for class_name, label in [('live', 1), ('spoof', 0)]:
                class_dir = os.path.join(data_dir, class_name)
                if os.path.exists(class_dir):
                    files = [f for f in os.listdir(class_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    for f in files:
                        self.images.append(os.path.join(class_dir, f))
                        self.labels.append(label)
            
            print(f"Loaded {len(self.images)} test images")
            print(f"Live: {sum(self.labels)}, Spoof: {len(self.labels) - sum(self.labels)}")
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            from PIL import Image
            img_path = self.images[idx]
            label = self.labels[idx]
            
            try:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, label
            except:
                # Fallback to dummy image
                if self.transform:
                    return self.transform(Image.new('RGB', (224, 224))), label
                return Image.new('RGB', (224, 224)), label
    
    # Configuration
    TEST_DATA_DIRS = [
        "CASIA-FASD/test",
        "dataset/casia-fasd/test", 
        "test",
        "../dataset/casia-fasd/test"
    ]
    
    # Find test data directory
    test_dir = None
    for dir_path in TEST_DATA_DIRS:
        if os.path.exists(dir_path):
            test_dir = dir_path
            break
    
    if not test_dir:
        print("Test data directory not found. Please specify the correct path:")
        for dir_path in TEST_DATA_DIRS:
            print(f"  Tried: {dir_path}")
        return
    
    print(f"Using test directory: {test_dir}")
    
    # Setup data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        dataset = CASIAFASDDataset(test_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return
    
    # Run testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probs = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for data, targets in tqdm(dataloader):
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            _, predicted = logits.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Calculate MSE and RMSE
    probs_array = np.array(all_probs)
    live_probs = probs_array[:, 1]
    targets_array = np.array(all_targets).astype(float)
    mse = mean_squared_error(targets_array, live_probs)
    rmse = np.sqrt(mse)
    
    # Display results
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    
    # Classification report
    class_names = ['spoof', 'live']
    report = classification_report(all_targets, all_predictions, target_names=class_names)
    print(f"\nDetailed Classification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Spoof', 'Live'], yticklabels=['Spoof', 'Live'])
    plt.title(f'Test Results - Accuracy: {accuracy*100:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('final_test_results.png', dpi=300)
    plt.show()
    
    print(f"\nResults saved to: final_test_results.png")
    print("Testing completed successfully!")

def main():
    """Main function to debug and run testing."""
    print("="*60)
    print("MODEL LOADING DEBUG AND TESTING")
    print("="*60)
    
    # Step 1: Load the model correctly
    model, model_path = load_model_correctly()
    
    if model is None:
        print("Failed to load model. Please check the error messages above.")
        return
    
    # Step 2: Quick test
    if not run_quick_test(model, model_path):
        print("Model quick test failed.")
        return
    
    # Step 3: Full testing
    print("\nProceeeding to full testing...")
    run_full_testing(model, model_path)

if __name__ == "__main__":
    main()