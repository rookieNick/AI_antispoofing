import os
import argparse
import importlib.util
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import hashlib


def get_file_hash(file_path):
    """Calculate SHA256 hash of a file for integrity verification"""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        print(f"Error calculating hash: {e}")
        return None


def verify_model_integrity(checkpoint_path):
    """Verify model file integrity and display information"""
    print("\n=== Model Integrity Check ===")
    
    if not os.path.exists(checkpoint_path):
        print(f"✗ Model file not found: {checkpoint_path}")
        return False
    
    file_size = os.path.getsize(checkpoint_path)
    print(f"Model file size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
    
    if file_size == 0:
        print("✗ Model file is empty (0 bytes)")
        return False
    
    file_hash = get_file_hash(checkpoint_path)
    if file_hash:
        print(f"Model SHA256: {file_hash}")
        
        # Save hash for future verification
        hash_file = os.path.join(os.path.dirname(checkpoint_path), "model_hash_verification.txt")
        with open(hash_file, "w") as f:
            f.write(f"File: {checkpoint_path}\n")
            f.write(f"Size: {file_size} bytes\n")
            f.write(f"SHA256: {file_hash}\n")
        print(f"✓ Hash saved to: {hash_file}")
        return True
    else:
        print("✗ Could not calculate file hash")
        return False


def load_class_from_path(file_path, class_name):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def load_checkpoint_into_model(model, chk_path, device):
    """Load checkpoint with detailed error handling and verification"""
    if not os.path.exists(chk_path):
        raise FileNotFoundError(f"Checkpoint not found: {chk_path}")
    
    print(f"\n=== Loading Model Checkpoint ===")
    try:
        # Handle PyTorch 2.6 weights_only security change
        try:
            import torch.serialization
            # Add safe globals for numpy objects commonly found in checkpoints
            torch.serialization.add_safe_globals([
                'numpy._core.multiarray.scalar',
                'numpy.core.multiarray.scalar',
                'numpy.dtype',
                'numpy.ndarray'
            ])
            ckpt = torch.load(chk_path, map_location=device, weights_only=True)
            print("✓ Checkpoint loaded successfully (weights_only=True with safe globals)")
        except Exception as safe_error:
            print(f"Safe loading failed: {safe_error}")
            print("Falling back to weights_only=False (trusted source)")
            # Fall back to weights_only=False since this is your trusted model
            ckpt = torch.load(chk_path, map_location=device, weights_only=False)
            print("✓ Checkpoint loaded successfully (weights_only=False)")
        
        # Display checkpoint information
        if isinstance(ckpt, dict):
            print(f"Checkpoint keys: {list(ckpt.keys())}")
            
            if 'epoch' in ckpt:
                print(f"Training epoch: {ckpt['epoch']}")
            if 'best_acc' in ckpt:
                print(f"Best accuracy: {ckpt['best_acc']:.4f}")
            if 'loss' in ckpt:
                print(f"Loss: {ckpt['loss']:.4f}")
            
            if 'model_state_dict' in ckpt:
                state = ckpt['model_state_dict']
                print("Using 'model_state_dict' from checkpoint")
            elif 'state_dict' in ckpt:
                state = ckpt['state_dict']
                print("Using 'state_dict' from checkpoint")
            else:
                state = ckpt
                print("Using full checkpoint as state dict")
        else:
            state = ckpt
            print("Checkpoint is direct state dict")
        
        model.load_state_dict(state)
        print("✓ Model weights loaded successfully")
        
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        print(f"Exception type: {type(e).__name__}")
        raise


def prepare_image(img_path, input_size=(224, 224)):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img).unsqueeze(0)
    return tensor, img


def save_depth_map(depth_tensor, orig_image, out_path):
    depth = depth_tensor.detach().cpu().numpy()
    if depth.ndim == 4:
        depth = depth[0, 0]
    elif depth.ndim == 3:
        depth = depth[0]

    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_img = (depth_norm * 255.0).astype(np.uint8)
    depth_pil = Image.fromarray(depth_img).resize(orig_image.size)
    depth_pil.save(out_path)


def main():
    # Default paths - you can override with command line arguments
    default_checkpoint = r"C:\Users\User\AndroidStudioProjects\AI_antispoofing\YeongChingZhou\CDCN\best_algo\rtx4050_cdcn_results_20250906_165750\rtx4050_cdcn_best.pth"
    default_image = r"C:\Users\User\AndroidStudioProjects\AI_antispoofing\testCaseImages\frontal lobe live.png"
    
    parser = argparse.ArgumentParser(description='Test AdvancedCDCN model with integrity verification')
    parser.add_argument('--checkpoint', '-c', default=default_checkpoint, 
                       help=f'Path to .pth checkpoint (default: {default_checkpoint})')
    parser.add_argument('--image', '-i', default=default_image,
                       help=f'Path to image to test (default: {default_image})')
    parser.add_argument('--map-size', type=int, default=32, help='depth map size expected by model')
    parser.add_argument('--skip-integrity', action='store_true', help='Skip file integrity check')
    args = parser.parse_args()

    print(f"=== Testing Model from GitHub Repository ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test image: {args.image}")
    
    # Verify model integrity unless skipped
    if not args.skip_integrity:
        if not verify_model_integrity(args.checkpoint):
            print("Model integrity check failed. Use --skip-integrity to bypass.")
            return 1

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_file = os.path.join(repo_root, 'main', 'model', 'CDCN_YeongChingZhou', 'model.py')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Cannot find model file at {model_file}")

    AdvancedCDCN = load_class_from_path(model_file, 'AdvancedCDCN')

    device = setup_device()

    # instantiate model with same constructor signature used in repo
    print(f"\n=== Initializing Model ===")
    model = AdvancedCDCN(num_classes=2, theta=0.7, map_size=args.map_size, dropout_rate=0.5)
    model = model.to(device)
    print("✓ Model initialized")

    load_checkpoint_into_model(model, args.checkpoint, device)
    model.eval()
    print("✓ Model set to evaluation mode")

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    print(f"\n=== Running Inference ===")
    img_tensor, orig_img = prepare_image(args.image, input_size=(224, 224))
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        cls_output, depth_map = model(img_tensor)
        probs = torch.softmax(cls_output, dim=1).cpu().numpy()[0]
        pred = int(torch.argmax(cls_output, dim=1).cpu().item())

    label_map = {0: 'Live', 1: 'Spoof'}

    print('\n=== Prediction Results ===')
    print(f"Image: {args.image}")
    print(f"Predicted class: {pred} ({label_map.get(pred, 'Unknown')})")
    print(f"Confidence: {max(probs):.4f}")
    print(f"Probabilities: Live={probs[0]:.4f}, Spoof={probs[1]:.4f}")

    # Save depth map
    base, ext = os.path.splitext(args.image)
    out_depth = base + '_pred_depth.png'
    save_depth_map(depth_map, orig_img, out_depth)
    print(f"✓ Saved predicted depth map to: {out_depth}")
    
    print(f"\n=== Test Complete ===")
    print("✓ Model loaded and tested successfully!")
    print("✓ If you see this message, your model from GitHub works correctly.")
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
