import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Import model and helper from local module
from cdcn_ver1 import AdvancedCDCN, setup_device


# ------- User-configurable: path to the image to test -------
# By default this points to a sample image included in the workspace.
IMAGE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'dataset', 'casia-fasd', 'test', 'live', 's10v1f0.png'
))

# If you want to test a different image, change IMAGE_PATH above to an absolute path.


def load_checkpoint(model, chk_path, device):
    if not os.path.exists(chk_path):
        raise FileNotFoundError(f"Checkpoint not found: {chk_path}")
    ckpt = torch.load(chk_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
    model.load_state_dict(state)


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
    # depth_tensor expected shape: (1, 1, H, W) or (1, H, W)
    depth = depth_tensor.detach().cpu().numpy()
    if depth.ndim == 4:
        depth = depth[0, 0]
    elif depth.ndim == 3:
        depth = depth[0]

    # Normalize to 0-255 and resize to original image size
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_img = (depth_norm * 255.0).astype(np.uint8)
    depth_pil = Image.fromarray(depth_img).resize(orig_image.size)
    depth_pil.save(out_path)


def main():
    device = setup_device()

    # Build model (must match architecture used during training)
    model = AdvancedCDCN(num_classes=2, theta=0.7, map_size=32, dropout_rate=0.5)
    model = model.to(device)

    # Try to find checkpoint in the same folder as this script
    chk_candidates = [
        os.path.join(os.path.dirname(__file__), 'advanced_cdcn_best.pth'),
        os.path.join(os.path.dirname(__file__), '..', 'advanced_cdcn_best.pth'),
        os.path.join(os.path.dirname(__file__), '..', 'GohWenKang', 'best_vit_model.pth')
    ]
    chk_path = None
    for c in chk_candidates:
        if os.path.exists(c):
            chk_path = c
            break

    if chk_path is None:
        print("No checkpoint found. Please place 'advanced_cdcn_best.pth' next to this script or adjust chk_candidates.")
        return

    print(f"Loading checkpoint: {chk_path}")
    load_checkpoint(model, chk_path, device)

    model.eval()

    if not os.path.exists(IMAGE_PATH):
        print(f"Test image not found: {IMAGE_PATH}")
        return

    img_tensor, orig_img = prepare_image(IMAGE_PATH, input_size=(224, 224))
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        cls_output, depth_map = model(img_tensor)
        probs = torch.softmax(cls_output, dim=1).cpu().numpy()[0]
        pred = int(torch.argmax(cls_output, dim=1).cpu().item())

    label_map = {0: 'Live', 1: 'Spoof'}

    print('\n=== Prediction result ===')
    print(f"Image: {IMAGE_PATH}")
    print(f"Predicted class: {pred} ({label_map.get(pred, 'Unknown')})")
    print(f"Probabilities: Live={probs[0]:.4f}, Spoof={probs[1]:.4f}")

    # Save depth map next to original image
    base, ext = os.path.splitext(IMAGE_PATH)
    out_depth = base + '_pred_depth.png'
    save_depth_map(depth_map, orig_img, out_depth)
    print(f"Saved predicted depth map to: {out_depth}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error during prediction: {e}")
