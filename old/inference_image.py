"""
Inference script for keypoint to image reconstruction
Load trained model and generate images from keypoints
"""
import torch
import pickle
import numpy as np
import cv2
from model_image import KeypointToImageTransformer
from utils_image import draw_keypoints_on_canvas

def load_model(checkpoint_path, image_size=256, device='cuda'):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        image_size: Size of output images
        device: Device to load model on
    
    Returns:
        model: Loaded model
        normalization: Normalization parameters
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = KeypointToImageTransformer(
        input_dim=3,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        num_keypoints=68,
        image_size=image_size,
        num_channels=1
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    normalization = checkpoint.get('normalization', None)
    
    return model, normalization

def generate_image(model, keypoints, normalization=None, device='cuda'):
    """
    Generate image from keypoints
    
    Args:
        model: Trained model
        keypoints: Keypoints array [68, 3] or [batch_size, 68, 3]
        normalization: Normalization parameters (min, max)
        device: Device to run on
    
    Returns:
        images: Generated images [batch_size, 1, H, W] or [1, H, W]
    """
    if normalization is not None:
        kp_min = torch.FloatTensor(normalization['kp_min']).to(device)
        kp_max = torch.FloatTensor(normalization['kp_max']).to(device)
        keypoints = (keypoints - kp_min) / (kp_max - kp_min + 1e-8)
    
    if len(keypoints.shape) == 2:
        keypoints = keypoints.unsqueeze(0)  # Add batch dimension
    
    keypoints = keypoints.to(device)
    
    with torch.no_grad():
        images = model(keypoints)
    
    return images

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints_image/best_model.pth')
    parser.add_argument('--keypoints_data', type=str, default='keypoints_data.pkl')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index to test')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--output_path', type=str, default='generated_image.png')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, normalization = load_model(args.checkpoint, image_size=args.image_size, device=args.device)
    print("Model loaded successfully!")
    
    # Load keypoints data
    print("Loading keypoints data...")
    with open(args.keypoints_data, 'rb') as f:
        data = pickle.load(f)
    
    keypoints = data['keypoints']
    sample_kp = torch.FloatTensor(keypoints[args.sample_idx])
    
    print(f"Original keypoints shape: {sample_kp.shape}")
    
    # Generate image
    print("Generating image from keypoints...")
    generated_images = generate_image(model, sample_kp, normalization, args.device)
    generated_image = generated_images[0, 0].cpu().numpy()  # Remove batch and channel dimensions
    
    # Convert from [-1, 1] to [0, 255]
    generated_image = (generated_image + 1) / 2 * 255
    generated_image = generated_image.astype(np.uint8)
    
    # Generate ground truth for comparison
    gt_image = draw_keypoints_on_canvas(sample_kp.numpy(), image_size=args.image_size)
    
    # Compute reconstruction error
    mse = np.mean((gt_image.astype(float) - generated_image.astype(float)) ** 2)
    print(f"Reconstruction MSE: {mse:.6f}")
    
    # Save generated image
    cv2.imwrite(args.output_path, generated_image)
    print(f"Saved generated image to {args.output_path}")
    
    # Save ground truth for comparison
    gt_path = args.output_path.replace('.png', '_gt.png')
    cv2.imwrite(gt_path, gt_image)
    print(f"Saved ground truth image to {gt_path}")
    
    print("\nDone!")

