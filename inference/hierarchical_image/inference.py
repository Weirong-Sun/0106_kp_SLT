"""
Inference script for hierarchical keypoint to image reconstruction
Load trained model and generate images from keypoints
"""
import torch
import pickle
import numpy as np
import cv2
import os
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from models.hierarchical_image.model import HierarchicalKeypointToImageTransformer
from utils.utils_image import draw_keypoints_on_canvas

def load_model(checkpoint_path, device='cuda'):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        model: Loaded model
        normalization: Normalization parameters
        model_config: Model configuration
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model_config = checkpoint.get('model_config', {
        'd_global': 256,
        'd_region': 128,
        'num_regions': 8,
        'image_size': 256
    })
    
    model = HierarchicalKeypointToImageTransformer(
        input_dim=3,
        d_global=model_config['d_global'],
        d_region=model_config['d_region'],
        nhead=8,
        num_region_layers=2,
        num_interaction_layers=2,
        dim_feedforward=1024,
        dropout=0.1,
        num_keypoints=68,
        num_regions=model_config['num_regions'],
        image_size=model_config['image_size'],
        num_channels=1
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    normalization = checkpoint.get('normalization', None)
    
    return model, normalization, model_config

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
    # Ensure keypoints are on the correct device first
    if len(keypoints.shape) == 2:
        keypoints = keypoints.unsqueeze(0)  # Add batch dimension
    
    keypoints = keypoints.to(device)
    
    if normalization is not None:
        kp_min = torch.FloatTensor(normalization['kp_min']).to(device)
        kp_max = torch.FloatTensor(normalization['kp_max']).to(device)
        keypoints = (keypoints - kp_min) / (kp_max - kp_min + 1e-8)
    
    with torch.no_grad():
        images = model(keypoints)
    
    return images

def extract_representations(model, keypoints, normalization=None, device='cuda'):
    """
    Extract global and regional representations from keypoints
    
    Args:
        model: Trained model
        keypoints: Keypoints array [68, 3] or [batch_size, 68, 3]
        normalization: Normalization parameters (min, max)
        device: Device to run on
    
    Returns:
        global_repr: Global representation [batch_size, d_global]
        regional_repr: Regional representation [batch_size, num_regions, d_region]
    """
    # Ensure keypoints are on the correct device first
    if len(keypoints.shape) == 2:
        keypoints = keypoints.unsqueeze(0)  # Add batch dimension
    
    keypoints = keypoints.to(device)
    
    if normalization is not None:
        kp_min = torch.FloatTensor(normalization['kp_min']).to(device)
        kp_max = torch.FloatTensor(normalization['kp_max']).to(device)
        keypoints = (keypoints - kp_min) / (kp_max - kp_min + 1e-8)
    
    with torch.no_grad():
        global_repr, regional_repr = model.encode(keypoints)
    
    return global_repr, regional_repr

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints_hierarchical_image/best_model.pth')
    parser.add_argument('--keypoints_data', type=str, default='keypoints_data.pkl')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index to test (single sample)')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to process')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, normalization, model_config = load_model(args.checkpoint, device=args.device)
    print("Model loaded successfully!")
    print(f"Model config: Global={model_config['d_global']}, Regional={model_config['d_region']}, Regions={model_config['num_regions']}, Image Size={model_config['image_size']}")
    
    # Load keypoints data
    print("Loading keypoints data...")
    with open(args.keypoints_data, 'rb') as f:
        data = pickle.load(f)
    
    keypoints = data['keypoints']
    num_total_samples = len(keypoints)
    num_samples_to_process = min(args.num_samples, num_total_samples)
    
    print(f"Total samples available: {num_total_samples}")
    print(f"Processing {num_samples_to_process} samples...")
    
    # Process multiple samples
    output_dir = "visualizations_hierarchical_image"
    os.makedirs(output_dir, exist_ok=True)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    all_comparisons = []
    all_mses = []
    
    for idx in range(num_samples_to_process):
        print(f"\n{'='*60}")
        print(f"Processing sample {idx+1}/{num_samples_to_process}")
        print(f"{'='*60}")
        
        sample_kp = torch.FloatTensor(keypoints[idx])
        
        # Extract representations
        global_repr, regional_repr = extract_representations(model, sample_kp, normalization, args.device)
        
        if idx == 0:  # Print representation info only for first sample
            print(f"Global representation shape: {global_repr.shape}")
            print(f"Regional representation shape: {regional_repr.shape}")
        
        # Generate image
        generated_images = generate_image(model, sample_kp, normalization, args.device)
        generated_image = generated_images[0, 0].cpu().numpy()  # Remove batch and channel dimensions
        
        # Convert from [-1, 1] to [0, 255]
        generated_image = (generated_image + 1) / 2 * 255
        generated_image = generated_image.astype(np.uint8)
        
        # Generate ground truth for comparison
        gt_image = draw_keypoints_on_canvas(sample_kp.numpy(), image_size=model_config['image_size'], draw_lines=True)
        
        # Compute reconstruction error
        mse = np.mean((gt_image.astype(float) - generated_image.astype(float)) ** 2)
        all_mses.append(mse)
        print(f"Reconstruction MSE: {mse:.6f}")
        
        # Save individual images
        generated_path = os.path.join(output_dir, f"generated_image_{idx}.png")
        gt_path = os.path.join(output_dir, f"ground_truth_{idx}.png")
        comparison_path = os.path.join(output_dir, f"comparison_{idx}.png")
        
        cv2.imwrite(generated_path, generated_image)
        cv2.imwrite(gt_path, gt_image)
        
        # Create side-by-side comparison
        comparison_image = np.hstack([gt_image, generated_image])
        
        # Add labels
        cv2.putText(comparison_image, f'Sample {idx+1} - GT', (10, 30), font, 1, 0, 2)
        cv2.putText(comparison_image, f'Sample {idx+1} - Gen', (model_config['image_size'] + 10, 30), font, 1, 0, 2)
        cv2.putText(comparison_image, f'MSE: {mse:.6f}', (model_config['image_size'] + 10, model_config['image_size'] - 20), font, 0.7, 0, 2)
        
        cv2.imwrite(comparison_path, comparison_image)
        all_comparisons.append(comparison_image)
        
        print(f"Saved: {comparison_path}")
        
        # Try to load and display original image if available
        if 'image_paths' in data and idx < len(data['image_paths']):
            original_img_path = data['image_paths'][idx]
            if os.path.exists(original_img_path):
                original_photo = cv2.imread(original_img_path)
                if original_photo is not None:
                    # Resize original photo to match keypoint image size
                    h, w = original_photo.shape[:2]
                    scale = model_config['image_size'] / max(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    original_photo_resized = cv2.resize(original_photo, (new_w, new_h))
                    
                    # Create a canvas and center the image
                    photo_canvas = np.ones((model_config['image_size'], model_config['image_size'], 3), dtype=np.uint8) * 255
                    y_offset = (model_config['image_size'] - new_h) // 2
                    x_offset = (model_config['image_size'] - new_w) // 2
                    photo_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = original_photo_resized
                    
                    # Convert to grayscale for consistency
                    photo_canvas_gray = cv2.cvtColor(photo_canvas, cv2.COLOR_BGR2GRAY)
                    
                    # Create full comparison: original photo | ground truth | generated
                    full_comparison = np.hstack([
                        photo_canvas_gray,
                        gt_image,
                        generated_image
                    ])
                    
                    # Add labels
                    cv2.putText(full_comparison, f'Sample {idx+1} - Photo', (10, 30), font, 1, 0, 2)
                    cv2.putText(full_comparison, f'Sample {idx+1} - GT KP', (model_config['image_size'] + 10, 30), font, 1, 0, 2)
                    cv2.putText(full_comparison, f'Sample {idx+1} - Gen', (model_config['image_size'] * 2 + 10, 30), font, 1, 0, 2)
                    cv2.putText(full_comparison, f'MSE: {mse:.6f}', (model_config['image_size'] * 2 + 10, model_config['image_size'] - 20), font, 0.7, 0, 2)
                    
                    full_comparison_path = os.path.join(output_dir, f"full_comparison_{idx}.png")
                    cv2.imwrite(full_comparison_path, full_comparison)
                    print(f"Saved full comparison: {full_comparison_path}")
    
    # Create grid of all comparisons
    print(f"\n{'='*60}")
    print("Creating summary grid...")
    print(f"{'='*60}")
    
    if len(all_comparisons) > 0:
        # Arrange comparisons in a grid
        cols = min(2, len(all_comparisons))  # 2 columns
        rows = (len(all_comparisons) + cols - 1) // cols
        
        grid_height = rows * model_config['image_size']
        grid_width = cols * model_config['image_size'] * 2  # Each comparison has 2 images side by side
        
        grid_image = np.ones((grid_height, grid_width), dtype=np.uint8) * 255
        
        for i, comp_img in enumerate(all_comparisons):
            row = i // cols
            col = i % cols
            y_start = row * model_config['image_size']
            x_start = col * model_config['image_size'] * 2
            grid_image[y_start:y_start+model_config['image_size'], x_start:x_start+comp_img.shape[1]] = comp_img
        
        grid_path = os.path.join(output_dir, f"grid_comparison_{num_samples_to_process}samples.png")
        cv2.imwrite(grid_path, grid_image)
        print(f"Saved grid comparison: {grid_path}")
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"  Number of samples: {num_samples_to_process}")
        print(f"  Average MSE: {np.mean(all_mses):.6f}")
        print(f"  Min MSE: {np.min(all_mses):.6f}")
        print(f"  Max MSE: {np.max(all_mses):.6f}")
        print(f"  Std MSE: {np.std(all_mses):.6f}")
    
    print("\nDone!")

