"""
Inference script for skeleton reconstruction
Load trained model and generate skeleton images from keypoints
"""
import torch
import pickle
import numpy as np
import cv2
import os
from model_skeleton import SkeletonReconstructionTransformer
from utils_skeleton import draw_full_skeleton

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
        'd_model': 512,
        'num_keypoints': 143,
        'image_size': 256
    })
    
    model = SkeletonReconstructionTransformer(
        input_dim=3,
        d_model=model_config['d_model'],
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        num_keypoints=model_config['num_keypoints'],
        image_size=model_config['image_size'],
        num_channels=1
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    normalization = checkpoint.get('normalization', None)
    
    return model, normalization, model_config

def prepare_keypoints(kp_dict, normalization=None, device='cuda'):
    """
    Prepare keypoints for model input
    
    Args:
        kp_dict: Dictionary with 'face', 'left_hand', 'right_hand', 'pose' keypoints
        normalization: Normalization parameters
        device: Device to run on
    
    Returns:
        keypoints: Prepared keypoints tensor [1, 143, 3]
    """
    # Flatten keypoints
    kp_list = []
    
    # Face (68 points)
    if kp_dict.get('face') is not None:
        kp_list.append(kp_dict['face'])
    else:
        kp_list.append(np.zeros((68, 3), dtype=np.float32))
    
    # Left hand (21 points)
    if kp_dict.get('left_hand') is not None:
        kp_list.append(kp_dict['left_hand'])
    else:
        kp_list.append(np.zeros((21, 3), dtype=np.float32))
    
    # Right hand (21 points)
    if kp_dict.get('right_hand') is not None:
        kp_list.append(kp_dict['right_hand'])
    else:
        kp_list.append(np.zeros((21, 3), dtype=np.float32))
    
    # Pose (33 points)
    if kp_dict.get('pose') is not None:
        kp_list.append(kp_dict['pose'])
    else:
        kp_list.append(np.zeros((33, 3), dtype=np.float32))
    
    # Concatenate
    flattened = np.concatenate(kp_list, axis=0)  # [143, 3]
    keypoints = torch.FloatTensor(flattened).unsqueeze(0).to(device)  # [1, 143, 3]
    
    # Normalize
    if normalization is not None:
        kp_min = torch.FloatTensor(normalization['kp_min']).to(device)
        kp_max = torch.FloatTensor(normalization['kp_max']).to(device)
        kp_range = kp_max - kp_min
        kp_range[kp_range < 1e-8] = 1.0
        keypoints = (keypoints - kp_min) / kp_range
    
    return keypoints

def generate_skeleton(model, kp_dict, normalization=None, device='cuda'):
    """
    Generate skeleton image from keypoints
    
    Args:
        model: Trained model
        kp_dict: Dictionary with keypoints
        normalization: Normalization parameters
        device: Device to run on
    
    Returns:
        image: Generated skeleton image [H, W]
    """
    keypoints = prepare_keypoints(kp_dict, normalization, device)
    
    with torch.no_grad():
        generated_images = model(keypoints)
        generated_image = generated_images[0, 0].cpu().numpy()  # Remove batch and channel dimensions
    
    # Convert from [-1, 1] to [0, 255]
    generated_image = (generated_image + 1) / 2 * 255
    generated_image = generated_image.astype(np.uint8)
    
    return generated_image

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints_skeleton/best_model.pth')
    parser.add_argument('--keypoints_data', type=str, default='sign_language_keypoints.pkl',
                        help='Path to body keypoints pickle file')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index to test')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to process')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, normalization, model_config = load_model(args.checkpoint, device=args.device)
    print("Model loaded successfully!")
    print(f"Model config: d_model={model_config['d_model']}, num_keypoints={model_config['num_keypoints']}, image_size={model_config['image_size']}")
    
    # Load keypoints data
    print("Loading keypoints data...")
    with open(args.keypoints_data, 'rb') as f:
        data = pickle.load(f)
    
    keypoints_list = data['keypoints']
    num_total_samples = len(keypoints_list)
    num_samples_to_process = min(args.num_samples, num_total_samples)
    
    print(f"Total samples available: {num_total_samples}")
    print(f"Processing {num_samples_to_process} samples...")
    
    # Process multiple samples
    output_dir = "visualizations_skeleton"
    os.makedirs(output_dir, exist_ok=True)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    all_comparisons = []
    all_mses = []
    
    for idx in range(num_samples_to_process):
        print(f"\n{'='*60}")
        print(f"Processing sample {idx+1}/{num_samples_to_process}")
        print(f"{'='*60}")
        
        kp_dict = keypoints_list[idx]
        
        # Generate skeleton image
        generated_image = generate_skeleton(model, kp_dict, normalization, args.device)
        
        # Generate ground truth for comparison
        gt_image = draw_full_skeleton(kp_dict, image_size=model_config['image_size'])
        
        # Compute reconstruction error
        mse = np.mean((gt_image.astype(float) - generated_image.astype(float)) ** 2)
        all_mses.append(mse)
        print(f"Reconstruction MSE: {mse:.6f}")
        
        # Save individual images
        generated_path = os.path.join(output_dir, f"generated_skeleton_{idx}.png")
        gt_path = os.path.join(output_dir, f"ground_truth_skeleton_{idx}.png")
        comparison_path = os.path.join(output_dir, f"comparison_skeleton_{idx}.png")
        
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

