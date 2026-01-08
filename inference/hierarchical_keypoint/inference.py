"""
Inference script for hierarchical keypoint representation learning
Extract global and regional representations
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
from models.hierarchical_keypoint.model import HierarchicalKeypointTransformer
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
        'num_regions': 8
    })
    
    model = HierarchicalKeypointTransformer(
        input_dim=3,
        d_global=model_config['d_global'],
        d_region=model_config['d_region'],
        nhead=8,
        num_region_layers=2,
        num_interaction_layers=2,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        num_keypoints=68,
        num_regions=model_config['num_regions']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    normalization = checkpoint.get('normalization', None)
    
    return model, normalization, model_config

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

def reconstruct_keypoints(model, global_repr, regional_repr, normalization=None, device='cuda'):
    """
    Reconstruct keypoints from representations
    
    Note: This is a simplified version. Full reconstruction requires the full forward pass.
    """
    # For full reconstruction, we need to use model.forward()
    # This function is mainly for demonstration
    pass

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints_hierarchical/best_model.pth')
    parser.add_argument('--keypoints_data', type=str, default='keypoints_data.pkl')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index to test')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--image_size', type=int, default=512, help='Size of visualization images')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, normalization, model_config = load_model(args.checkpoint, device=args.device)
    print("Model loaded successfully!")
    print(f"Model config: Global={model_config['d_global']}, Regional={model_config['d_region']}, Regions={model_config['num_regions']}")
    
    # Load keypoints data
    print("Loading keypoints data...")
    with open(args.keypoints_data, 'rb') as f:
        data = pickle.load(f)
    
    keypoints = data['keypoints']
    sample_kp = torch.FloatTensor(keypoints[args.sample_idx])
    
    print(f"Original keypoints shape: {sample_kp.shape}")
    
    # Extract representations
    print("Extracting representations...")
    global_repr, regional_repr = extract_representations(model, sample_kp, normalization, args.device)
    
    print(f"Global representation shape: {global_repr.shape}")
    print(f"Regional representation shape: {regional_repr.shape}")
    print(f"Total representation size: {global_repr.shape[1] + regional_repr.shape[1] * regional_repr.shape[2]} dimensions")
    
    # Print detailed representation statistics
    print("\n" + "="*60)
    print("REPRESENTATION STATISTICS")
    print("="*60)
    
    global_repr_np = global_repr[0].cpu().numpy()
    regional_repr_np = regional_repr[0].cpu().numpy()
    
    print(f"\nGlobal Representation:")
    print(f"  Shape: {global_repr_np.shape}")
    print(f"  Mean: {global_repr_np.mean():.6f}")
    print(f"  Std:  {global_repr_np.std():.6f}")
    print(f"  Min:  {global_repr_np.min():.6f}")
    print(f"  Max:  {global_repr_np.max():.6f}")
    print(f"  Non-zero elements: {np.count_nonzero(global_repr_np)} / {len(global_repr_np)}")
    
    print(f"\nRegional Representation (8 regions):")
    region_names = [
        "Face outline (0-16)",
        "Right eyebrow (17-21)",
        "Left eyebrow (22-26)",
        "Nose (27-35)",
        "Right eye (36-41)",
        "Left eye (42-47)",
        "Mouth outer (48-59)",
        "Mouth inner (60-67)"
    ]
    
    for i, name in enumerate(region_names):
        region_repr = regional_repr_np[i]
        print(f"  Region {i} - {name}:")
        print(f"    Mean: {region_repr.mean():.6f}, Std: {region_repr.std():.6f}")
        print(f"    Min: {region_repr.min():.6f}, Max: {region_repr.max():.6f}")
    
    print(f"\nOverall Regional Stats:")
    print(f"  Mean: {regional_repr_np.mean():.6f}, Std: {regional_repr_np.std():.6f}")
    print(f"  Min: {regional_repr_np.min():.6f}, Max: {regional_repr_np.max():.6f}")
    
    # Reconstruct keypoints
    print("\n" + "="*60)
    print("RECONSTRUCTION ANALYSIS")
    print("="*60)
    
    print("\nReconstructing keypoints...")
    sample_kp_batch = sample_kp.unsqueeze(0).to(args.device)
    if normalization is not None:
        kp_min = torch.FloatTensor(normalization['kp_min']).to(args.device)
        kp_max = torch.FloatTensor(normalization['kp_max']).to(args.device)
        sample_kp_normalized = (sample_kp_batch - kp_min) / (kp_max - kp_min + 1e-8)
    else:
        sample_kp_normalized = sample_kp_batch
    
    with torch.no_grad():
        reconstructed = model(sample_kp_normalized)
        reconstructed = reconstructed[0].cpu().numpy()  # Remove batch dimension
    
    # Compute reconstruction error
    original_normalized = sample_kp.numpy()
    if normalization is not None:
        kp_min_np = normalization['kp_min']
        kp_max_np = normalization['kp_max']
        original_normalized = (original_normalized - kp_min_np) / (kp_max_np - kp_min_np + 1e-8)
    
    # Overall error
    mse = np.mean((original_normalized - reconstructed) ** 2)
    mae = np.mean(np.abs(original_normalized - reconstructed))
    rmse = np.sqrt(mse)
    
    print(f"\nOverall Reconstruction Error:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    
    # Per-coordinate error
    error_per_coord = np.mean((original_normalized - reconstructed) ** 2, axis=0)
    print(f"\nPer-coordinate Error (X, Y, Z):")
    print(f"  X: {error_per_coord[0]:.6f}")
    print(f"  Y: {error_per_coord[1]:.6f}")
    print(f"  Z: {error_per_coord[2]:.6f}")
    
    # Per-region error
    print(f"\nPer-region Reconstruction Error:")
    for i, (name, indices) in enumerate(zip(region_names, model.region_indices)):
        region_original = original_normalized[indices]
        region_reconstructed = reconstructed[indices]
        region_mse = np.mean((region_original - region_reconstructed) ** 2)
        region_mae = np.mean(np.abs(region_original - region_reconstructed))
        print(f"  Region {i} - {name}:")
        print(f"    MSE: {region_mse:.6f}, MAE: {region_mae:.6f}, Points: {len(indices)}")
    
    # Per-keypoint error (top 5 worst and best)
    per_kp_error = np.mean((original_normalized - reconstructed) ** 2, axis=1)
    worst_indices = np.argsort(per_kp_error)[-5:][::-1]
    best_indices = np.argsort(per_kp_error)[:5]
    
    print(f"\nTop 5 Worst Reconstructed Keypoints:")
    for idx in worst_indices:
        print(f"  KP {idx:2d}: Error = {per_kp_error[idx]:.6f}")
        print(f"    Original:     [{original_normalized[idx, 0]:.4f}, {original_normalized[idx, 1]:.4f}, {original_normalized[idx, 2]:.4f}]")
        print(f"    Reconstructed: [{reconstructed[idx, 0]:.4f}, {reconstructed[idx, 1]:.4f}, {reconstructed[idx, 2]:.4f}]")
    
    print(f"\nTop 5 Best Reconstructed Keypoints:")
    for idx in best_indices:
        print(f"  KP {idx:2d}: Error = {per_kp_error[idx]:.6f}")
    
    # Error distribution
    print(f"\nError Distribution:")
    print(f"  Min error:  {per_kp_error.min():.6f}")
    print(f"  Max error:  {per_kp_error.max():.6f}")
    print(f"  Mean error: {per_kp_error.mean():.6f}")
    print(f"  Median error: {np.median(per_kp_error):.6f}")
    print(f"  Std error:  {per_kp_error.std():.6f}")
    
    # Sample comparison
    print(f"\n" + "="*60)
    print("SAMPLE COMPARISON (First 10 keypoints)")
    print("="*60)
    print(f"{'KP':<4} {'Original (X,Y,Z)':<30} {'Reconstructed (X,Y,Z)':<30} {'Error':<10}")
    print("-" * 80)
    for i in range(min(10, len(original_normalized))):
        orig_str = f"[{original_normalized[i,0]:.4f},{original_normalized[i,1]:.4f},{original_normalized[i,2]:.4f}]"
        recon_str = f"[{reconstructed[i,0]:.4f},{reconstructed[i,1]:.4f},{reconstructed[i,2]:.4f}]"
        error_str = f"{per_kp_error[i]:.6f}"
        print(f"{i:<4} {orig_str:<30} {recon_str:<30} {error_str:<10}")
    
    # Visualize keypoints
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    
    # Denormalize for visualization (keypoints are already in [0,1] range from MediaPipe)
    # We need to use the original keypoints directly since they're already normalized
    original_denorm = sample_kp.numpy()
    
    # Denormalize reconstructed keypoints
    if normalization is not None:
        kp_min_np = normalization['kp_min']
        kp_max_np = normalization['kp_max']
        reconstructed_denorm = reconstructed * (kp_max_np - kp_min_np + 1e-8) + kp_min_np
    else:
        reconstructed_denorm = reconstructed
    
    # Draw original keypoints
    print("\nGenerating visualization images...")
    image_size = args.image_size
    original_image = draw_keypoints_on_canvas(original_denorm, image_size=image_size, point_radius=3, line_thickness=2, draw_lines=True)
    reconstructed_image = draw_keypoints_on_canvas(reconstructed_denorm, image_size=image_size, point_radius=3, line_thickness=2, draw_lines=False)
    
    # Create side-by-side comparison
    comparison_image = np.hstack([original_image, reconstructed_image])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison_image, 'Original', (10, 30), font, 1, 0, 2)
    cv2.putText(comparison_image, 'Reconstructed', (image_size + 10, 30), font, 1, 0, 2)
    
    # Add error text
    error_text = f'MSE: {mse:.6f}'
    cv2.putText(comparison_image, error_text, (image_size + 10, image_size - 20), font, 0.7, 0, 2)
    
    # Save images
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    original_path = os.path.join(output_dir, f"original_kp_{args.sample_idx}.png")
    reconstructed_path = os.path.join(output_dir, f"reconstructed_kp_{args.sample_idx}.png")
    comparison_path = os.path.join(output_dir, f"comparison_kp_{args.sample_idx}.png")
    
    cv2.imwrite(original_path, original_image)
    cv2.imwrite(reconstructed_path, reconstructed_image)
    cv2.imwrite(comparison_path, comparison_image)
    
    print(f"  Saved original keypoints image: {original_path}")
    print(f"  Saved reconstructed keypoints image: {reconstructed_path}")
    print(f"  Saved comparison image: {comparison_path}")
    
    # Try to load and display original image if available
    if 'image_paths' in data and args.sample_idx < len(data['image_paths']):
        original_img_path = data['image_paths'][args.sample_idx]
        if os.path.exists(original_img_path):
            original_photo = cv2.imread(original_img_path)
            if original_photo is not None:
                # Resize original photo to match keypoint image size
                h, w = original_photo.shape[:2]
                scale = image_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                original_photo_resized = cv2.resize(original_photo, (new_w, new_h))
                
                # Create a canvas and center the image
                photo_canvas = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
                y_offset = (image_size - new_h) // 2
                x_offset = (image_size - new_w) // 2
                photo_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = original_photo_resized
                
                # Convert to grayscale for consistency
                photo_canvas_gray = cv2.cvtColor(photo_canvas, cv2.COLOR_BGR2GRAY)
                
                # Create full comparison: original photo | original kp | reconstructed kp
                full_comparison = np.hstack([
                    photo_canvas_gray,
                    original_image,
                    reconstructed_image
                ])
                
                # Add labels
                cv2.putText(full_comparison, 'Original Photo', (10, 30), font, 1, 0, 2)
                cv2.putText(full_comparison, 'Original KP', (image_size + 10, 30), font, 1, 0, 2)
                cv2.putText(full_comparison, 'Reconstructed KP', (image_size * 2 + 10, 30), font, 1, 0, 2)
                cv2.putText(full_comparison, error_text, (image_size * 2 + 10, image_size - 20), font, 0.7, 0, 2)
                
                full_comparison_path = os.path.join(output_dir, f"full_comparison_{args.sample_idx}.png")
                cv2.imwrite(full_comparison_path, full_comparison)
                print(f"  Saved full comparison (photo + keypoints): {full_comparison_path}")
    
    print("\n" + "="*60)
    print("Done!")

