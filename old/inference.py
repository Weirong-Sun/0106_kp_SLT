"""
Inference script for keypoint representation learning
Load trained model and extract keypoint representations
"""
import torch
import pickle
import numpy as np
from model import KeypointTransformer

def load_model(checkpoint_path, device='cuda'):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        model: Loaded model
        normalization: Normalization parameters
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = KeypointTransformer(
        input_dim=3,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        num_keypoints=68
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    normalization = checkpoint.get('normalization', None)
    
    return model, normalization

def extract_representation(model, keypoints, normalization=None, device='cuda'):
    """
    Extract latent representation from keypoints
    
    Args:
        model: Trained model
        keypoints: Keypoints array [68, 3] or [batch_size, 68, 3]
        normalization: Normalization parameters (min, max)
        device: Device to run on
    
    Returns:
        representation: Latent representation [num_keypoints, batch_size, d_model]
    """
    if normalization is not None:
        kp_min = torch.FloatTensor(normalization['kp_min']).to(device)
        kp_max = torch.FloatTensor(normalization['kp_max']).to(device)
        keypoints = (keypoints - kp_min) / (kp_max - kp_min + 1e-8)
    
    if len(keypoints.shape) == 2:
        keypoints = keypoints.unsqueeze(0)  # Add batch dimension
    
    keypoints = keypoints.to(device)
    
    with torch.no_grad():
        representation = model.encode(keypoints)
    
    return representation

def reconstruct_keypoints(model, representation, normalization=None, device='cuda'):
    """
    Reconstruct keypoints from latent representation
    
    Args:
        model: Trained model
        representation: Latent representation [num_keypoints, batch_size, d_model]
        normalization: Normalization parameters (min, max)
        device: Device to run on
    
    Returns:
        keypoints: Reconstructed keypoints [batch_size, 68, 3]
    """
    with torch.no_grad():
        keypoints = model.decode(representation)
    
    if normalization is not None:
        kp_min = torch.FloatTensor(normalization['kp_min']).to(device)
        kp_max = torch.FloatTensor(normalization['kp_max']).to(device)
        keypoints = keypoints * (kp_max - kp_min + 1e-8) + kp_min
    
    return keypoints

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--keypoints_data', type=str, default='keypoints_data.pkl')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index to test')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, normalization = load_model(args.checkpoint, device=args.device)
    print("Model loaded successfully!")
    
    # Load keypoints data
    print("Loading keypoints data...")
    with open(args.keypoints_data, 'rb') as f:
        data = pickle.load(f)
    
    keypoints = data['keypoints']
    sample_kp = torch.FloatTensor(keypoints[args.sample_idx])
    
    print(f"Original keypoints shape: {sample_kp.shape}")
    
    # Extract representation
    print("Extracting representation...")
    representation = extract_representation(model, sample_kp, normalization, args.device)
    print(f"Representation shape: {representation.shape}")
    
    # Reconstruct keypoints
    print("Reconstructing keypoints...")
    reconstructed = reconstruct_keypoints(model, representation, normalization, args.device)
    reconstructed = reconstructed[0].cpu().numpy()  # Remove batch dimension
    
    # Compute reconstruction error
    original_normalized = sample_kp.numpy()
    if normalization is not None:
        kp_min = normalization['kp_min']
        kp_max = normalization['kp_max']
        original_normalized = (original_normalized - kp_min) / (kp_max - kp_min + 1e-8)
    
    mse = np.mean((original_normalized - reconstructed) ** 2)
    print(f"Reconstruction MSE: {mse:.6f}")
    
    print("\nDone!")


