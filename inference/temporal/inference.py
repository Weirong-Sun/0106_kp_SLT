"""
Inference script for temporal skeleton representation
Load trained temporal model and extract representations from video sequences
"""
import torch
import pickle
import numpy as np
import os
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from models.temporal.model import TemporalSkeletonTransformer
from models.skeleton.model import HierarchicalSkeletonTransformer

def load_model(checkpoint_path, device='cuda'):
    """
    Load trained temporal model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        model: Loaded model
        model_config: Model configuration
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model_config = checkpoint.get('model_config', {
        'd_global': 256,
        'd_region': 128,
        'd_temporal': 512,
        'd_final': 512,
        'num_temporal_layers': 4,
        'seq_len': 30
    })
    
    # Load frame encoder
    frame_config = {
        'd_global': model_config['d_global'],
        'd_region': model_config['d_region'],
        'num_regions': 4,
        'num_keypoints': 143
    }
    
    frame_encoder = HierarchicalSkeletonTransformer(
        input_dim=3,
        d_global=frame_config['d_global'],
        d_region=frame_config['d_region'],
        nhead=8,
        num_region_layers=2,
        num_interaction_layers=2,
        dim_feedforward=1024,
        dropout=0.1,
        num_keypoints=143,
        num_regions=4,
        image_size=256,
        num_channels=1
    ).to(device)
    
    # Load frame encoder state if available
    if 'frame_encoder_state_dict' in checkpoint:
        frame_encoder.load_state_dict(checkpoint['frame_encoder_state_dict'])
    frame_encoder.eval()
    
    # Create temporal model
    model = TemporalSkeletonTransformer(
        frame_encoder=frame_encoder,
        d_global=frame_config['d_global'],
        d_region=frame_config['d_region'],
        num_regions=4,
        d_temporal=model_config['d_temporal'],
        d_final=model_config['d_final'],
        nhead=8,
        num_temporal_layers=model_config['num_temporal_layers'],
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=model_config['seq_len'] * 2,
        freeze_frame_encoder=True,
        fusion_method='concat'
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, model_config

def prepare_sequence(kp_dict_list, normalization=None, device='cuda', seq_len=None):
    """
    Prepare keypoint sequence for model input
    
    Args:
        kp_dict_list: List of keypoint dictionaries (one per frame)
        normalization: Normalization parameters (min, max)
        device: Device to run on
        seq_len: Target sequence length (if None, use actual length)
    
    Returns:
        sequence: Prepared sequence tensor [1, seq_len, 143, 3]
        actual_len: Actual sequence length
    """
    seq_keypoints = []
    
    for kp_dict in kp_dict_list:
        kp_list = []
        if kp_dict.get('face') is not None:
            kp_list.append(kp_dict['face'])
        else:
            kp_list.append(np.zeros((68, 3), dtype=np.float32))
        if kp_dict.get('left_hand') is not None:
            kp_list.append(kp_dict['left_hand'])
        else:
            kp_list.append(np.zeros((21, 3), dtype=np.float32))
        if kp_dict.get('right_hand') is not None:
            kp_list.append(kp_dict['right_hand'])
        else:
            kp_list.append(np.zeros((21, 3), dtype=np.float32))
        if kp_dict.get('pose') is not None:
            kp_list.append(kp_dict['pose'])
        else:
            kp_list.append(np.zeros((33, 3), dtype=np.float32))
        flattened = np.concatenate(kp_list, axis=0)
        seq_keypoints.append(flattened)
    
    actual_len = len(seq_keypoints)
    
    # Pad or truncate to seq_len if specified
    if seq_len is not None:
        if len(seq_keypoints) < seq_len:
            # Pad with last frame
            seq_keypoints = seq_keypoints + [seq_keypoints[-1]] * (seq_len - len(seq_keypoints))
        elif len(seq_keypoints) > seq_len:
            # Truncate
            seq_keypoints = seq_keypoints[:seq_len]
    
    sequence = np.array(seq_keypoints)  # [seq_len, 143, 3]
    sequence = torch.FloatTensor(sequence).unsqueeze(0).to(device)  # [1, seq_len, 143, 3]
    
    # Normalize
    if normalization is not None:
        kp_min = torch.FloatTensor(normalization['kp_min']).to(device)
        kp_max = torch.FloatTensor(normalization['kp_max']).to(device)
        kp_range = kp_max - kp_min
        kp_range[kp_range < 1e-8] = 1.0
        sequence = (sequence - kp_min) / kp_range
    
    return sequence, actual_len

def extract_representations(model, kp_dict_list, normalization=None, device='cuda', seq_len=None):
    """
    Extract compressed temporal representations from video sequence
    
    Args:
        model: Trained temporal model
        kp_dict_list: List of keypoint dictionaries (one per frame)
        normalization: Normalization parameters
        device: Device to run on
        seq_len: Target sequence length
    
    Returns:
        Dictionary containing:
        - global_repr: Global representation [d_final] - attends to all frames
        - local_reprs: Local representations [num_local_vars, d_final] - different temporal windows
        - temporal_reprs: Temporal sequence representations [seq_len, d_temporal]
        - frame_reprs: Frame-level representations (global + regional)
    """
    sequence, actual_len = prepare_sequence(kp_dict_list, normalization, device, seq_len)
    
    with torch.no_grad():
        # Extract compressed representations (global + local)
        global_repr, local_reprs, temporal_reprs = model(sequence)
        
        # Extract frame-level representations for detailed analysis
        global_reprs, regional_reprs = model.encode_frames(sequence)
    
    return {
        'global_repr': global_repr[0].cpu().numpy(),  # [d_final] - global variable
        'local_reprs': local_reprs[0].cpu().numpy(),  # [num_local_vars, d_final] - local variables
        'temporal_reprs': temporal_reprs[0].cpu().numpy(),  # [seq_len, d_temporal]
        'global_reprs': global_reprs[0].cpu().numpy(),  # [seq_len, d_global] - per-frame global
        'regional_reprs': regional_reprs[0].cpu().numpy(),  # [seq_len, num_regions, d_region] - per-frame regional
        'actual_len': actual_len
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (will load TEMPORAL config)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to temporal model checkpoint')
    parser.add_argument('--video_sequences', type=str, default=None,
                        help='Path to video sequences pickle file')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of video sequences to process')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save representations')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Load from config if provided
    if args.config:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", args.config)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config_dict = config_module.TEMPORAL
        
        inference_config = config_dict.get('inference', {})
        args.checkpoint = args.checkpoint or inference_config.get('checkpoint')
        args.video_sequences = args.video_sequences or inference_config.get('video_sequences')
        args.num_samples = args.num_samples or inference_config.get('num_samples', 10)
        args.output_dir = args.output_dir or inference_config.get('output_dir')
    
    # Validate required arguments
    if not args.checkpoint:
        parser.error("--checkpoint is required")
    
    # Load model
    print("Loading temporal model...")
    model, model_config = load_model(args.checkpoint, device=args.device)
    print("Model loaded successfully!")
    print(f"Model config: d_temporal={model_config['d_temporal']}, d_final={model_config['d_final']}, seq_len={model_config['seq_len']}")
    
    # Load video sequences
    print(f"\nLoading video sequences from {args.video_sequences}...")
    with open(args.video_sequences, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict) and 'sequences' in data:
        video_sequences = data['sequences']
    elif isinstance(data, list):
        video_sequences = data
    else:
        raise ValueError("Unknown data format")
    
    num_samples = min(args.num_samples, len(video_sequences))
    print(f"Processing {num_samples} video sequences...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load normalization parameters from training data if available
    # We need to compute normalization from the sequences
    print("\nComputing normalization parameters...")
    all_kp_for_norm = []
    for seq in video_sequences[:num_samples]:
        for kp_dict in seq:
            kp_list = []
            if kp_dict.get('face') is not None:
                kp_list.append(kp_dict['face'])
            else:
                kp_list.append(np.zeros((68, 3), dtype=np.float32))
            if kp_dict.get('left_hand') is not None:
                kp_list.append(kp_dict['left_hand'])
            else:
                kp_list.append(np.zeros((21, 3), dtype=np.float32))
            if kp_dict.get('right_hand') is not None:
                kp_list.append(kp_dict['right_hand'])
            else:
                kp_list.append(np.zeros((21, 3), dtype=np.float32))
            if kp_dict.get('pose') is not None:
                kp_list.append(kp_dict['pose'])
            else:
                kp_list.append(np.zeros((33, 3), dtype=np.float32))
            flattened = np.concatenate(kp_list, axis=0)
            all_kp_for_norm.append(flattened)
    
    all_kp_for_norm = np.array(all_kp_for_norm)
    kp_min = torch.FloatTensor(all_kp_for_norm.reshape(-1, 3).min(axis=0))
    kp_max = torch.FloatTensor(all_kp_for_norm.reshape(-1, 3).max(axis=0))
    kp_range = kp_max - kp_min
    kp_range[kp_range < 1e-8] = 1.0
    normalization = {'kp_min': kp_min.numpy(), 'kp_max': kp_max.numpy()}
    print(f"Normalization: min={kp_min.numpy()}, max={kp_max.numpy()}")
    
    # Process sequences
    all_global_reprs = []
    all_local_reprs = []
    all_temporal_reprs = []
    
    print("\n" + "="*60)
    print("EXTRACTING COMPRESSED REPRESENTATIONS")
    print("="*60)
    
    for idx in range(num_samples):
        print(f"\nProcessing sequence {idx+1}/{num_samples}")
        print(f"  Sequence length: {len(video_sequences[idx])} frames")
        
        # Extract representations
        reprs = extract_representations(
            model, 
            video_sequences[idx], 
            normalization=normalization,  # Use computed normalization
            device=args.device,
            seq_len=model_config['seq_len']
        )
        
        all_global_reprs.append(reprs['global_repr'])
        all_local_reprs.append(reprs['local_reprs'])
        all_temporal_reprs.append(reprs['temporal_reprs'])
        
        # Print statistics
        print(f"  Global representation shape: {reprs['global_repr'].shape}")
        print(f"  Global repr - Mean: {reprs['global_repr'].mean():.4f}, Std: {reprs['global_repr'].std():.4f}")
        print(f"  Global repr - Min: {reprs['global_repr'].min():.4f}, Max: {reprs['global_repr'].max():.4f}")
        print(f"  Global repr - First 5 dims: {reprs['global_repr'][:5]}")
        print(f"  Local representations shape: {reprs['local_reprs'].shape}")
        for i, local_repr in enumerate(reprs['local_reprs']):
            print(f"    Local var {i+1} - Mean: {local_repr.mean():.4f}, Std: {local_repr.std():.4f}")
        print(f"  Temporal reprs shape: {reprs['temporal_reprs'].shape}")
        
        # Save individual representation
        output_path = os.path.join(args.output_dir, f"sequence_{idx}_reprs.npz")
        np.savez(
            output_path,
            global_repr=reprs['global_repr'],
            local_reprs=reprs['local_reprs'],
            temporal_reprs=reprs['temporal_reprs'],
            global_reprs=reprs['global_reprs'],
            regional_reprs=reprs['regional_reprs'],
            actual_len=reprs['actual_len']
        )
        print(f"  Saved to: {output_path}")
    
    # Save all representations
    all_global_reprs = np.array(all_global_reprs)  # [num_samples, d_final]
    all_local_reprs = np.array(all_local_reprs)  # [num_samples, num_local_vars, d_final]
    all_temporal_reprs = np.array(all_temporal_reprs)  # [num_samples, seq_len, d_temporal]
    
    summary_path = os.path.join(args.output_dir, "all_representations.npz")
    np.savez(
        summary_path,
        global_reprs=all_global_reprs,
        local_reprs=all_local_reprs,
        temporal_reprs=all_temporal_reprs
    )
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total sequences processed: {num_samples}")
    print(f"Global representations shape: {all_global_reprs.shape}")
    print(f"Local representations shape: {all_local_reprs.shape}")
    print(f"Temporal representations shape: {all_temporal_reprs.shape}")
    print(f"\nGlobal representation statistics:")
    print(f"  Mean: {all_global_reprs.mean():.4f}")
    print(f"  Std: {all_global_reprs.std():.4f}")
    print(f"  Min: {all_global_reprs.min():.4f}")
    print(f"  Max: {all_global_reprs.max():.4f}")
    print(f"\nLocal representation statistics:")
    for i in range(all_local_reprs.shape[1]):
        local_var = all_local_reprs[:, i, :]
        print(f"  Local var {i+1}: Mean={local_var.mean():.4f}, Std={local_var.std():.4f}")
    print(f"\nTotal compressed representation size: {all_global_reprs.shape[1] + all_local_reprs.shape[1] * all_local_reprs.shape[2]} dimensions")
    print(f"\nSaved summary to: {summary_path}")
    
    # Compute similarity matrix (cosine similarity) using combined global + local representations
    print("\n" + "="*60)
    print("COMPUTING SIMILARITY MATRIX")
    print("="*60)
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Combine global and local representations for similarity computation
    # Flatten local_reprs: [num_samples, num_local_vars * d_final]
    local_reprs_flat = all_local_reprs.reshape(num_samples, -1)
    # Concatenate global and local: [num_samples, d_final + num_local_vars * d_final]
    combined_reprs = np.concatenate([all_global_reprs, local_reprs_flat], axis=1)
    
    similarity_matrix = cosine_similarity(combined_reprs)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Mean similarity: {similarity_matrix.mean():.4f}")
    
    # Exclude diagonal for off-diagonal statistics
    mask = ~np.eye(len(similarity_matrix), dtype=bool)
    off_diagonal = similarity_matrix[mask]
    print(f"Off-diagonal similarity:")
    print(f"  Mean: {off_diagonal.mean():.4f}")
    print(f"  Std: {off_diagonal.std():.4f}")
    print(f"  Min: {off_diagonal.min():.4f}")
    print(f"  Max: {off_diagonal.max():.4f}")
    
    # Check if representations are too similar (potential issue)
    if off_diagonal.mean() > 0.99:
        print("\n⚠️  WARNING: All representations are extremely similar!")
        print("   This may indicate:")
        print("   1. Model is not learning meaningful differences")
        print("   2. Representations are being normalized/standardized")
        print("   3. Model needs more training")
        
        # Print sample representations for debugging
        print("\nSample representations (first 3 sequences, first 10 dims of global):")
        for i in range(min(3, len(all_global_reprs))):
            print(f"  Seq {i} Global: {all_global_reprs[i][:10]}")
            print(f"  Seq {i} Local var 1: {all_local_reprs[i][0][:10]}")
    
    print(f"\nDiagonal (self-similarity): {np.diag(similarity_matrix).mean():.4f}")
    
    # Additional analysis: check representation diversity
    print("\n" + "="*60)
    print("REPRESENTATION DIVERSITY ANALYSIS")
    print("="*60)
    
    # Compute pairwise Euclidean distances
    from scipy.spatial.distance import pdist, squareform
    euclidean_distances = squareform(pdist(combined_reprs))
    print(f"Euclidean distance matrix shape: {euclidean_distances.shape}")
    mask = ~np.eye(len(euclidean_distances), dtype=bool)
    off_diagonal_dist = euclidean_distances[mask]
    print(f"Off-diagonal Euclidean distances:")
    print(f"  Mean: {off_diagonal_dist.mean():.4f}")
    print(f"  Std: {off_diagonal_dist.std():.4f}")
    print(f"  Min: {off_diagonal_dist.min():.4f}")
    print(f"  Max: {off_diagonal_dist.max():.4f}")
    
    # Check if representations are collapsed (all very similar)
    if off_diagonal_dist.mean() < 0.01:
        print("\n⚠️  WARNING: Representations appear to be collapsed!")
        print("   All sequences have nearly identical representations.")
        print("   Possible causes:")
        print("   1. Model needs more training")
        print("   2. Learning rate too high (causing collapse)")
        print("   3. Loss function issue")
        print("   4. Model capacity insufficient")
    
    # Check representation variance per dimension (for combined representation)
    repr_variance = combined_reprs.var(axis=0)
    print(f"\nPer-dimension variance:")
    print(f"  Mean variance: {repr_variance.mean():.6f}")
    print(f"  Max variance: {repr_variance.max():.6f}")
    print(f"  Min variance: {repr_variance.min():.6f}")
    print(f"  Dimensions with variance < 0.001: {(repr_variance < 0.001).sum()}/{len(repr_variance)}")
    
    # Save similarity matrix
    similarity_path = os.path.join(args.output_dir, "similarity_matrix.npy")
    np.save(similarity_path, similarity_matrix)
    print(f"\nSaved similarity matrix to: {similarity_path}")
    
    # Save distance matrix
    distance_path = os.path.join(args.output_dir, "distance_matrix.npy")
    np.save(distance_path, euclidean_distances)
    print(f"Saved distance matrix to: {distance_path}")
    
    print("\nDone!")

