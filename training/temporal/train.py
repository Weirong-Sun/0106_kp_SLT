"""
Training script for temporal skeleton representation
Two-stage training strategy:
1. Stage 1: Freeze frame encoder, train temporal transformer (self-supervised)
2. Stage 2: End-to-end fine-tuning
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from tqdm import tqdm
import os
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from models.temporal.model import TemporalSkeletonTransformer
from models.skeleton.model import HierarchicalSkeletonTransformer

class TemporalSkeletonDataset(Dataset):
    """
    Dataset for temporal skeleton sequences
    """
    def __init__(self, video_sequences, seq_len=30, normalize_kp=True, augmentation=False):
        """
        Args:
            video_sequences: List of video sequences, each is a list of keypoint dicts
            seq_len: Length of sequence to use
            normalize_kp: Whether to normalize keypoints
            augmentation: Whether to use data augmentation
        """
        self.seq_len = seq_len
        self.normalize_kp = normalize_kp
        self.augmentation = augmentation
        
        # Process sequences
        self.sequences = []
        for video_seq in video_sequences:
            if len(video_seq) < seq_len:
                # Pad short sequences
                padded_seq = video_seq + [video_seq[-1]] * (seq_len - len(video_seq))
                self.sequences.append(padded_seq)
            else:
                # Sliding window for long sequences
                for i in range(len(video_seq) - seq_len + 1):
                    self.sequences.append(video_seq[i:i+seq_len])
        
        print(f"Created {len(self.sequences)} sequences of length {seq_len}")
        
        # Flatten all keypoints for normalization
        all_keypoints = []
        for seq in self.sequences:
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
                all_keypoints.append(flattened)
        
        all_keypoints = np.array(all_keypoints)  # [total_frames, 143, 3]
        
        if normalize_kp:
            self.kp_min = torch.FloatTensor(all_keypoints.reshape(-1, 3).min(axis=0))
            self.kp_max = torch.FloatTensor(all_keypoints.reshape(-1, 3).max(axis=0))
            kp_range = self.kp_max - self.kp_min
            kp_range[kp_range < 1e-8] = 1.0
            self.kp_range = kp_range
        else:
            self.kp_min = None
            self.kp_max = None
            self.kp_range = None
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Convert sequence to tensor
        seq_keypoints = []
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
            seq_keypoints.append(flattened)
        
        seq_keypoints = np.array(seq_keypoints)  # [seq_len, 143, 3]
        seq_keypoints = torch.FloatTensor(seq_keypoints)
        
        # Normalize
        if self.normalize_kp and self.kp_min is not None:
            seq_keypoints = (seq_keypoints - self.kp_min) / self.kp_range
        
        return seq_keypoints  # [seq_len, 143, 3]

def masked_frame_prediction_loss(model, sequences, mask_ratio=0.15, device='cuda'):
    """
    Self-supervised loss: predict masked frames
    
    Args:
        model: Temporal model
        sequences: [batch, seq_len, num_kp, 3]
        mask_ratio: Ratio of frames to mask
        device: Device
    
    Returns:
        loss: Reconstruction loss
    """
    batch_size, seq_len = sequences.shape[:2]
    
    # Ensure at least 1 frame is masked (but not all)
    num_masked = max(1, min(int(seq_len * mask_ratio), seq_len - 1))
    
    # Get original frame representations first (detached)
    with torch.no_grad():
        _, original_temporal = model(sequences)
    
    # Randomly mask frames
    masked_indices = []
    valid_batches = []
    for b in range(batch_size):
        indices = torch.randperm(seq_len, device=device)[:num_masked]
        masked_indices.append(indices)
        valid_batches.append(b)
    
    # Create masked sequences (replace with zeros)
    masked_sequences = sequences.clone()
    for b, indices in enumerate(masked_indices):
        masked_sequences[b, indices] = 0  # Simple masking
    
    # Encode masked sequence
    final_repr, temporal_reprs = model(masked_sequences)  # [batch, d_final], [batch, seq_len, d_temporal]
    
    # Loss on masked positions
    loss_list = []
    for b, indices in enumerate(masked_indices):
        if len(indices) > 0:
            pred = temporal_reprs[b, indices]
            target = original_temporal[b, indices]
            # Add small epsilon for numerical stability
            mse = nn.functional.mse_loss(pred, target, reduction='mean')
            if torch.isfinite(mse):
                loss_list.append(mse)
    
    if len(loss_list) == 0:
        # Fallback: use next frame prediction if masking fails
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    loss = torch.stack(loss_list).mean()
    
    # Check for NaN or Inf
    if not torch.isfinite(loss):
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return loss

def reconstruction_loss(model, sequences, device='cuda'):
    """
    Reconstruction loss: encode and decode sequence using compressed representations
    
    Args:
        model: Temporal model
        sequences: [batch, seq_len, num_kp, 3]
        device: Device
    
    Returns:
        loss: Reconstruction loss (MSE between original and reconstructed)
    """
    batch_size, seq_len = sequences.shape[:2]
    
    # Encode to compressed representations and decode
    global_repr, local_reprs, temporal_reprs, reconstructed = model(sequences, return_reconstruction=True)
    # global_repr: [batch, d_final]
    # local_reprs: [batch, num_local_vars, d_final]
    # reconstructed: [batch, seq_len, num_kp, 3]
    
    # Compute MSE loss
    loss = nn.functional.mse_loss(reconstructed, sequences)
    
    return loss

def next_frame_prediction_loss(model, sequences, device='cuda'):
    """
    Self-supervised loss: predict next frame (optional, can be combined with reconstruction)
    
    Args:
        model: Temporal model
        sequences: [batch, seq_len, num_kp, 3]
        device: Device
    
    Returns:
        loss: Prediction loss
    """
    # Use first seq_len-1 frames to predict last frame
    input_seq = sequences[:, :-1]  # [batch, seq_len-1, num_kp, 3]
    target_frame = sequences[:, -1]  # [batch, num_kp, 3]
    
    # Encode sequence
    final_repr, temporal_reprs = model(input_seq)
    
    # Predict last frame representation
    predicted_last = temporal_reprs[:, -1]  # [batch, d_temporal]
    
    # Get target frame representation
    with torch.no_grad():
        target_seq = target_frame.unsqueeze(1)  # [batch, 1, num_kp, 3]
        _, target_temporal = model(target_seq)
        target_last = target_temporal[:, 0]  # [batch, d_temporal]
    
    # MSE loss
    loss = nn.functional.mse_loss(predicted_last, target_last)
    
    return loss

def train_stage1(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    use_masked_frame=False,
    use_next_frame=True
):
    """
    Stage 1: Freeze frame encoder, train temporal transformer
    """
    model.train()
    # Ensure frame encoder is frozen
    for param in model.frame_encoder.parameters():
        param.requires_grad = False
    
    total_loss = 0
    num_batches = 0
    
    for sequences in tqdm(dataloader, desc=f"Stage1 Epoch {epoch+1}"):
        sequences = sequences.to(device)  # [batch, seq_len, 143, 3]
        
        optimizer.zero_grad()
        
        # Reconstruction loss (main training objective)
        loss = reconstruction_loss(model, sequences, device=device)
        
        # Check for NaN or Inf before backward
        if torch.isfinite(loss):
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 
                max_norm=1.0
            )
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        else:
            print(f"Warning: NaN/Inf loss detected, skipping batch")
            optimizer.zero_grad()  # Clear gradients
    
    return total_loss / max(num_batches, 1)

# Stage 2 removed - we only use Stage 1 with compressed representations

def validate(model, dataloader, device):
    """Validation using reconstruction loss"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for sequences in tqdm(dataloader, desc="Validating"):
            sequences = sequences.to(device)
            
            # Use reconstruction loss as validation metric
            loss = reconstruction_loss(model, sequences, device=device)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)

def train(
    video_data_path,
    frame_encoder_checkpoint,
    batch_size=8,
    seq_len=30,
    stage1_epochs=50,
    lr_stage1=1e-4,
    d_temporal=512,
    d_final=512,
    nhead=8,
    num_temporal_layers=4,
    dim_feedforward=2048,
    dropout=0.1,
    save_dir="checkpoints_temporal",
    freeze_frame_encoder=True,
    use_masked_frame=False,
    use_next_frame=True
):
    # Load video sequences
    print("Loading video sequences...")
    with open(video_data_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict) and 'sequences' in data:
        video_sequences = data['sequences']
    elif isinstance(data, list):
        video_sequences = data
    else:
        raise ValueError("Unknown data format")
    
    print(f"Loaded {len(video_sequences)} video sequences")
    
    # Split train/val
    split_idx = int(len(video_sequences) * 0.8)
    train_sequences = video_sequences[:split_idx]
    val_sequences = video_sequences[split_idx:]
    
    print(f"Train sequences: {len(train_sequences)}, Val sequences: {len(val_sequences)}")
    
    # Create datasets
    train_dataset = TemporalSkeletonDataset(train_sequences, seq_len=seq_len, normalize_kp=True)
    val_dataset = TemporalSkeletonDataset(val_sequences, seq_len=seq_len, normalize_kp=True)
    # Use normalization stats from training set for validation
    val_dataset.kp_min = train_dataset.kp_min
    val_dataset.kp_max = train_dataset.kp_max
    val_dataset.kp_range = train_dataset.kp_range
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Load frame encoder
    print("Loading frame encoder...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    frame_checkpoint = torch.load(frame_encoder_checkpoint, map_location=device, weights_only=False)
    frame_config = frame_checkpoint.get('model_config', {
        'd_global': 256,
        'd_region': 128,
        'num_regions': 4,
        'num_keypoints': 143
    })
    
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
    
    frame_encoder.load_state_dict(frame_checkpoint['model_state_dict'])
    frame_encoder.eval()
    
    print("Frame encoder loaded successfully!")
    print(f"  Frame encoder config: d_global={frame_config['d_global']}, d_region={frame_config['d_region']}, num_regions={frame_config['num_regions']}")
    print(f"  Frame encoder will encode each frame to:")
    print(f"    - Global representation: [batch*seq_len, {frame_config['d_global']}]")
    print(f"    - Regional representation: [batch*seq_len, {frame_config['num_regions']}, {frame_config['d_region']}]")
    
    # Create temporal model
    model = TemporalSkeletonTransformer(
        frame_encoder=frame_encoder,
        d_global=frame_config['d_global'],
        d_region=frame_config['d_region'],
        num_regions=4,
        d_temporal=d_temporal,
        d_final=d_final,
        nhead=nhead,
        num_temporal_layers=num_temporal_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_seq_len=seq_len * 2,
        freeze_frame_encoder=freeze_frame_encoder,
        fusion_method='concat'
    ).to(device)
    
    print(f"Temporal model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Print model architecture flow
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE FLOW")
    print("="*60)
    print("Input: Video sequence [batch, seq_len, 143, 3]")
    print("\nStep 1: Frame-level Encoding (using pre-trained hierarchical encoder)")
    print(f"  - Each frame [143, 3] → Global [{frame_config['d_global']}] + Regional [{frame_config['num_regions']}×{frame_config['d_region']}]")
    print(f"  - Output: Global [batch, seq_len, {frame_config['d_global']}], Regional [batch, seq_len, {frame_config['num_regions']}, {frame_config['d_region']}]")
    print(f"  - Frame encoder status: {'FROZEN' if freeze_frame_encoder else 'TRAINABLE'}")
    print("\nStep 2: Temporal Fusion")
    print(f"  - Combine global + regional → Temporal representation [{d_temporal}]")
    print(f"  - Output: [batch, seq_len, {d_temporal}]")
    print("\nStep 3: Temporal Transformer Encoding")
    print(f"  - Learn temporal dependencies across frames")
    print(f"  - Output: [batch, seq_len, {d_temporal}]")
    print("\nStep 4: Multi-Scale Compressed Representations")
    print(f"  - Global variable: Attention pooling over all frames → [{d_final}]")
    print(f"  - Local variables: Attention pooling over temporal windows → [{model.num_local_vars}, {d_final}]")
    print(f"    * Pattern 0: frames [1, 3, 5, ...] (stride=2, offset=0)")
    print(f"    * Pattern 1: frames [2, 4, 6, ...] (stride=2, offset=1)")
    print(f"  - Output: Global [batch, {d_final}], Local [batch, {model.num_local_vars}, {d_final}]")
    print("="*60 + "\n")
    
    # Print training configuration
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Masked frame prediction: {use_masked_frame}")
    print(f"Next frame prediction: {use_next_frame}")
    print(f"Frame encoder frozen: {freeze_frame_encoder}")
    print("="*60)
    
    # Stage 1: Train temporal transformer
    print("\n" + "="*60)
    print("STAGE 1: Training Temporal Transformer (Frame Encoder Frozen)")
    print("="*60)
    
    optimizer_stage1 = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr_stage1
    )
    scheduler_stage1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_stage1, mode='min', factor=0.5, patience=5)
    
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    for epoch in range(stage1_epochs):
        print(f"\nEpoch {epoch+1}/{stage1_epochs}")
        
        train_loss = train_stage1(
            model, train_loader, optimizer_stage1, device, epoch,
            use_masked_frame=use_masked_frame,
            use_next_frame=use_next_frame
        )
        val_loss = validate(model, val_loader, device)
        
        scheduler_stage1.step(val_loss)
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'frame_encoder_state_dict': frame_encoder.state_dict(),
                'optimizer_state_dict': optimizer_stage1.state_dict(),
                'val_loss': val_loss,
                'stage': 1,
                'model_config': {
                    'd_global': frame_config['d_global'],
                    'd_region': frame_config['d_region'],
                    'd_temporal': d_temporal,
                    'd_final': d_final,
                    'num_temporal_layers': num_temporal_layers,
                    'seq_len': seq_len
                }
            }, os.path.join(save_dir, 'best_model_stage1.pth'))
            print(f"Saved best Stage 1 model")
    
    print("\nTraining completed!")
    print(f"\nCompressed representation structure:")
    print(f"  - Global variable: [batch, {d_final}] (attends to all frames)")
    print(f"  - Local variables: [batch, {model.num_local_vars}, {d_final}] (different temporal windows)")
    print(f"  - Total representation size: {d_final + model.num_local_vars * d_final} dimensions")

def train_from_config(config_dict):
    """Train from configuration dictionary"""
    model_config = config_dict.get('model', {})
    training_config = config_dict.get('training', {})
    
    train(
        video_data_path=training_config.get('video_data_path'),
        frame_encoder_checkpoint=training_config.get('frame_encoder_checkpoint'),
        batch_size=training_config.get('batch_size', 8),
        seq_len=training_config.get('seq_len', 6),
        stage1_epochs=training_config.get('epochs', 50),
        lr_stage1=training_config.get('lr', 1e-4),
        d_temporal=model_config.get('d_temporal', 512),
        d_final=model_config.get('d_final', 512),
        nhead=model_config.get('nhead', 8),
        num_temporal_layers=model_config.get('num_temporal_layers', 4),
        dim_feedforward=model_config.get('dim_feedforward', 2048),
        dropout=model_config.get('dropout', 0.1),
        save_dir=training_config.get('save_dir', 'checkpoints_temporal'),
        freeze_frame_encoder=model_config.get('freeze_frame_encoder', True),
        use_masked_frame=False,  # Not used in current training
        use_next_frame=False  # Not used in current training
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (will load TEMPORAL config)')
    parser.add_argument('--video_data_path', type=str, default=None,
                        help='Path to video sequences pickle file')
    parser.add_argument('--frame_encoder_checkpoint', type=str, default=None,
                        help='Path to pre-trained frame encoder checkpoint')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--seq_len', type=int, default=None,
                        help='Sequence length')
    parser.add_argument('--stage1_epochs', type=int, default=None)
    parser.add_argument('--lr_stage1', type=float, default=None)
    parser.add_argument('--d_temporal', type=int, default=None)
    parser.add_argument('--d_final', type=int, default=None)
    parser.add_argument('--nhead', type=int, default=None)
    parser.add_argument('--num_temporal_layers', type=int, default=None)
    parser.add_argument('--dim_feedforward', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--freeze_frame_encoder', action='store_true', default=None)
    
    args = parser.parse_args()
    
    # Load from config if provided
    if args.config:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", args.config)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config_dict = config_module.TEMPORAL
        
        # Override with command line arguments if provided
        if args.video_data_path: config_dict['training']['video_data_path'] = args.video_data_path
        if args.frame_encoder_checkpoint: config_dict['training']['frame_encoder_checkpoint'] = args.frame_encoder_checkpoint
        if args.batch_size is not None: config_dict['training']['batch_size'] = args.batch_size
        if args.seq_len is not None: config_dict['training']['seq_len'] = args.seq_len
        if args.stage1_epochs is not None: config_dict['training']['epochs'] = args.stage1_epochs
        if args.lr_stage1 is not None: config_dict['training']['lr'] = args.lr_stage1
        if args.d_temporal is not None: config_dict['model']['d_temporal'] = args.d_temporal
        if args.d_final is not None: config_dict['model']['d_final'] = args.d_final
        if args.nhead is not None: config_dict['model']['nhead'] = args.nhead
        if args.num_temporal_layers is not None: config_dict['model']['num_temporal_layers'] = args.num_temporal_layers
        if args.dim_feedforward is not None: config_dict['model']['dim_feedforward'] = args.dim_feedforward
        if args.dropout is not None: config_dict['model']['dropout'] = args.dropout
        if args.save_dir: config_dict['training']['save_dir'] = args.save_dir
        if args.freeze_frame_encoder is not None: config_dict['model']['freeze_frame_encoder'] = args.freeze_frame_encoder
        
        train_from_config(config_dict)
    else:
        # Use command line arguments
        if not args.video_data_path or not args.frame_encoder_checkpoint:
            parser.error("--video_data_path and --frame_encoder_checkpoint are required when not using --config")
        
        train(
            video_data_path=args.video_data_path,
            frame_encoder_checkpoint=args.frame_encoder_checkpoint,
            batch_size=args.batch_size or 8,
            seq_len=args.seq_len or 6,
            stage1_epochs=args.stage1_epochs or 50,
            lr_stage1=args.lr_stage1 or 1e-4,
            d_temporal=args.d_temporal or 512,
            d_final=args.d_final or 512,
            nhead=args.nhead or 8,
            num_temporal_layers=args.num_temporal_layers or 4,
            dim_feedforward=args.dim_feedforward or 2048,
            dropout=args.dropout or 0.1,
            save_dir=args.save_dir or 'checkpoints_temporal',
            freeze_frame_encoder=args.freeze_frame_encoder if args.freeze_frame_encoder is not None else True,
            use_masked_frame=False,
            use_next_frame=False
        )

