"""
Training script for hierarchical skeleton reconstruction from body keypoints
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
from models.skeleton.model import HierarchicalSkeletonTransformer
from utils.utils_skeleton import generate_skeleton_dataset

class SkeletonDataset(Dataset):
    def __init__(self, keypoints_data, images_data=None, image_size=256, normalize_kp=True):
        """
        Args:
            keypoints_data: List of keypoint dictionaries
            images_data: numpy array of shape [num_samples, image_size, image_size] (optional)
                        If None, will generate images from keypoints
            image_size: Size of images
            normalize_kp: Whether to normalize keypoints
        """
        self.image_size = image_size
        self.normalize_kp = normalize_kp
        
        # Flatten keypoints to a single array
        # Format: [face(68), left_hand(21), right_hand(21), pose(33)] = 143 points
        flattened_keypoints = []
        for kp_dict in keypoints_data:
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
            
            # Concatenate all keypoints
            flattened = np.concatenate(kp_list, axis=0)  # [143, 3]
            flattened_keypoints.append(flattened)
        
        self.keypoints = torch.FloatTensor(np.array(flattened_keypoints))  # [num_samples, 143, 3]
        
        if normalize_kp:
            # Normalize to [0, 1] range per coordinate
            self.kp_min = self.keypoints.view(-1, 3).min(0)[0]
            self.kp_max = self.keypoints.view(-1, 3).max(0)[0]
            # Avoid division by zero
            kp_range = self.kp_max - self.kp_min
            kp_range[kp_range < 1e-8] = 1.0
            self.keypoints = (self.keypoints - self.kp_min) / kp_range
        
        if images_data is None:
            # Generate images from keypoints
            print("Generating ground truth skeleton images from keypoints...")
            images_data = generate_skeleton_dataset(keypoints_data, image_size=image_size)
        
        # Convert images to tensor and normalize to [-1, 1]
        self.images = torch.FloatTensor(images_data).unsqueeze(1) / 255.0 * 2.0 - 1.0  # [num_samples, 1, image_size, image_size]
    
    def __len__(self):
        return len(self.keypoints)
    
    def __getitem__(self, idx):
        kp = self.keypoints[idx]
        img = self.images[idx]
        return kp, img

def compute_weighted_loss(generated, target, image_size=256, hand_weight=2.0, face_weight=1.5):
    """
    Compute weighted loss with higher weight for hand regions
    
    Args:
        generated: Generated images [batch, 1, H, W]
        target: Target images [batch, 1, H, W]
        image_size: Size of images
        hand_weight: Weight multiplier for hand regions
        face_weight: Weight multiplier for face region
    
    Returns:
        loss: Weighted MSE loss
    """
    # Create weight mask
    weight_mask = torch.ones_like(target)
    
    # Define approximate regions (normalized coordinates)
    h, w = image_size, image_size
    
    # Hand regions (lower part of image, left and right sides)
    hand_left_x_start, hand_left_x_end = int(0.1 * w), int(0.4 * w)
    hand_right_x_start, hand_right_x_end = int(0.6 * w), int(0.9 * w)
    hand_y_start, hand_y_end = int(0.5 * h), int(0.95 * h)
    
    # Face region (upper-middle)
    face_x_start, face_x_end = int(0.2 * w), int(0.8 * w)
    face_y_start, face_y_end = int(0.1 * h), int(0.5 * h)
    
    # Apply weights
    weight_mask[:, :, hand_y_start:hand_y_end, hand_left_x_start:hand_left_x_end] *= hand_weight
    weight_mask[:, :, hand_y_start:hand_y_end, hand_right_x_start:hand_right_x_end] *= hand_weight
    weight_mask[:, :, face_y_start:face_y_end, face_x_start:face_x_end] *= face_weight
    
    # Compute weighted MSE loss
    loss = torch.mean(weight_mask * (generated - target) ** 2)
    
    return loss

def train_epoch(model, dataloader, criterion, optimizer, device, use_weighted_loss=True, image_size=256, hand_weight=2.0, face_weight=1.5):
    model.train()
    total_loss = 0
    
    for keypoints, images in tqdm(dataloader, desc="Training"):
        keypoints = keypoints.to(device)
        images = images.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        generated_images = model(keypoints)
        
        # Compute loss
        if use_weighted_loss:
            loss = compute_weighted_loss(generated_images, images, image_size=image_size, 
                                        hand_weight=hand_weight, face_weight=face_weight)
        else:
            loss = criterion(generated_images, images)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device, use_weighted_loss=True, image_size=256, hand_weight=2.0, face_weight=1.5):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for keypoints, images in tqdm(dataloader, desc="Validating"):
            keypoints = keypoints.to(device)
            images = images.to(device)
            
            generated_images = model(keypoints)
            
            if use_weighted_loss:
                loss = compute_weighted_loss(generated_images, images, image_size=image_size,
                                            hand_weight=hand_weight, face_weight=face_weight)
            else:
                loss = criterion(generated_images, images)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train(
    data_path,
    batch_size=16,
    epochs=100,
    lr=1e-4,
    d_global=256,
    d_region=128,
    nhead=8,
    num_region_layers=2,
    num_interaction_layers=2,
    dim_feedforward=1024,
    dropout=0.1,
    image_size=256,
    save_dir="checkpoints_skeleton_hierarchical",
    use_weighted_loss=True,
    hand_weight=2.0,
    face_weight=1.5
):
    # Load data
    print("Loading keypoints data...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different data formats
    if isinstance(data, dict) and 'keypoints' in data:
        keypoints_list = data['keypoints']
    elif isinstance(data, list):
        keypoints_list = data
    else:
        raise ValueError("Unknown data format. Expected dict with 'keypoints' key or list of keypoint dicts.")
    
    print(f"Loaded {len(keypoints_list)} samples")
    
    # Split train/val (80/20)
    split_idx = int(len(keypoints_list) * 0.8)
    train_keypoints = keypoints_list[:split_idx]
    val_keypoints = keypoints_list[split_idx:]
    
    print(f"Train samples: {len(train_keypoints)}, Val samples: {len(val_keypoints)}")
    
    # Create datasets
    train_dataset = SkeletonDataset(train_keypoints, image_size=image_size, normalize_kp=True)
    val_dataset = SkeletonDataset(val_keypoints, image_size=image_size, normalize_kp=False)
    
    # Use training normalization for validation
    val_dataset.kp_min = train_dataset.kp_min
    val_dataset.kp_max = train_dataset.kp_max
    kp_range = val_dataset.kp_max - val_dataset.kp_min
    kp_range[kp_range < 1e-8] = 1.0
    
    # Re-normalize validation keypoints
    val_flattened = []
    for kp_dict in val_keypoints:
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
        val_flattened.append(flattened)
    
    val_dataset.keypoints = torch.FloatTensor(np.array(val_flattened))
    val_dataset.keypoints = (val_dataset.keypoints - val_dataset.kp_min) / kp_range
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = HierarchicalSkeletonTransformer(
        input_dim=3,
        d_global=d_global,
        d_region=d_region,
        nhead=nhead,
        num_region_layers=num_region_layers,
        num_interaction_layers=num_interaction_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_keypoints=143,  # 68 + 21 + 21 + 33
        num_regions=4,  # Face, left_hand, right_hand, pose
        image_size=image_size,
        num_channels=1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Print model structure
    print("\nModel structure:")
    print(f"  Input keypoints: [batch, 143, 3]")
    print(f"  Global representation: [batch, {d_global}]")
    print(f"  Regional representation: [batch, 4, {d_region}]")
    print(f"  Output image: [batch, 1, {image_size}, {image_size}]")
    print(f"  Regions: Face(68), Left_hand(21), Right_hand(21), Pose(33)")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Use weighted loss to emphasize hand regions
    print(f"\nUsing weighted loss: {use_weighted_loss}")
    if use_weighted_loss:
        print(f"  Hand region weight: {hand_weight}x")
        print(f"  Face region weight: {face_weight}x")
        print(f"  Other regions weight: 1.0x")
    
    # Training loop
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, 
                                use_weighted_loss=use_weighted_loss, image_size=image_size,
                                hand_weight=hand_weight, face_weight=face_weight)
        val_loss = validate(model, val_loader, criterion, device, 
                           use_weighted_loss=use_weighted_loss, image_size=image_size,
                           hand_weight=hand_weight, face_weight=face_weight)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'normalization': {
                    'kp_min': train_dataset.kp_min.numpy(),
                    'kp_max': train_dataset.kp_max.numpy()
                },
                'model_config': {
                    'd_global': d_global,
                    'd_region': d_region,
                    'num_keypoints': 143,
                    'num_regions': 4,
                    'image_size': image_size
                }
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"Saved best model with val loss: {val_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print("\nTraining completed!")

def train_from_config(config_dict):
    """Train from configuration dictionary"""
    model_config = config_dict.get('model', {})
    training_config = config_dict.get('training', {})
    
    train(
        data_path=training_config.get('data_path', 'sign_language_keypoints.pkl'),
        batch_size=training_config.get('batch_size', 16),
        epochs=training_config.get('epochs', 100),
        lr=training_config.get('lr', 1e-4),
        d_global=model_config.get('d_global', 256),
        d_region=model_config.get('d_region', 128),
        nhead=model_config.get('nhead', 8),
        num_region_layers=model_config.get('num_region_layers', 2),
        num_interaction_layers=model_config.get('num_interaction_layers', 2),
        dim_feedforward=model_config.get('dim_feedforward', 1024),
        dropout=model_config.get('dropout', 0.1),
        image_size=model_config.get('image_size', 256),
        save_dir=training_config.get('save_dir', 'checkpoints_skeleton_hierarchical'),
        use_weighted_loss=training_config.get('use_weighted_loss', True),
        hand_weight=training_config.get('hand_weight', 2.0),
        face_weight=training_config.get('face_weight', 1.5)
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (will load SKELETON config)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to body keypoints pickle file')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--d_global', type=int, default=None,
                        help='Dimension of global representation')
    parser.add_argument('--d_region', type=int, default=None,
                        help='Dimension of regional representation')
    parser.add_argument('--nhead', type=int, default=None)
    parser.add_argument('--num_region_layers', type=int, default=None,
                        help='Number of layers for regional encoding')
    parser.add_argument('--num_interaction_layers', type=int, default=None,
                        help='Number of layers for cross-region interaction')
    parser.add_argument('--dim_feedforward', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--image_size', type=int, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--use_weighted_loss', action='store_true', default=None,
                        help='Use weighted loss to emphasize hand regions')
    parser.add_argument('--hand_weight', type=float, default=None,
                        help='Weight multiplier for hand regions')
    parser.add_argument('--face_weight', type=float, default=None,
                        help='Weight multiplier for face region')
    
    args = parser.parse_args()
    
    # Load from config if provided
    if args.config:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", args.config)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config_dict = config_module.SKELETON
        
        # Override with command line arguments if provided
        if args.data_path: config_dict['training']['data_path'] = args.data_path
        if args.batch_size is not None: config_dict['training']['batch_size'] = args.batch_size
        if args.epochs is not None: config_dict['training']['epochs'] = args.epochs
        if args.lr is not None: config_dict['training']['lr'] = args.lr
        if args.d_global is not None: config_dict['model']['d_global'] = args.d_global
        if args.d_region is not None: config_dict['model']['d_region'] = args.d_region
        if args.nhead is not None: config_dict['model']['nhead'] = args.nhead
        if args.num_region_layers is not None: config_dict['model']['num_region_layers'] = args.num_region_layers
        if args.num_interaction_layers is not None: config_dict['model']['num_interaction_layers'] = args.num_interaction_layers
        if args.dim_feedforward is not None: config_dict['model']['dim_feedforward'] = args.dim_feedforward
        if args.dropout is not None: config_dict['model']['dropout'] = args.dropout
        if args.image_size is not None: config_dict['model']['image_size'] = args.image_size
        if args.save_dir: config_dict['training']['save_dir'] = args.save_dir
        if args.use_weighted_loss is not None: config_dict['training']['use_weighted_loss'] = args.use_weighted_loss
        if args.hand_weight is not None: config_dict['training']['hand_weight'] = args.hand_weight
        if args.face_weight is not None: config_dict['training']['face_weight'] = args.face_weight
        
        train_from_config(config_dict)
    else:
        # Use command line arguments with defaults
        train(
            data_path=args.data_path or 'sign_language_keypoints.pkl',
            batch_size=args.batch_size or 16,
            epochs=args.epochs or 100,
            lr=args.lr or 1e-4,
            d_global=args.d_global or 256,
            d_region=args.d_region or 128,
            nhead=args.nhead or 8,
            num_region_layers=args.num_region_layers or 2,
            num_interaction_layers=args.num_interaction_layers or 2,
            dim_feedforward=args.dim_feedforward or 1024,
            dropout=args.dropout or 0.1,
            image_size=args.image_size or 256,
            save_dir=args.save_dir or 'checkpoints_skeleton_hierarchical',
            use_weighted_loss=args.use_weighted_loss if args.use_weighted_loss is not None else True,
            hand_weight=args.hand_weight or 2.0,
            face_weight=args.face_weight or 1.5
        )

