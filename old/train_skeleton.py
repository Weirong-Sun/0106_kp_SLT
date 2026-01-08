"""
Training script for skeleton reconstruction from body keypoints
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from tqdm import tqdm
import os
from model_skeleton import SkeletonReconstructionTransformer
from utils_skeleton import generate_skeleton_dataset

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
        self.images = torch.FloatTensor(images_data) / 255.0 * 2.0 - 1.0  # [0, 255] -> [-1, 1]
        self.images = self.images.unsqueeze(1)  # Add channel dimension [num_samples, 1, H, W]
    
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
    # Hand regions are typically in the lower-middle and lower-right areas
    # Face region is typically in the upper-middle area
    weight_mask = torch.ones_like(target)
    
    # Define approximate regions (normalized coordinates)
    # These are rough estimates and can be adjusted based on your data
    h, w = image_size, image_size
    
    # Hand regions (lower part of image, left and right sides)
    # Left hand region: left side, lower half
    # Right hand region: right side, lower half
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
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    image_size=256,
    save_dir="checkpoints_skeleton",
    use_weighted_loss=True,
    hand_weight=2.0,
    face_weight=1.5
):
    # Load data
    print("Loading keypoints data...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    keypoints_list = data['keypoints']
    print(f"Loaded {len(keypoints_list)} samples")
    
    # Split data
    train_size = int(0.8 * len(keypoints_list))
    train_keypoints = keypoints_list[:train_size]
    val_keypoints = keypoints_list[train_size:]
    
    # Create datasets
    print("Creating training dataset...")
    train_dataset = SkeletonDataset(train_keypoints, image_size=image_size, normalize_kp=True)
    
    print("Creating validation dataset...")
    val_dataset = SkeletonDataset(val_keypoints, image_size=image_size, normalize_kp=True)
    
    # Use normalization stats from training set for validation
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
    
    model = SkeletonReconstructionTransformer(
        input_dim=3,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_keypoints=143,  # 68 + 21 + 21 + 33
        image_size=image_size,
        num_channels=1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Print model structure
    print("\nModel structure:")
    print(f"  Input keypoints: [batch, 143, 3]")
    print(f"  Output image: [batch, 1, {image_size}, {image_size}]")
    
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
                    'd_model': d_model,
                    'num_keypoints': 143,
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='sign_language_keypoints.pkl',
                        help='Path to body keypoints pickle file')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--save_dir', type=str, default='checkpoints_skeleton')
    parser.add_argument('--use_weighted_loss', action='store_true', default=True,
                        help='Use weighted loss to emphasize hand regions')
    parser.add_argument('--hand_weight', type=float, default=2.0,
                        help='Weight multiplier for hand regions')
    parser.add_argument('--face_weight', type=float, default=1.5,
                        help='Weight multiplier for face region')
    
    args = parser.parse_args()
    
    train(
        data_path=args.data_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        image_size=args.image_size,
        save_dir=args.save_dir,
        use_weighted_loss=args.use_weighted_loss,
        hand_weight=args.hand_weight,
        face_weight=args.face_weight
    )

