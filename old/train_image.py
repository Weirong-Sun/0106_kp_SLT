"""
Training script for keypoint to image reconstruction
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from tqdm import tqdm
import os
from model_image import KeypointToImageTransformer
from utils_image import generate_image_dataset

class KeypointImageDataset(Dataset):
    def __init__(self, keypoints_data, images_data=None, image_size=256, normalize_kp=True):
        """
        Args:
            keypoints_data: numpy array of shape [num_samples, 68, 3]
            images_data: numpy array of shape [num_samples, image_size, image_size] (optional)
                        If None, will generate images from keypoints
            image_size: Size of images
            normalize_kp: Whether to normalize keypoints
        """
        self.keypoints = torch.FloatTensor(keypoints_data)
        self.image_size = image_size
        self.normalize_kp = normalize_kp
        
        if normalize_kp:
            # Normalize to [0, 1] range per coordinate
            self.kp_min = self.keypoints.view(-1, 3).min(0)[0]
            self.kp_max = self.keypoints.view(-1, 3).max(0)[0]
            self.keypoints = (self.keypoints - self.kp_min) / (self.kp_max - self.kp_min + 1e-8)
        
        if images_data is None:
            # Generate images from keypoints
            print("Generating ground truth images from keypoints...")
            from utils_image import generate_image_dataset
            images_data = generate_image_dataset(keypoints_data, image_size=image_size)
        
        # Convert images to tensor and normalize to [-1, 1]
        self.images = torch.FloatTensor(images_data) / 255.0 * 2.0 - 1.0  # [0, 255] -> [-1, 1]
        self.images = self.images.unsqueeze(1)  # Add channel dimension [num_samples, 1, H, W]
    
    def __len__(self):
        return len(self.keypoints)
    
    def __getitem__(self, idx):
        kp = self.keypoints[idx]
        img = self.images[idx]
        return kp, img

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for keypoints, images in tqdm(dataloader, desc="Training"):
        keypoints = keypoints.to(device)
        images = images.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        generated_images = model(keypoints)
        
        # Compute loss
        loss = criterion(generated_images, images)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for keypoints, images in tqdm(dataloader, desc="Validating"):
            keypoints = keypoints.to(device)
            images = images.to(device)
            
            generated_images = model(keypoints)
            loss = criterion(generated_images, images)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train(
    data_path,
    batch_size=16,
    epochs=100,
    lr=1e-4,
    d_model=256,
    nhead=8,
    num_encoder_layers=4,
    dim_feedforward=1024,
    dropout=0.1,
    image_size=256,
    save_dir="checkpoints_image"
):
    # Load data
    print("Loading keypoints data...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    keypoints = data['keypoints']
    print(f"Loaded {len(keypoints)} samples")
    print(f"Keypoints shape: {keypoints.shape}")
    
    # Split data
    train_size = int(0.8 * len(keypoints))
    train_keypoints = keypoints[:train_size]
    val_keypoints = keypoints[train_size:]
    
    # Create datasets
    print("Creating training dataset...")
    train_dataset = KeypointImageDataset(train_keypoints, image_size=image_size, normalize_kp=True)
    
    print("Creating validation dataset...")
    val_dataset = KeypointImageDataset(val_keypoints, image_size=image_size, normalize_kp=True)
    
    # Use normalization stats from training set for validation
    val_dataset.kp_min = train_dataset.kp_min
    val_dataset.kp_max = train_dataset.kp_max
    val_dataset.keypoints = (torch.FloatTensor(val_keypoints) - val_dataset.kp_min) / (val_dataset.kp_max - val_dataset.kp_min + 1e-8)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = KeypointToImageTransformer(
        input_dim=3,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_keypoints=68,
        image_size=image_size,
        num_channels=1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training loop
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
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
    parser.add_argument('--data_path', type=str, default='keypoints_data.pkl')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_encoder_layers', type=int, default=4)
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--save_dir', type=str, default='checkpoints_image')
    
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
        save_dir=args.save_dir
    )



