"""
Training script for keypoint representation learning
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from tqdm import tqdm
import os
from model import KeypointTransformer

class KeypointDataset(Dataset):
    def __init__(self, keypoints_data, normalize=True):
        """
        Args:
            keypoints_data: numpy array of shape [num_samples, 68, 3]
            normalize: Whether to normalize keypoints
        """
        self.keypoints = torch.FloatTensor(keypoints_data)
        self.normalize = normalize
        
        if normalize:
            # Normalize to [0, 1] range per coordinate
            self.kp_min = self.keypoints.view(-1, 3).min(0)[0]
            self.kp_max = self.keypoints.view(-1, 3).max(0)[0]
            self.keypoints = (self.keypoints - self.kp_min) / (self.kp_max - self.kp_min + 1e-8)
    
    def __len__(self):
        return len(self.keypoints)
    
    def __getitem__(self, idx):
        kp = self.keypoints[idx]
        return kp, kp  # Input and target are the same for reconstruction

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for src, tgt in tqdm(dataloader, desc="Training"):
        src = src.to(device)
        tgt = tgt.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, tgt)
        
        # Compute loss
        loss = criterion(output, tgt)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Validating"):
            src = src.to(device)
            tgt = tgt.to(device)
            
            output = model(src)
            loss = criterion(output, tgt)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train(
    data_path,
    batch_size=32,
    epochs=100,
    lr=1e-4,
    d_model=256,
    nhead=8,
    num_encoder_layers=4,
    num_decoder_layers=4,
    dim_feedforward=1024,
    dropout=0.1,
    save_dir="checkpoints"
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
    train_dataset = KeypointDataset(train_keypoints, normalize=True)
    val_dataset = KeypointDataset(val_keypoints, normalize=True)
    
    # Use normalization stats from training set for validation
    val_dataset.kp_min = train_dataset.kp_min
    val_dataset.kp_max = train_dataset.kp_max
    val_dataset.keypoints = (torch.FloatTensor(val_keypoints) - val_dataset.kp_min) / (val_dataset.kp_max - val_dataset.kp_min + 1e-8)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = KeypointTransformer(
        input_dim=3,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_keypoints=68
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
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_encoder_layers', type=int, default=4)
    parser.add_argument('--num_decoder_layers', type=int, default=4)
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    
    args = parser.parse_args()
    
    train(
        data_path=args.data_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        save_dir=args.save_dir
    )



