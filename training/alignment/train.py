"""
Training script for Video-Language Alignment using mBART
Trains alignment between compressed video representations and text descriptions
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from tqdm import tqdm
import os
import json
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from models.alignment.model import VideoLanguageAlignment
from transformers import MBartTokenizer

class VideoTextDataset(Dataset):
    """
    Dataset for video-text pairs
    Each sample contains:
    - Video representations (global + local)
    - Text description
    """
    def __init__(
        self,
        video_reprs_path,  # Path to video representations npz file
        text_data_path,  # Path to text data (json or pickle)
        tokenizer,
        max_text_length=128
    ):
        """
        Args:
            video_reprs_path: Path to video representations file (from inference_temporal.py)
            text_data_path: Path to text data file
            tokenizer: mBART tokenizer
            max_text_length: Maximum text sequence length
        """
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        
        # Load video representations
        print(f"Loading video representations from {video_reprs_path}...")
        video_data = np.load(video_reprs_path)
        self.global_reprs = video_data['global_reprs']  # [num_samples, 512]
        self.local_reprs = video_data['local_reprs']  # [num_samples, num_local_vars, 512]
        
        print(f"Loaded {len(self.global_reprs)} video representations")
        print(f"  Global reprs shape: {self.global_reprs.shape}")
        print(f"  Local reprs shape: {self.local_reprs.shape}")
        
        # Load text data
        print(f"Loading text data from {text_data_path}...")
        if text_data_path.endswith('.json'):
            with open(text_data_path, 'r', encoding='utf-8') as f:
                text_data = json.load(f)
        elif text_data_path.endswith('.pkl'):
            with open(text_data_path, 'rb') as f:
                text_data = pickle.load(f)
        else:
            raise ValueError(f"Unknown text data format: {text_data_path}")
        
        # Handle different text data formats
        if isinstance(text_data, dict):
            if 'texts' in text_data:
                self.texts = text_data['texts']
            elif 'descriptions' in text_data:
                self.texts = text_data['descriptions']
            else:
                # Assume keys are indices
                self.texts = [text_data.get(str(i), '') for i in range(len(self.global_reprs))]
        elif isinstance(text_data, list):
            self.texts = text_data
        else:
            raise ValueError("Unknown text data format")
        
        # Ensure lengths match
        if len(self.texts) != len(self.global_reprs):
            print(f"Warning: Text data length ({len(self.texts)}) != Video reprs length ({len(self.global_reprs)})")
            min_len = min(len(self.texts), len(self.global_reprs))
            self.texts = self.texts[:min_len]
            self.global_reprs = self.global_reprs[:min_len]
            self.local_reprs = self.local_reprs[:min_len]
            print(f"Truncated to {min_len} samples")
        
        print(f"Loaded {len(self.texts)} text descriptions")
        print(f"Sample texts:")
        for i in range(min(3, len(self.texts))):
            print(f"  {i}: {self.texts[i][:50]}...")
    
    def __len__(self):
        return len(self.global_reprs)
    
    def __getitem__(self, idx):
        # Get video representations
        global_repr = torch.FloatTensor(self.global_reprs[idx])  # [512]
        local_reprs = torch.FloatTensor(self.local_reprs[idx])  # [num_local_vars, 512]
        
        # Get and tokenize text
        text = str(self.texts[idx])
        
        # Tokenize text
        encoded = self.tokenizer(
            text,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        text_ids = encoded['input_ids'].squeeze(0)  # [max_length]
        attention_mask = encoded['attention_mask'].squeeze(0)  # [max_length]
        
        # Labels for language modeling (shifted by 1)
        labels = text_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding in loss
        
        return {
            'global_repr': global_repr,
            'local_reprs': local_reprs,
            'text_ids': text_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'text': text
        }

def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        # Move to device
        global_repr = batch['global_repr'].to(device)
        local_reprs = batch['local_reprs'].to(device)
        text_ids = batch['text_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        loss, logits = model(
            global_repr=global_repr,
            local_reprs=local_reprs,
            text_ids=text_ids,
            text_attention_mask=attention_mask,
            labels=labels,
            return_loss=True
        )
        
        # Backward pass
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        else:
            print(f"Warning: NaN/Inf loss detected, skipping batch")
    
    return total_loss / max(num_batches, 1)

def validate(model, dataloader, device):
    """Validate"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            global_repr = batch['global_repr'].to(device)
            local_reprs = batch['local_reprs'].to(device)
            text_ids = batch['text_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            loss, _ = model(
                global_repr=global_repr,
                local_reprs=local_reprs,
                text_ids=text_ids,
                text_attention_mask=attention_mask,
                labels=labels,
                return_loss=True
            )
            
            if torch.isfinite(loss):
                total_loss += loss.item()
                num_batches += 1
    
    return total_loss / max(num_batches, 1)

def train(
    video_reprs_path,
    text_data_path,
    temporal_model_checkpoint=None,  # Optional: to extract fresh representations
    batch_size=4,
    epochs=20,
    lr=1e-4,
    video_repr_dim=1536,
    mbart_model_name='facebook/mbart-large-50',
    mbart_model_path=None,  # Local path to mBART model
    d_model=1024,
    dropout=0.1,
    freeze_mbart=False,
    save_dir="checkpoints_alignment",
    max_text_length=128
):
    """
    Train video-language alignment model
    
    Args:
        video_reprs_path: Path to video representations npz file
        text_data_path: Path to text data (json or pickle)
        temporal_model_checkpoint: Optional path to temporal model (if need to extract fresh reprs)
        batch_size: Batch size
        epochs: Number of training epochs
        lr: Learning rate
        video_repr_dim: Video representation dimension (global + local)
        mbart_model_name: mBART model name
        d_model: mBART embedding dimension
        dropout: Dropout rate
        freeze_mbart: Whether to freeze mBART parameters
        save_dir: Directory to save checkpoints
        max_text_length: Maximum text sequence length
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer - use local path if provided
    model_path = mbart_model_path if mbart_model_path is not None else mbart_model_name
    print(f"Loading mBART tokenizer from {model_path}...")
    tokenizer = MBartTokenizer.from_pretrained(
        model_path,
        local_files_only=True if mbart_model_path is not None else False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    dataset = VideoTextDataset(
        video_reprs_path=video_reprs_path,
        text_data_path=text_data_path,
        tokenizer=tokenizer,
        max_text_length=max_text_length
    )
    
    # Split train/val
    split_idx = int(len(dataset) * 0.8)
    train_dataset = torch.utils.data.Subset(dataset, range(split_idx))
    val_dataset = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nTrain samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
    model = VideoLanguageAlignment(
        video_repr_dim=video_repr_dim,
        mbart_model_name=mbart_model_name,
        mbart_model_path=mbart_model_path,
        d_model=d_model,
        dropout=dropout,
        freeze_mbart=freeze_mbart
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Optimizer
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate(model, val_loader, device)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'model_config': {
                    'video_repr_dim': video_repr_dim,
                    'mbart_model_name': mbart_model_name,
                    'mbart_model_path': mbart_model_path,
                    'd_model': d_model,
                    'dropout': dropout
                }
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"Saved best model (val_loss: {val_loss:.4f})")
    
    print("\nTraining completed!")

def train_from_config(config_dict):
    """Train from configuration dictionary"""
    model_config = config_dict.get('model', {})
    training_config = config_dict.get('training', {})
    
    train(
        video_reprs_path=training_config.get('video_reprs_path'),
        text_data_path=training_config.get('text_data_path'),
        batch_size=training_config.get('batch_size', 4),
        epochs=training_config.get('epochs', 20),
        lr=training_config.get('lr', 1e-4),
        video_repr_dim=model_config.get('video_repr_dim', 1536),
        mbart_model_name=model_config.get('mbart_model_name', 'facebook/mbart-large-50'),
        mbart_model_path=model_config.get('mbart_model_path'),
        d_model=model_config.get('d_model', 1024),
        dropout=model_config.get('dropout', 0.1),
        freeze_mbart=model_config.get('freeze_mbart', False),
        save_dir=training_config.get('save_dir', 'checkpoints_alignment'),
        max_text_length=training_config.get('max_text_length', 128)
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train video-language alignment model")
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (will load ALIGNMENT config)')
    parser.add_argument('--video_reprs_path', type=str, default=None,
                        help='Path to video representations npz file')
    parser.add_argument('--text_data_path', type=str, default=None,
                        help='Path to text data file (json or pickle)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--video_repr_dim', type=int, default=None,
                        help='Video representation dimension')
    parser.add_argument('--mbart_model_name', type=str, default=None,
                        help='mBART model name (if not using local path)')
    parser.add_argument('--mbart_model_path', type=str, default=None,
                        help='Local path to mBART model directory')
    parser.add_argument('--d_model', type=int, default=None,
                        help='mBART embedding dimension')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout rate')
    parser.add_argument('--freeze_mbart', action='store_true', default=None,
                        help='Freeze mBART parameters')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save checkpoints')
    parser.add_argument('--max_text_length', type=int, default=None,
                        help='Maximum text sequence length')
    
    args = parser.parse_args()
    
    # Load from config if provided
    if args.config:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", args.config)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config_dict = config_module.ALIGNMENT
        
        # Override with command line arguments if provided
        if args.video_reprs_path: config_dict['training']['video_reprs_path'] = args.video_reprs_path
        if args.text_data_path: config_dict['training']['text_data_path'] = args.text_data_path
        if args.batch_size is not None: config_dict['training']['batch_size'] = args.batch_size
        if args.epochs is not None: config_dict['training']['epochs'] = args.epochs
        if args.lr is not None: config_dict['training']['lr'] = args.lr
        if args.video_repr_dim is not None: config_dict['model']['video_repr_dim'] = args.video_repr_dim
        if args.mbart_model_name: config_dict['model']['mbart_model_name'] = args.mbart_model_name
        if args.mbart_model_path: config_dict['model']['mbart_model_path'] = args.mbart_model_path
        if args.d_model is not None: config_dict['model']['d_model'] = args.d_model
        if args.dropout is not None: config_dict['model']['dropout'] = args.dropout
        if args.freeze_mbart is not None: config_dict['model']['freeze_mbart'] = args.freeze_mbart
        if args.save_dir: config_dict['training']['save_dir'] = args.save_dir
        if args.max_text_length is not None: config_dict['training']['max_text_length'] = args.max_text_length
        
        train_from_config(config_dict)
    else:
        # Use command line arguments
        if not args.video_reprs_path or not args.text_data_path:
            parser.error("--video_reprs_path and --text_data_path are required when not using --config")
        
        train(
            video_reprs_path=args.video_reprs_path,
            text_data_path=args.text_data_path,
            batch_size=args.batch_size or 4,
            epochs=args.epochs or 20,
            lr=args.lr or 1e-4,
            video_repr_dim=args.video_repr_dim or 1536,
            mbart_model_name=args.mbart_model_name or 'facebook/mbart-large-50',
            mbart_model_path=args.mbart_model_path,
            d_model=args.d_model or 1024,
            dropout=args.dropout or 0.1,
            freeze_mbart=args.freeze_mbart if args.freeze_mbart is not None else False,
            save_dir=args.save_dir or 'checkpoints_alignment',
            max_text_length=args.max_text_length or 128
        )

