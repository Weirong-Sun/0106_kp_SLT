"""
Transformer Encoder + CNN Decoder model for skeleton reconstruction from body keypoints
Reconstructs skeleton images from keypoint coordinates
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class ImageDecoder(nn.Module):
    """CNN Decoder to generate skeleton images from latent representation"""
    def __init__(self, d_model, output_size=256, num_channels=1):
        super(ImageDecoder, self).__init__()
        self.output_size = output_size
        self.num_channels = num_channels
        
        # Project latent to initial feature map
        self.latent_proj = nn.Linear(d_model, 8 * 8 * d_model)
        
        # Upsampling layers
        self.decoder = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(d_model, d_model // 2, 4, 2, 1),
            nn.BatchNorm2d(d_model // 2),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(d_model // 2, d_model // 4, 4, 2, 1),
            nn.BatchNorm2d(d_model // 4),
            nn.ReLU(True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(d_model // 4, d_model // 8, 4, 2, 1),
            nn.BatchNorm2d(d_model // 8),
            nn.ReLU(True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(d_model // 8, d_model // 16, 4, 2, 1),
            nn.BatchNorm2d(d_model // 16),
            nn.ReLU(True),
            
            # 128x128 -> 256x256
            nn.ConvTranspose2d(d_model // 16, num_channels, 4, 2, 1),
            nn.Tanh()  # Output in [-1, 1] range
        )
    
    def forward(self, latent):
        """
        Args:
            latent: [batch_size, d_model] or [batch_size, num_keypoints, d_model]
        
        Returns:
            images: [batch_size, num_channels, output_size, output_size]
        """
        if len(latent.shape) == 3:
            # If input is [batch_size, num_keypoints, d_model], pool to [batch_size, d_model]
            latent = latent.mean(dim=1)  # Average pooling
        
        batch_size = latent.shape[0]
        
        # Project to feature map
        x = self.latent_proj(latent)  # [batch_size, 8*8*d_model]
        x = x.view(batch_size, -1, 8, 8)  # [batch_size, d_model, 8, 8]
        
        # Decode to image
        images = self.decoder(x)
        
        return images

class SkeletonReconstructionTransformer(nn.Module):
    """
    Transformer Encoder + CNN Decoder for skeleton reconstruction from body keypoints
    
    Input: Flattened keypoints [batch_size, total_keypoints, 3]
           where total_keypoints = 68 (face) + 21 (left_hand) + 21 (right_hand) + 33 (pose) = 143
    Output: Skeleton images [batch_size, 1, image_size, image_size]
    """
    def __init__(
        self,
        input_dim=3,  # x, y, z coordinates
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        num_keypoints=143,  # 68 + 21 + 21 + 33
        image_size=256,
        num_channels=1  # Grayscale image
    ):
        super(SkeletonReconstructionTransformer, self).__init__()
        
        self.d_model = d_model
        self.num_keypoints = num_keypoints
        self.image_size = image_size
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=num_keypoints)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Image Decoder
        self.image_decoder = ImageDecoder(d_model, output_size=image_size, num_channels=num_channels)
        
    def forward(self, keypoints):
        """
        Forward pass: generate skeleton image from keypoints
        
        Args:
            keypoints: Keypoint coordinates [batch_size, num_keypoints, 3]
        
        Returns:
            images: Generated skeleton images [batch_size, num_channels, image_size, image_size]
        """
        batch_size, num_kp, _ = keypoints.shape
        
        # Project input to d_model
        src = self.input_projection(keypoints)  # [batch_size, num_kp, d_model]
        
        # Add positional encoding
        src = src.transpose(0, 1)  # [num_kp, batch_size, d_model]
        src = self.pos_encoder(src)
        
        # Encode
        memory = self.encoder(src)  # [num_kp, batch_size, d_model]
        
        # Pool encoded representation (average pooling)
        latent = memory.transpose(0, 1).mean(dim=1)  # [batch_size, d_model]
        
        # Decode to image
        images = self.image_decoder(latent)
        
        return images
    
    def encode(self, keypoints):
        """
        Encode keypoints to latent representation
        
        Args:
            keypoints: Keypoints [batch_size, num_keypoints, 3]
        
        Returns:
            latent: Encoded representation [batch_size, d_model]
        """
        batch_size, num_kp, _ = keypoints.shape
        
        src = self.input_projection(keypoints)
        src = src.transpose(0, 1)
        src = self.pos_encoder(src)
        
        memory = self.encoder(src)
        latent = memory.transpose(0, 1).mean(dim=1)
        
        return latent

