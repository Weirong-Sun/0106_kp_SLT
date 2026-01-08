"""
Hierarchical Transformer Encoder + CNN Decoder model for skeleton reconstruction from body keypoints
Uses hierarchical encoding with global and regional representations
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
    """CNN Decoder to generate skeleton images from hierarchical latent representation"""
    def __init__(self, d_global, d_region, num_regions=4, output_size=256, num_channels=1):
        super(ImageDecoder, self).__init__()
        self.output_size = output_size
        self.num_channels = num_channels
        self.d_global = d_global
        self.d_region = d_region
        self.num_regions = num_regions
        
        # Combine global and regional representations
        combined_dim = d_global + d_region
        
        # Project combined representation to initial feature map
        self.latent_proj = nn.Linear(combined_dim, 8 * 8 * d_global)
        
        # Upsampling layers
        self.decoder = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(d_global, d_global // 2, 4, 2, 1),
            nn.BatchNorm2d(d_global // 2),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(d_global // 2, d_global // 4, 4, 2, 1),
            nn.BatchNorm2d(d_global // 4),
            nn.ReLU(True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(d_global // 4, d_global // 8, 4, 2, 1),
            nn.BatchNorm2d(d_global // 8),
            nn.ReLU(True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(d_global // 8, d_global // 16, 4, 2, 1),
            nn.BatchNorm2d(d_global // 16),
            nn.ReLU(True),
            
            # 128x128 -> 256x256
            nn.ConvTranspose2d(d_global // 16, num_channels, 4, 2, 1),
            nn.Tanh()  # Output in [-1, 1] range
        )
    
    def forward(self, global_repr, regional_repr):
        """
        Args:
            global_repr: [batch_size, d_global]
            regional_repr: [batch_size, num_regions, d_region]
        
        Returns:
            images: [batch_size, num_channels, output_size, output_size]
        """
        batch_size = global_repr.shape[0]
        
        # Combine global and regional representations
        # Broadcast global to each region and concatenate
        global_expanded = global_repr.unsqueeze(1).expand(-1, self.num_regions, -1)  # [batch, num_regions, d_global]
        combined = torch.cat([global_expanded, regional_repr], dim=-1)  # [batch, num_regions, d_global+d_region]
        
        # Pool regional representations (average pooling)
        pooled = combined.mean(dim=1)  # [batch, d_global+d_region]
        
        # Project to feature map
        x = self.latent_proj(pooled)  # [batch, 8*8*d_global]
        x = x.view(batch_size, -1, 8, 8)  # [batch, d_global, 8, 8]
        
        # Decode to image
        images = self.decoder(x)
        
        return images

class HierarchicalSkeletonTransformer(nn.Module):
    """
    Hierarchical Transformer Encoder + CNN Decoder for skeleton reconstruction from body keypoints
    
    Regional divisions (4 regions):
    - Region 0: Face (68 points, indices 0-67)
    - Region 1: Left hand (21 points, indices 68-88)
    - Region 2: Right hand (21 points, indices 89-109)
    - Region 3: Pose (33 points, indices 110-142)
    
    Total: 68 + 21 + 21 + 33 = 143 keypoints
    """
    def __init__(
        self,
        input_dim=3,  # x, y, z coordinates
        d_global=256,
        d_region=128,
        nhead=8,
        num_region_layers=2,  # Layers for regional encoding
        num_interaction_layers=2,  # Layers for cross-region interaction
        dim_feedforward=1024,
        dropout=0.1,
        num_keypoints=143,  # 68 + 21 + 21 + 33
        num_regions=4,  # Face, left_hand, right_hand, pose
        image_size=256,
        num_channels=1  # Grayscale image
    ):
        super(HierarchicalSkeletonTransformer, self).__init__()
        
        self.d_global = d_global
        self.d_region = d_region
        self.num_keypoints = num_keypoints
        self.num_regions = num_regions
        self.image_size = image_size
        
        # Define region indices (4 regions)
        self.region_indices = self._get_region_indices()
        
        # Input projection for each region
        self.region_projections = nn.ModuleList([
            nn.Linear(input_dim, d_region) for _ in range(num_regions)
        ])
        
        # Positional encoding for regions
        self.region_pos_enc = PositionalEncoding(d_region, max_len=70)  # Max region size is 68 (face)
        
        # Regional encoders (one for each region)
        self.region_encoders = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_region,
                    nhead=nhead // 2 if nhead >= 2 else 1,
                    dim_feedforward=dim_feedforward // 2,
                    dropout=dropout,
                    batch_first=False
                ),
                num_layers=num_region_layers
            ) for _ in range(num_regions)
        ])
        
        # Cross-region interaction layers (self-attention across regions)
        interaction_layer = nn.TransformerEncoderLayer(
            d_model=d_region,
            nhead=nhead // 2 if nhead >= 2 else 1,
            dim_feedforward=dim_feedforward // 2,
            dropout=dropout,
            batch_first=False
        )
        self.region_interaction = nn.TransformerEncoder(
            interaction_layer,
            num_layers=num_interaction_layers
        )
        
        # Global aggregation (from regional to global)
        # Global attention pooling
        self.global_attention = nn.MultiheadAttention(
            embed_dim=d_region,
            num_heads=nhead // 2 if nhead >= 2 else 1,
            dropout=dropout,
            batch_first=False
        )
        self.global_proj = nn.Linear(d_region, d_global)
        
        # Image Decoder
        self.image_decoder = ImageDecoder(
            d_global=d_global,
            d_region=d_region,
            num_regions=num_regions,
            output_size=image_size,
            num_channels=num_channels
        )
    
    def _get_region_indices(self):
        """Define region indices for 143-point body keypoints"""
        return [
            list(range(68)),        # Region 0: Face (0-67)
            list(range(68, 89)),    # Region 1: Left hand (68-88)
            list(range(89, 110)),   # Region 2: Right hand (89-109)
            list(range(110, 143))   # Region 3: Pose (110-142)
        ]
    
    def _group_keypoints_by_region(self, keypoints):
        """
        Group keypoints by region
        
        Args:
            keypoints: [batch_size, num_keypoints, 3]
        
        Returns:
            region_keypoints: List of [batch_size, region_size, 3]
        """
        region_keypoints = []
        for region_idx in self.region_indices:
            region_kp = keypoints[:, region_idx, :]
            region_keypoints.append(region_kp)
        return region_keypoints
    
    def encode(self, keypoints):
        """
        Encode keypoints to global and regional representations
        
        Args:
            keypoints: [batch_size, num_keypoints, 3]
        
        Returns:
            global_repr: [batch_size, d_global]
            regional_repr: [batch_size, num_regions, d_region]
        """
        batch_size = keypoints.shape[0]
        
        # Group keypoints by region
        region_keypoints = self._group_keypoints_by_region(keypoints)
        
        # Encode each region independently
        region_embeddings = []
        for i, (region_kp, proj, encoder) in enumerate(zip(
            region_keypoints, self.region_projections, self.region_encoders
        )):
            # Project to d_region
            region_emb = proj(region_kp)  # [batch, region_size, d_region]
            
            # Add positional encoding within region
            region_size = region_emb.shape[1]
            region_emb = region_emb.transpose(0, 1)  # [region_size, batch, d_region]
            region_emb = self.region_pos_enc(region_emb)
            
            # Encode region
            region_encoded = encoder(region_emb)  # [region_size, batch, d_region]
            
            # Pool region (average pooling)
            region_pooled = region_encoded.mean(dim=0)  # [batch, d_region]
            region_embeddings.append(region_pooled)
        
        # Stack regional representations
        regional_repr = torch.stack(region_embeddings, dim=1)  # [batch, num_regions, d_region]
        
        # Cross-region interaction
        regional_repr_seq = regional_repr.transpose(0, 1)  # [num_regions, batch, d_region]
        regional_repr_interacted = self.region_interaction(regional_repr_seq)  # [num_regions, batch, d_region]
        regional_repr = regional_repr_interacted.transpose(0, 1)  # [batch, num_regions, d_region]
        
        # Generate global representation
        # Method: Attention pooling
        regional_seq = regional_repr_interacted  # [num_regions, batch, d_region]
        global_query = regional_seq.mean(dim=0, keepdim=True)  # [1, batch, d_region]
        global_attn, _ = self.global_attention(global_query, regional_seq, regional_seq)
        global_attn = global_attn.squeeze(0)  # [batch, d_region]
        global_repr = self.global_proj(global_attn)  # [batch, d_global]
        
        return global_repr, regional_repr
    
    def forward(self, keypoints):
        """
        Forward pass: generate skeleton image from keypoints
        
        Args:
            keypoints: Keypoint coordinates [batch_size, num_keypoints, 3]
        
        Returns:
            images: Generated skeleton images [batch_size, num_channels, image_size, image_size]
        """
        # Encode to global and regional representations
        global_repr, regional_repr = self.encode(keypoints)
        
        # Decode to image
        images = self.image_decoder(global_repr, regional_repr)
        
        return images

