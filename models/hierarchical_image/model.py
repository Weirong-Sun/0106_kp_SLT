"""
Hierarchical Transformer Encoder + CNN Decoder model for image reconstruction from keypoints
Combines hierarchical encoding (global + regional) with image generation decoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
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
    """CNN Decoder to generate images from hierarchical latent representation"""
    def __init__(self, d_global, d_region, num_regions=8, output_size=256, num_channels=1):
        super(ImageDecoder, self).__init__()
        self.output_size = output_size
        self.num_channels = num_channels
        self.d_global = d_global
        self.d_region = d_region
        self.num_regions = num_regions
        
        # Combine global and regional representations
        # Method 1: Concatenate global (broadcasted) + regional, then pool
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

class HierarchicalKeypointToImageTransformer(nn.Module):
    """
    Hierarchical Transformer Encoder + CNN Decoder for keypoint to image reconstruction
    
    Regional divisions (8 regions):
    - Region 0: Face outline (points 0-16)
    - Region 1: Right eyebrow (points 17-21)
    - Region 2: Left eyebrow (points 22-26)
    - Region 3: Nose (points 27-35)
    - Region 4: Right eye (points 36-41)
    - Region 5: Left eye (points 42-47)
    - Region 6: Mouth outer (points 48-59)
    - Region 7: Mouth inner (points 60-67)
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
        num_keypoints=68,
        num_regions=8,
        image_size=256,
        num_channels=1  # Grayscale image
    ):
        super(HierarchicalKeypointToImageTransformer, self).__init__()
        
        self.d_global = d_global
        self.d_region = d_region
        self.num_keypoints = num_keypoints
        self.num_regions = num_regions
        self.image_size = image_size
        
        # Define region indices (8 regions)
        self.region_indices = self._get_region_indices()
        
        # Input projection for each region
        self.region_projections = nn.ModuleList([
            nn.Linear(input_dim, d_region) for _ in range(num_regions)
        ])
        
        # Positional encoding for regions
        self.region_pos_enc = PositionalEncoding(d_region, max_len=20)
        
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
        """Define region indices for 68-point facial landmarks"""
        return [
            list(range(17)),      # Region 0: Face outline (0-16)
            list(range(17, 22)),  # Region 1: Right eyebrow (17-21)
            list(range(22, 27)),  # Region 2: Left eyebrow (22-26)
            list(range(27, 36)),  # Region 3: Nose (27-35)
            list(range(36, 42)),  # Region 4: Right eye (36-41)
            list(range(42, 48)),  # Region 5: Left eye (42-47)
            list(range(48, 60)),  # Region 6: Mouth outer (48-59)
            list(range(60, 68))   # Region 7: Mouth inner (60-67)
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
        Forward pass: generate image from keypoints
        
        Args:
            keypoints: Keypoint coordinates [batch_size, num_keypoints, 3]
        
        Returns:
            images: Generated images [batch_size, num_channels, image_size, image_size]
        """
        # Encode to global and regional representations
        global_repr, regional_repr = self.encode(keypoints)
        
        # Decode to image
        images = self.image_decoder(global_repr, regional_repr)
        
        return images


