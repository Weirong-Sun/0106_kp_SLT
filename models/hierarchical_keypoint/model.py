"""
Hierarchical Transformer model with global and regional representations
Global: [batch, d_global] - overall face representation
Regional: [batch, num_regions, d_region] - regional representations with cross-attention
"""
import torch
import torch.nn as nn
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

class HierarchicalKeypointTransformer(nn.Module):
    """
    Hierarchical Transformer with global and regional representations
    
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
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        num_keypoints=68,
        num_regions=8
    ):
        super(HierarchicalKeypointTransformer, self).__init__()
        
        self.d_global = d_global
        self.d_region = d_region
        self.num_keypoints = num_keypoints
        self.num_regions = num_regions
        
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
        self.global_aggregator = nn.Sequential(
            nn.Linear(d_region, d_global),
            nn.LayerNorm(d_global),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Global attention pooling
        self.global_attention = nn.MultiheadAttention(
            embed_dim=d_region,
            num_heads=nhead // 2 if nhead >= 2 else 1,
            dropout=dropout,
            batch_first=False
        )
        self.global_proj = nn.Linear(d_region, d_global)
        
        # Decoder: reconstruct keypoints from global + regional
        # Decoder input: concatenate global (broadcasted) + regional
        decoder_input_dim = d_global + d_region
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Output projection
        self.output_projection = nn.Linear(decoder_input_dim, input_dim)
        
        # Learnable query tokens for decoder (one per keypoint)
        self.decoder_queries = nn.Parameter(torch.randn(num_keypoints, decoder_input_dim))
        
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
        # Method 1: Attention pooling
        regional_seq = regional_repr_interacted  # [num_regions, batch, d_region]
        global_query = regional_seq.mean(dim=0, keepdim=True)  # [1, batch, d_region]
        global_attn, _ = self.global_attention(global_query, regional_seq, regional_seq)
        global_attn = global_attn.squeeze(0)  # [batch, d_region]
        global_repr = self.global_proj(global_attn)  # [batch, d_global]
        
        return global_repr, regional_repr
    
    def forward(self, src, tgt=None):
        """
        Forward pass: reconstruct keypoints
        
        Args:
            src: Source keypoints [batch_size, num_keypoints, 3]
            tgt: Target keypoints for training [batch_size, num_keypoints, 3]
        
        Returns:
            output: Reconstructed keypoints [batch_size, num_keypoints, 3]
        """
        batch_size = src.shape[0]
        
        # Encode to global and regional representations
        global_repr, regional_repr = self.encode(src)  # [batch, d_global], [batch, num_regions, d_region]
        
        # Prepare decoder input
        # Broadcast global to each keypoint and combine with regional info
        num_kp = self.num_keypoints
        
        # Map regional repr to keypoints (each keypoint belongs to one region)
        regional_to_kp = []
        for kp_idx in range(num_kp):
            # Find which region this keypoint belongs to
            region_idx = None
            for r_idx, region_indices in enumerate(self.region_indices):
                if kp_idx in region_indices:
                    region_idx = r_idx
                    break
            regional_to_kp.append(regional_repr[:, region_idx, :])  # [batch, d_region]
        
        regional_kp_repr = torch.stack(regional_to_kp, dim=1)  # [batch, num_kp, d_region]
        
        # Broadcast global to all keypoints
        global_kp_repr = global_repr.unsqueeze(1).expand(-1, num_kp, -1)  # [batch, num_kp, d_global]
        
        # Concatenate global + regional
        decoder_input = torch.cat([global_kp_repr, regional_kp_repr], dim=-1)  # [batch, num_kp, d_global+d_region]
        
        # Use learnable queries or decoder input
        if tgt is None:
            # Use learnable queries
            decoder_queries = self.decoder_queries.unsqueeze(1).expand(-1, batch_size, -1)  # [num_kp, batch, d_global+d_region]
        else:
            # Use target keypoints projected
            tgt_proj = torch.cat([
                global_kp_repr,
                regional_kp_repr
            ], dim=-1)
            decoder_queries = tgt_proj.transpose(0, 1)  # [num_kp, batch, d_global+d_region]
        
        # Memory: encoded representation
        memory = decoder_input.transpose(0, 1)  # [num_kp, batch, d_global+d_region]
        
        # Decode
        output = self.decoder(decoder_queries, memory)  # [num_kp, batch, d_global+d_region]
        output = output.transpose(0, 1)  # [batch, num_kp, d_global+d_region]
        
        # Project to output dimension
        output = self.output_projection(output)  # [batch, num_kp, 3]
        
        return output

