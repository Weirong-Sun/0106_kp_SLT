"""
Temporal Transformer model for video sequence representation
Uses pre-trained hierarchical skeleton encoder + temporal transformer
"""
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
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

class TemporalSkeletonTransformer(nn.Module):
    """
    Temporal Transformer for video sequence representation
    
    Architecture:
    1. Frame-level encoder (hierarchical skeleton model) - can be frozen or fine-tuned
    2. Temporal fusion layer - combines global + regional representations
    3. Temporal Transformer Encoder - learns temporal dependencies
    4. Final representation pooling
    
    Input: Video sequence of keypoints [batch, seq_len, num_keypoints, 3]
    Output: Final temporal representation [batch, d_final]
    """
    def __init__(
        self,
        frame_encoder,  # Pre-trained hierarchical skeleton encoder
        d_global=256,
        d_region=128,
        num_regions=4,
        d_temporal=512,  # Dimension for temporal sequence
        d_final=512,  # Final representation dimension
        nhead=8,
        num_temporal_layers=4,
        num_decoder_layers=None,  # If None, use same as encoder
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=300,
        freeze_frame_encoder=True,  # Whether to freeze frame encoder
        fusion_method='concat'  # 'concat' or 'weighted'
    ):
        super(TemporalSkeletonTransformer, self).__init__()
        
        self.frame_encoder = frame_encoder
        self.d_global = d_global
        self.d_region = d_region
        self.num_regions = num_regions
        self.d_temporal = d_temporal
        self.d_final = d_final
        self.freeze_frame_encoder = freeze_frame_encoder
        self.fusion_method = fusion_method
        
        # Freeze frame encoder if specified
        if freeze_frame_encoder:
            for param in self.frame_encoder.parameters():
                param.requires_grad = False
        
        # Temporal fusion: combine global + regional representations
        if fusion_method == 'concat':
            # Concatenate global + flattened regional
            fusion_input_dim = d_global + num_regions * d_region
        elif fusion_method == 'weighted':
            # Weighted combination
            fusion_input_dim = d_global
            self.region_weight = nn.Parameter(torch.ones(num_regions) / num_regions)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, d_temporal),
            nn.LayerNorm(d_temporal),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Temporal positional encoding
        self.temporal_pos_enc = PositionalEncoding(d_temporal, max_len=max_seq_len)
        
        # Temporal Transformer Encoder
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_temporal,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=num_temporal_layers
        )
        
        # Multi-scale compressed representations
        # Global variable: attends to all frames
        self.global_query = nn.Parameter(torch.randn(1, 1, d_temporal))
        self.global_attention = nn.MultiheadAttention(
            embed_dim=d_temporal,
            num_heads=nhead,
            dropout=dropout,
            batch_first=False
        )
        self.global_proj = nn.Linear(d_temporal, d_final)
        
        # Local variables: attend to different temporal windows
        # Define temporal window patterns (e.g., [1,3,5], [2,4,6], etc.)
        # We'll use stride-based sampling: stride=2, offset=0 for [1,3,5], offset=1 for [2,4,6]
        self.num_local_vars = 2  # Number of local variables
        self.local_queries = nn.Parameter(torch.randn(self.num_local_vars, 1, d_temporal))
        self.local_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_temporal,
                num_heads=nhead,
                dropout=dropout,
                batch_first=False
            ) for _ in range(self.num_local_vars)
        ])
        self.local_projs = nn.ModuleList([
            nn.Linear(d_temporal, d_final) for _ in range(self.num_local_vars)
        ])
        
        # Store temporal window patterns
        # Each pattern is (stride, offset)
        # Pattern 0: stride=2, offset=0 -> frames [0, 2, 4, ...] (1-indexed: 1, 3, 5, ...)
        # Pattern 1: stride=2, offset=1 -> frames [1, 3, 5, ...] (1-indexed: 2, 4, 6, ...)
        self.temporal_patterns = [(2, 0), (2, 1)]  # Can be extended
        
        # Decoder: reconstruct keypoint sequences from temporal representation
        # Decoder input: final representation + positional encoding
        self.decoder_proj = nn.Linear(d_final, d_temporal)
        
        # Temporal Transformer Decoder
        num_dec_layers = num_decoder_layers if num_decoder_layers is not None else num_temporal_layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_temporal,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.temporal_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_dec_layers
        )
        
        # Output projection: decode back to frame representations
        # We decode to the same dimension as frame encoder output
        # Then use frame encoder's decoder to reconstruct keypoints
        self.output_proj = nn.Linear(d_temporal, d_global + num_regions * d_region)
        
        # Frame decoder: reconstruct keypoints from frame representations
        # This will use the frame encoder's structure
        self.frame_decoder_proj = nn.Sequential(
            nn.Linear(d_global + num_regions * d_region, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 143 * 3)  # 143 keypoints * 3 coordinates
        )
        
    def encode_frames(self, keypoints_sequence):
        """
        Encode each frame in the sequence using hierarchical encoder (pre-trained frame encoder)
        
        Args:
            keypoints_sequence: [batch, seq_len, num_keypoints, 3]
        
        Returns:
            global_reprs: [batch, seq_len, d_global]
            regional_reprs: [batch, seq_len, num_regions, d_region]
        """
        batch_size, seq_len, num_kp, _ = keypoints_sequence.shape
        
        # Reshape to process all frames at once
        keypoints_flat = keypoints_sequence.reshape(-1, num_kp, 3)  # [batch*seq_len, num_kp, 3]
        
        # Encode all frames using pre-trained hierarchical frame encoder
        # This is where the pre-trained model generates representations for each frame
        global_reprs, regional_reprs = self.frame_encoder.encode(keypoints_flat)
        # Output: global_reprs [batch*seq_len, d_global], regional_reprs [batch*seq_len, num_regions, d_region]
        
        # Reshape back to sequence format
        global_reprs = global_reprs.reshape(batch_size, seq_len, self.d_global)
        regional_reprs = regional_reprs.reshape(batch_size, seq_len, self.num_regions, self.d_region)
        
        return global_reprs, regional_reprs
    
    def fuse_representations(self, global_reprs, regional_reprs):
        """
        Fuse global and regional representations
        
        Args:
            global_reprs: [batch, seq_len, d_global]
            regional_reprs: [batch, seq_len, num_regions, d_region]
        
        Returns:
            fused: [batch, seq_len, d_temporal]
        """
        if self.fusion_method == 'concat':
            # Flatten regional representations
            regional_flat = regional_reprs.reshape(regional_reprs.shape[0], regional_reprs.shape[1], -1)
            # Concatenate
            combined = torch.cat([global_reprs, regional_flat], dim=-1)
        elif self.fusion_method == 'weighted':
            # Weighted sum of regional representations
            weights = torch.softmax(self.region_weight, dim=0)
            regional_weighted = (regional_reprs * weights.view(1, 1, -1, 1)).sum(dim=2)
            # Combine with global
            combined = global_reprs + regional_weighted
        
        # Project to temporal dimension
        fused = self.fusion(combined)
        
        return fused
    
    def decode(self, global_repr, seq_len, mask=None):
        """
        Decode global representation back to keypoint sequence
        
        Args:
            global_repr: Global representation [batch, d_final]
            seq_len: Target sequence length
            mask: Optional mask [batch, seq_len]
        
        Returns:
            reconstructed_keypoints: [batch, seq_len, num_keypoints, 3]
        """
        batch_size = global_repr.shape[0]
        
        # Project global representation to temporal dimension
        decoder_input = self.decoder_proj(global_repr)  # [batch, d_temporal]
        decoder_input = decoder_input.unsqueeze(0)  # [1, batch, d_temporal]
        
        # Create decoder queries with positional encoding
        decoder_queries = torch.zeros(seq_len, batch_size, self.d_temporal, 
                                     device=global_repr.device)
        decoder_queries = self.temporal_pos_enc(decoder_queries)
        
        # Expand decoder input to match sequence length (memory)
        memory = decoder_input.expand(seq_len, -1, -1)  # [seq_len, batch, d_temporal]
        
        # Create padding mask if provided
        if mask is not None:
            padding_mask = ~mask
        else:
            padding_mask = None
        
        # Decode
        decoded = self.temporal_decoder(
            decoder_queries, 
            memory,
            tgt_key_padding_mask=padding_mask
        )  # [seq_len, batch, d_temporal]
        
        # Project to frame representation dimension
        decoded = decoded.transpose(0, 1)  # [batch, seq_len, d_temporal]
        frame_reprs = self.output_proj(decoded)  # [batch, seq_len, d_global + num_regions*d_region]
        
        # Decode to keypoints
        keypoints_flat = self.frame_decoder_proj(frame_reprs)  # [batch, seq_len, 143*3]
        reconstructed_keypoints = keypoints_flat.reshape(batch_size, seq_len, 143, 3)
        
        return reconstructed_keypoints
    
    def forward(self, keypoints_sequence, mask=None, return_reconstruction=False):
        """
        Forward pass: encode and optionally decode
        
        Args:
            keypoints_sequence: Keypoint sequences [batch, seq_len, num_keypoints, 3]
            mask: Optional mask for padded sequences [batch, seq_len]
            return_reconstruction: If True, also return reconstructed keypoints
        
        Returns:
            final_repr: Final temporal representation [batch, d_final]
            temporal_reprs: Temporal sequence representations [batch, seq_len, d_temporal]
            reconstructed_keypoints: (optional) Reconstructed keypoints [batch, seq_len, num_keypoints, 3]
        """
        batch_size, seq_len = keypoints_sequence.shape[:2]
        
        # Encode each frame
        global_reprs, regional_reprs = self.encode_frames(keypoints_sequence)
        
        # Fuse global + regional representations
        temporal_input = self.fuse_representations(global_reprs, regional_reprs)  # [batch, seq_len, d_temporal]
        
        # Add temporal positional encoding
        temporal_input = temporal_input.transpose(0, 1)  # [seq_len, batch, d_temporal]
        temporal_input = self.temporal_pos_enc(temporal_input)
        
        # Create padding mask if provided
        if mask is not None:
            padding_mask = ~mask  # [batch, seq_len]
        else:
            padding_mask = None
        
        # Temporal encoding
        temporal_encoded = self.temporal_encoder(temporal_input, src_key_padding_mask=padding_mask)
        # temporal_encoded: [seq_len, batch, d_temporal]
        
        # Global variable: attend to all frames
        global_query = self.global_query.expand(1, batch_size, -1)  # [1, batch, d_temporal]
        global_attn, _ = self.global_attention(global_query, temporal_encoded, temporal_encoded)
        global_attn = global_attn.squeeze(0)  # [batch, d_temporal]
        global_repr = self.global_proj(global_attn)  # [batch, d_final]
        
        # Local variables: attend to different temporal windows
        local_reprs = []
        for i, (stride, offset) in enumerate(self.temporal_patterns):
            # Select frames based on pattern: frames at indices [offset, offset+stride, offset+2*stride, ...]
            pattern_indices = torch.arange(offset, seq_len, stride, device=temporal_encoded.device)
            if len(pattern_indices) > 0:
                # Extract frames matching the pattern
                pattern_frames = temporal_encoded[pattern_indices]  # [num_pattern_frames, batch, d_temporal]
                
                # Attention pooling over pattern frames
                local_query = self.local_queries[i].expand(1, batch_size, -1)  # [1, batch, d_temporal]
                local_attn, _ = self.local_attentions[i](local_query, pattern_frames, pattern_frames)
                local_attn = local_attn.squeeze(0)  # [batch, d_temporal]
                local_repr = self.local_projs[i](local_attn)  # [batch, d_final]
                local_reprs.append(local_repr)
            else:
                # If pattern has no frames, use zero representation
                local_repr = torch.zeros(batch_size, self.d_final, device=temporal_encoded.device)
                local_reprs.append(local_repr)
        
        # Stack local representations: [batch, num_local_vars, d_final]
        local_reprs = torch.stack(local_reprs, dim=1)  # [batch, num_local_vars, d_final]
        
        # Also return temporal sequence representations for reconstruction
        temporal_reprs = temporal_encoded.transpose(0, 1)  # [batch, seq_len, d_temporal]
        
        if return_reconstruction:
            # For reconstruction, use global representation
            reconstructed_keypoints = self.decode(global_repr, seq_len, mask)
            return global_repr, local_reprs, temporal_reprs, reconstructed_keypoints
        
        return global_repr, local_reprs, temporal_reprs
    
    def encode(self, keypoints_sequence, mask=None):
        """
        Encode video sequence to compressed representations
        
        Args:
            keypoints_sequence: [batch, seq_len, num_keypoints, 3]
            mask: Optional mask [batch, seq_len]
        
        Returns:
            global_repr: Global representation [batch, d_final]
            local_reprs: Local representations [batch, num_local_vars, d_final]
        """
        global_repr, local_reprs, _ = self.forward(keypoints_sequence, mask)
        return global_repr, local_reprs

