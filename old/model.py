"""
Transformer Encoder-Decoder model for facial keypoint representation learning
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

class KeypointTransformer(nn.Module):
    def __init__(
        self,
        input_dim=3,  # x, y, z coordinates
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        num_keypoints=68
    ):
        super(KeypointTransformer, self).__init__()
        
        self.d_model = d_model
        self.num_keypoints = num_keypoints
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=num_keypoints)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, input_dim)
        
    def forward(self, src, tgt=None):
        """
        Args:
            src: Source keypoints [batch_size, num_keypoints, 3]
            tgt: Target keypoints for training [batch_size, num_keypoints, 3]
                 If None, use src shifted by one position for autoregressive decoding
        
        Returns:
            output: Reconstructed keypoints [batch_size, num_keypoints, 3]
        """
        batch_size, num_kp, _ = src.shape
        
        # Project input to d_model
        src = self.input_projection(src)  # [batch_size, num_kp, d_model]
        
        # Add positional encoding
        src = src.transpose(0, 1)  # [num_kp, batch_size, d_model]
        src = self.pos_encoder(src)
        
        # Encode
        memory = self.encoder(src)  # [num_kp, batch_size, d_model]
        
        # Prepare decoder input
        if tgt is None:
            # For inference, use src shifted
            tgt = src
        else:
            # For training, use target keypoints
            tgt = self.input_projection(tgt)
            tgt = tgt.transpose(0, 1)
            tgt = self.pos_encoder(tgt)
        
        # Create causal mask for autoregressive decoding
        tgt_mask = self._generate_square_subsequent_mask(num_kp).to(src.device)
        
        # Decode
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)  # [num_kp, batch_size, d_model]
        
        # Project to output dimension
        output = output.transpose(0, 1)  # [batch_size, num_kp, d_model]
        output = self.output_projection(output)  # [batch_size, num_kp, 3]
        
        return output
    
    def _generate_square_subsequent_mask(self, sz):
        """Generate causal mask for autoregressive decoding"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def encode(self, src):
        """
        Encode keypoints to latent representation
        
        Args:
            src: Keypoints [batch_size, num_keypoints, 3]
        
        Returns:
            memory: Encoded representation [num_keypoints, batch_size, d_model]
        """
        batch_size, num_kp, _ = src.shape
        
        src = self.input_projection(src)
        src = src.transpose(0, 1)
        src = self.pos_encoder(src)
        
        memory = self.encoder(src)
        return memory
    
    def decode(self, memory, tgt=None):
        """
        Decode latent representation to keypoints
        
        Args:
            memory: Encoded representation [num_keypoints, batch_size, d_model]
            tgt: Target sequence for decoding (optional)
        
        Returns:
            output: Reconstructed keypoints [batch_size, num_keypoints, 3]
        """
        num_kp, batch_size, d_model = memory.shape
        
        if tgt is None:
            # Use zero-initialized target
            tgt = torch.zeros(num_kp, batch_size, d_model, device=memory.device)
        else:
            tgt = self.input_projection(tgt)
            tgt = tgt.transpose(0, 1)
            tgt = self.pos_encoder(tgt)
        
        tgt_mask = self._generate_square_subsequent_mask(num_kp).to(memory.device)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        output = output.transpose(0, 1)
        output = self.output_projection(output)
        
        return output



