"""
Video-Language Alignment Model using mBART
Maps compressed video representations to mBART decoder for language generation
"""
import torch
import torch.nn as nn
from transformers import MBartForConditionalGeneration, MBartTokenizer
import math

class VideoLanguageAlignment(nn.Module):
    """
    Alignment model that maps video representations to mBART decoder
    
    Architecture:
    1. Video encoder projection: Maps compressed video representations to mBART embedding space
    2. mBART decoder: Generates text from video representations
    """
    def __init__(
        self,
        video_repr_dim=1536,  # global (512) + local (2*512)
        mbart_model_name='facebook/mbart-large-50',
        mbart_model_path=None,  # Local path to mBART model
        d_model=1024,  # mBART embedding dimension
        dropout=0.1,
        freeze_mbart=False
    ):
        super(VideoLanguageAlignment, self).__init__()
        
        self.video_repr_dim = video_repr_dim
        self.d_model = d_model
        
        # Load mBART model and tokenizer
        # Use local path if provided, otherwise use model name
        model_path = mbart_model_path if mbart_model_path is not None else mbart_model_name
        print(f"Loading mBART model from: {model_path}")
        
        self.mbart = MBartForConditionalGeneration.from_pretrained(model_path, local_files_only=True if mbart_model_path else False)
        self.tokenizer = MBartTokenizer.from_pretrained(model_path, local_files_only=True if mbart_model_path else False)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get decoder start token ID (for mBART, usually the target language token)
        # For mBART-large-cc25, we can use 'en_XX' as default
        self.decoder_start_token_id = self.mbart.config.decoder_start_token_id
        if self.decoder_start_token_id is None:
            # Try to get from tokenizer
            if hasattr(self.tokenizer, 'lang_code_to_id'):
                self.decoder_start_token_id = self.tokenizer.lang_code_to_id.get('en_XX', self.tokenizer.eos_token_id)
            else:
                self.decoder_start_token_id = self.tokenizer.eos_token_id
        
        # Freeze mBART if specified
        if freeze_mbart:
            for param in self.mbart.parameters():
                param.requires_grad = False
        
        # Video representation projection to mBART embedding space
        # Input: [batch, video_repr_dim] (global + flattened local)
        # Output: [batch, d_model] (mBART embedding dimension)
        self.video_projection = nn.Sequential(
            nn.Linear(video_repr_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Cross-modal attention: Allow video to attend to text and vice versa
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=16,
            dropout=dropout,
            batch_first=True
        )
        
        # Optional: Learnable position encoding for video representations
        self.video_pos_encoding = nn.Parameter(torch.randn(1, 1, d_model))
    
    def encode_video(self, global_repr, local_reprs):
        """
        Encode video representations to mBART embedding space
        
        Args:
            global_repr: Global representation [batch, 512]
            local_reprs: Local representations [batch, num_local_vars, 512]
        
        Returns:
            video_embeddings: [batch, d_model] or [batch, num_local_vars+1, d_model]
        """
        batch_size = global_repr.shape[0]
        num_local_vars = local_reprs.shape[1]
        
        # Flatten local representations: [batch, num_local_vars * 512]
        local_flat = local_reprs.reshape(batch_size, -1)  # [batch, num_local_vars * 512]
        
        # Concatenate global and local: [batch, video_repr_dim]
        combined_repr = torch.cat([global_repr, local_flat], dim=-1)  # [batch, 1536]
        
        # Project to mBART embedding space
        video_embedding = self.video_projection(combined_repr)  # [batch, d_model]
        
        # Add position encoding and expand for sequence (if needed)
        video_embedding = video_embedding.unsqueeze(1)  # [batch, 1, d_model]
        video_embedding = video_embedding + self.video_pos_encoding
        
        return video_embedding  # [batch, 1, d_model]
    
    def forward(
        self,
        global_repr,
        local_reprs,
        text_ids=None,
        text_attention_mask=None,
        labels=None,
        return_loss=True
    ):
        """
        Forward pass: Generate text from video representations
        
        Args:
            global_repr: Global video representation [batch, 512]
            local_reprs: Local video representations [batch, num_local_vars, 512]
            text_ids: Text token ids [batch, seq_len] (for training)
            text_attention_mask: Text attention mask [batch, seq_len]
            labels: Labels for language modeling [batch, seq_len]
            return_loss: Whether to compute and return loss
        
        Returns:
            If return_loss=True:
                loss: Language modeling loss
                logits: Prediction logits [batch, seq_len, vocab_size]
            If return_loss=False:
                generated_ids: Generated text token ids [batch, max_length]
        """
        # Encode video to mBART embedding space
        video_embedding = self.encode_video(global_repr, local_reprs)  # [batch, 1, d_model]
        
        if return_loss and text_ids is not None:
            # Training mode: Use mBART with video as encoder output
            # video_embedding: [batch, 1, d_model]
            
            # Expand video embedding to match expected encoder output format
            # mBART expects encoder_hidden_states: [batch, seq_len, d_model]
            encoder_hidden_states = video_embedding  # [batch, 1, d_model]
            encoder_attention_mask = torch.ones(
                encoder_hidden_states.shape[:2], 
                dtype=torch.long,
                device=encoder_hidden_states.device
            )
            
            # Use mBART model with custom encoder outputs
            # Shift text_ids for decoder input (teacher forcing)
            decoder_input_ids = text_ids[:, :-1].contiguous()  # Remove last token
            decoder_attention_mask = text_attention_mask[:, :-1].contiguous() if text_attention_mask is not None else None
            
            # Forward through decoder
            decoder_outputs = self.mbart.model.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask
            )
            
            # Get logits
            logits = self.mbart.lm_head(decoder_outputs.last_hidden_state)
            
            # Compute loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Labels are already shifted (labels[:, 1:] corresponds to decoder_input_ids)
            shift_labels = labels[:, 1:].contiguous()
            loss = loss_fct(logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            
            return loss, logits
        
        else:
            # Inference mode: Generate text from video
            # Use video embedding as encoder output
            encoder_hidden_states = video_embedding  # [batch, 1, d_model]
            
            # mBART.generate with custom encoder outputs
            # We need to pass encoder_outputs as a tuple of BaseModelOutput
            from transformers.modeling_outputs import BaseModelOutput
            
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_hidden_states
            )
            
            generated_ids = self.mbart.generate(
                encoder_outputs=encoder_outputs,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            return generated_ids
    
    def generate(self, global_repr, local_reprs, max_length=128, num_beams=4):
        """
        Generate text from video representations
        
        Args:
            global_repr: Global video representation [batch, 512]
            local_reprs: Local video representations [batch, num_local_vars, 512]
            max_length: Maximum generation length
            num_beams: Beam search width
        
        Returns:
            generated_texts: List of generated text strings
        """
        self.eval()
        with torch.no_grad():
            # Encode video
            video_embedding = self.encode_video(global_repr, local_reprs)  # [batch, 1, d_model]
            
            # Create encoder outputs
            from transformers.modeling_outputs import BaseModelOutput
            
            encoder_outputs = BaseModelOutput(
                last_hidden_state=video_embedding
            )
            
            # Generate with decoder_start_token_id
            # Use the stored decoder_start_token_id
            generated_ids = self.mbart.generate(
                encoder_outputs=encoder_outputs,
                decoder_start_token_id=self.decoder_start_token_id,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,  # Prevent repetition
                do_sample=False,  # Use greedy/beam search
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode to text
            generated_texts = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            return generated_texts

