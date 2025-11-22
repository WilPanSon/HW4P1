import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, FeedForwardLayer

'''
TODO: Implement this Module.

This file contains the encoder layer implementation used in transformer architectures:

SelfAttentionEncoderLayer: Used in encoder part of transformers
- Contains self-attention and feed-forward sublayers
- Unlike decoder, does not use causal masking (can attend to all positions)
- Used for tasks like encoding input sequences where bidirectional context is needed

The layer follows a Pre-LN (Layer Normalization) architecture where:
- Layer normalization is applied before each sublayer operation
- Residual connections wrap around each sublayer

Implementation Steps:
1. Initialize the required sublayers in __init__:
   - SelfAttentionLayer for self-attention (no causal mask needed)
   - FeedForwardLayer for position-wise processing

2. Implement the forward pass to:
   - Apply sublayers in the correct order
   - Pass appropriate padding masks (no causal mask needed)
   - Return both outputs and attention weights
'''

class SelfAttentionEncoderLayer(nn.Module):
    '''
    Pre-LN Encoder Layer with self-attention mechanism.
    Used in the encoder part of transformer architectures.
    '''
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        '''
        Initialize the SelfAttentionEncoderLayer. 
        '''
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ffn = FeedForwardLayer(d_model, d_ff, dropout) 

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the EncoderLayer.
        '''
        
        residual = x

        x_norm = self.norm1(x)

        attn_output, mha_attn_weights = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True
        )
        
        x = residual + self.dropout1(attn_output)

        x = self.ffn(x)
        
        return x, mha_attn_weights