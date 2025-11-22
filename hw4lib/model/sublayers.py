import torch.nn as nn
import torch 
from typing import Tuple, Optional

'''
TODO: Implement these Modules.

The file contains three key sublayers used in transformer decoders:
1. SelfAttentionLayer: For masked self-attention
2. CrossAttentionLayer: For cross-attention between encoder and decoder
3. FeedForwardLayer: For position-wise feed-forward processing

Each layer follows a Pre-LN (Layer Normalization) architecture where:
- Normalization is applied before the main operation
- A residual connection wraps around the operation
'''

class SelfAttentionLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 1.
    This layer is responsible for the causally-masked self-attention mechanism.
    
    Steps to implement:
    1. Initialize the multi-head attention with proper parameters
    2. Initialize layer normalization for d_model dimensionality
    3. Initialize dropout with specified rate
    4. In forward pass:
       a. Store residual connection
       b. Apply pre-normalization
       c. Apply self-attention with masking
       d. Apply residual connection with dropout
       e. Return the output tensor and attention weights    
    ''' 
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        '''
        Initialize the SelfAttentionLayer. 
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
        '''
        super().__init__()
        # TODO: Implement __init__
        
        # TODO: Initialize the multi-head attention mechanism (use nn.MultiheadAttention)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # TODO: Initialize the normalization layer (use nn.LayerNorm)
        self.norm = nn.LayerNorm(d_model)
        
        # TODO: Initialize the dropout layer
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, key_padding_mask=None, attn_mask=None):
        """
        Forward pass for SelfAttentionLayer.
        Args:
            x: Input tensor of shape (N, L, E)
            key_padding_mask: Padding mask of shape (N, L)
            attn_mask: Causal mask of shape (L, L)
        """
        residual = x

        x_norm = self.norm(x)

        attn_output, attn_weights = self.mha(
            query=x_norm, 
            key=x_norm, 
            value=x_norm, 
            attn_mask=attn_mask, 
            key_padding_mask=key_padding_mask
        )
        
        output = residual + self.dropout(attn_output)
        
        # e. Return the output tensor and attention weights
        return output, attn_weights



    
## -------------------------------------------------------------------------------------------------  
class CrossAttentionLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 2.
    This layer is responsible for the cross-attention mechanism between encoder and decoder.
    '''     
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        '''
        Initialize the CrossAttentionLayer. 
        '''
        super().__init__()
        
        # Initialize the multi-head attention mechanism
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # Initialize the normalization layer (Pre-LN)
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize the dropout layer
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x: torch.Tensor, y: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the CrossAttentionLayer.
        '''
        # 1. Store residual connection
        residual = x
        
        # 2. Apply pre-normalization (Pre-LN)
        x_norm = self.norm(x)
        
        # 3. Apply cross-attention
        # FIX: Set average_attn_weights=True to match expected shape (Batch, Seq, Seq)
        attn_output, mha_attn_weights = self.mha(
            query=x_norm, 
            key=y, 
            value=y, 
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=True # <--- CHANGED TO TRUE
        )
        
        # 4. Apply residual connection with dropout
        x = residual + self.dropout(attn_output)
        
        # 5. Return the output tensor and attention weights
        return x, mha_attn_weights

## -------------------------------------------------------------------------------------------------  
class FeedForwardLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 3.
    This layer is responsible for the position-wise feed-forward network.
    '''
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        '''
        Initialize the FeedForwardLayer. 
        '''
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the FeedForwardLayer.
        ''' 
        residual = x
        
        # Pre-LN: Normalize before the block
        x_norm = self.norm(x)
        
        ff_output = self.ffn(x_norm)

        # Apply dropout to FFN output and add residual
        x = residual + self.dropout(ff_output)
        
        return x