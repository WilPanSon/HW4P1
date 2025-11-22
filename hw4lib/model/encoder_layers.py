import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, FeedForwardLayer

class SelfAttentionEncoderLayer(nn.Module):
    '''
    Pre-LN Encoder Layer with self-attention mechanism.
    Used in the encoder part of transformer architectures.
    '''
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # use the HW SelfAttentionLayer (has .mha inside)
        self.self_attn = SelfAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # HW FeedForwardLayer (already Pre-LN + residual inside)
        self.ffn = FeedForwardLayer(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        x: (B, T, d_model)
        key_padding_mask: (B, T) True at PAD positions
        returns:
          x: (B, T, d_model)
          attn_weights: (B, T, T)
        '''
        residual = x
        x_norm = self.norm1(x)

        attn_out, attn_weights = self.self_attn(
            x_norm,
            key_padding_mask=key_padding_mask
        )

        # if SelfAttentionLayer returns per-head weights (B, H, T, T), average them
        if attn_weights is not None and attn_weights.dim() == 4:
            attn_weights = attn_weights.mean(dim=1)

        x = residual + self.dropout1(attn_out)

        x = self.ffn(x)

        return x, attn_weights