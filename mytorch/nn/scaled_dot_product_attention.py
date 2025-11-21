import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # Initialize your softmax layer
        # What dimension should you pass to the softmax constructor?
        self.eps = 1e10 # DO NOT MODIFY
        self.softmax = Softmax(-1)
        
    
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape where 1/True indicates a position to ignore
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        # TODO: Implement forward pass
        self.Q = Q
        self.K = K
        self.V = V
        # Calculate attention scores: (N, ..., H, L, S)
        # (N, ..., H, L, E) @ (N, ..., H, E, S) -> (N, ..., H, L, S)
        d_k = K.shape[-1]
        K_transposed = np.swapaxes(K, -2, -1)
        scaled_dot_product = (Q @ K_transposed) / np.sqrt(d_k)
        # Apply mask before softmax if provided
        # If mask is not None, add -self.eps to the attention scores for positions to ignore
        if mask is not None:
            scaled_dot_product += (mask * -self.eps)

        # Compute attention scores: Apply softmax along S dimension (N, ..., H, L, S)
        self.attention_weights = self.softmax.forward(scaled_dot_product)

        # Calculate output: (N, ..., H, L, Ev)
        # (N, ..., H, L, S) @ (N, ..., H, S, Ev) -> (N, ..., H, L, Ev) 
        output = self.attention_weights @ V
        return output
    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """

        d_V = np.swapaxes(self.attention_weights, -1, -2) @ d_output

        d_attention_scores = d_output @ np.swapaxes(self.V, -1, -2)
        
        d_scaled_dot_product = self.softmax.backward(d_attention_scores)
        
        d_k = self.K.shape[-1]
        d_scaled_dot_product = d_scaled_dot_product / np.sqrt(d_k)
        
        # (N, ..., H, L, S) @ (N, ..., H, S, E) -> (N, ..., H, L, E)   
        d_Q = d_scaled_dot_product @ self.K
        
        # (N, ..., H, L, S)^T @ (N, ..., H, L, E) -> (N, ..., H, S, E)
        d_K = np.swapaxes(d_scaled_dot_product, -1, -2) @ self.Q
        
        return d_Q, d_K, d_V

