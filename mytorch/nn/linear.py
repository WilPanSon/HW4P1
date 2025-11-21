import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        """
        self.A = A
        
        A_flat = A.reshape(-1, A.shape[-1])
        
        Z_flat = A_flat @ self.W.T 
        
        Z_flat_biased = Z_flat + self.b
        
        output_shape = A.shape[:-1] + (self.W.shape[0],)
        
        Z = Z_flat_biased.reshape(output_shape)
        
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        self.batch_shape = dLdZ.shape[:-1]
        batch_size_N = np.prod(self.batch_shape) 
        
        dLdZ_flat = dLdZ.reshape(batch_size_N, -1)
        
        A_flat = self.A.reshape(batch_size_N, -1)
        
        
        self.dLdW = dLdZ_flat.T @ A_flat
        
        self.dLdb = np.sum(dLdZ_flat, axis=0) 
        
        dLdA_flat = dLdZ_flat @ self.W
        
        in_features = self.W.shape[1] 
        dLdA_shape = self.batch_shape + (in_features,)
        
        self.dLdA = dLdA_flat.reshape(dLdA_shape)
        
        return self.dLdA
        
