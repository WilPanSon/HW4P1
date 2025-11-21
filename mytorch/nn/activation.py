import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        self.Z = Z
        
        Z_max = np.max(Z, axis=self.dim, keepdims=True)
        
        Z_shifted = Z - Z_max

        numerator = np.exp(Z_shifted)
        
        denominator = np.sum(numerator, axis=self.dim, keepdims=True)

        A = numerator / denominator
        
        self.A = A
        return A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output A (*, C)
        :return: Gradient of loss with respect to activation input Z (*, C)
        """
        shape = self.A.shape

        dim = self.dim
        C = shape[dim]

        A_moved = np.moveaxis(self.A, dim, -1)
        dLdA_moved = np.moveaxis(dLdA, dim, -1)

        moved_shape = A_moved.shape

        N = np.prod(moved_shape[:-1])
        
        A_flat = A_moved.reshape(N, C)
        dLdA_flat = dLdA_moved.reshape(N, C)

        product = A_flat * dLdA_flat 
        
        scalar_part = np.sum(product, axis=1, keepdims=True)
        
        dLdZ_flat = A_flat * (dLdA_flat - scalar_part)

        dLdZ_moved = dLdZ_flat.reshape(moved_shape)
        dLdZ = np.moveaxis(dLdZ_moved, -1, dim)
        
        self.dLdZ = dLdZ
        
        return self.dLdZ
 

    