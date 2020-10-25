'''
    Alex Gavin
    Fall 2020

    Basic linear layer class for use in training deep learning models.
'''
import numpy as np
from Utils import kaiming


# Linear layer initialization
class LinearLayer:
    def __init__(self, input_dim: int, output_dim: int):
        self.weights = kaiming(input_dim, output_dim)
        self.bias = np.zeros(shape=[output_dim, 1])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = (self.weights.T @ x) + self.bias

        return z
