'''
    Alex Gavin
    Fall 2020

    Abstract base class for implementing deep learning models.
'''
import numpy as np


class Model:
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def backward(self, y: np.ndarray, lr: float) -> None:
        raise NotImplementedError
