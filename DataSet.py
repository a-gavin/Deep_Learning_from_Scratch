'''
    Alex Gavin
    Fall 2020

    DataSet class for use in training deep learning models.
'''
import numpy as np


class DataSet:
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs = inputs
        self.targets = targets

        n, d = self.inputs.shape
        self.dim = d
        self.len = n

    def __len__(self) -> int:
        return self.len