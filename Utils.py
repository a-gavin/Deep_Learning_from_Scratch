'''
    Alex Gavin
    Fall 2020

    General utilities. Weight initialization, activation functions.
'''
import numpy as np
from scipy.special import expit as sigmoid


# Activation function stuff
def relu(z: np.ndarray) -> np.ndarray:
    z[z < 0] = 0
    return z


def relu_deriv(z: np.ndarray) -> np.ndarray:
    z[z < 0] = 0
    z[z >= 0] = 1
    return z


def sigmoid_deriv(z: np.ndarray) -> np.ndarray:
    sigmoid_output = sigmoid(z)
    return sigmoid_output * (1 - sigmoid_output)


def tanh_deriv(z: np.ndarray) -> np.ndarray:
    return 1 - (np.tanh(z) ** 2)


# Weight initialization
def kaiming(input_dim, output_dim):
    return np.random.randn(input_dim, output_dim) * np.sqrt(2./input_dim)