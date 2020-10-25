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


# Misc
def calc_num_updates(data_size: int, mb: int) -> int:
    # If num train data points not perfectly
    # divisible by minibatch size, inc updates.
    # Ensures final chunk of data not ignored.
    num_train_updates = int(data_size / mb)
    if data_size % mb != 0:
        num_train_updates += 1

    return num_train_updates