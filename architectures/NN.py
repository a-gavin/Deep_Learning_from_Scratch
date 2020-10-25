'''
    Alex Gavin
    Fall 2020

    Flexible neural network class for use in training deep learning models.
'''
import numpy as np
from re import split
from scipy.special import expit as sigmoid
from scipy.special import softmax

from architectures.Model import Model
from architectures.LinearLayer import LinearLayer
from Utils import sigmoid_deriv, tanh_deriv, relu, relu_deriv


class NN(Model):
    def __init__(self, D: int, L: str, C:int, f1: str):
        self.init_layers(D, L, C)

        if f1 == "sigmoid":
            self.activation = sigmoid
            self.activation_deriv = sigmoid_deriv
        elif f1 == "tanh":
            self.activation = np.tanh
            self.activation_deriv = tanh_deriv
        elif f1 == "relu":
            self.activation = relu
            self.activation_deriv = relu_deriv
        else:
            print(f"\tError: \"{f1}\" is not a valid activation function.")
            exit()

        self.pre_activations = []
        self.post_activations = []

    def init_layers(self, D: int, L: str, C: int) -> None:
        layers = []
        layers_str = [split("x", layer) for layer in split(",", L)]

        if not layers_str:
            print(f"\tError: Layers \"{L}\" specified incorrectly.")
            exit()

        # Iterate through comma separated layers
        for ix, layer in enumerate(layers_str):
            # Iterate through layers of specified dimension
            for layer_num in range(int(layer[1])):
                cur_dim = int(layers_str[ix][0])

                if layer_num == 0:
                    # First layer of first dim, set input size to data size
                    if ix == 0:
                        prev_dim = D
                    else:
                        prev_dim = int(layers_str[ix - 1][0])
                else:
                    prev_dim = cur_dim

                layers.append({'input_dim': prev_dim,
                               'output_dim': cur_dim})
        # Final layer
        layers.append({'input_dim': int(layers_str[-1][0]),
                       'output_dim': C})

        self.linears = [LinearLayer(l['input_dim'], l['output_dim']) for l in layers]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x

        # Evaluate layers 1 through k-1
        y_hat = x
        for ix in range(len(self.linears)-1):
            layer = self.linears[ix]

            y_hat = layer(y_hat)
            self.pre_activations.append(y_hat)

            y_hat = self.activation(y_hat)
            self.post_activations.append(y_hat)

        # Evaluate layer k
        last_layer = self.linears[-1]

        y_hat = last_layer(y_hat)
        self.pre_activations.append(y_hat)

        y_hat = softmax(y_hat, axis=0)
        self.post_activations.append(y_hat)

        return y_hat

    def backward(self, y: np.ndarray, lr: float) -> None:
        _, mb_size = y.shape

        # Start computing final layer gradient
        post_activation = self.post_activations[-1]
        sensitivity = post_activation - y

        # Gradient for final and intermediary layers
        for ix in range(len(self.post_activations) - 1, 0, -1):
            prev_post_activation = self.post_activations[ix - 1]
            weight_gradient = prev_post_activation @ sensitivity.T

            # Update weights and bias
            layer = self.linears[ix]
            layer.weights -= (lr * weight_gradient) / mb_size
            layer.bias -= (lr * (sensitivity @ np.ones(shape=[mb_size, 1]))) / mb_size

            sensitivity = self.activation_deriv(self.pre_activations[ix - 1]) * (layer.weights @ sensitivity)

        # Gradient for first layer
        weight_gradient = self.input @ sensitivity.T
        self.linears[0].weights -= (lr * weight_gradient) / mb_size
        self.linears[0].bias -= (lr * (sensitivity @ np.ones(shape=[mb_size, 1]))) / mb_size

        # Reset storage variables
        self.pre_activations = []
        self.post_activations = []
        self.input = None
