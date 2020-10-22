'''
    Alex Gavin
    Fall 2020
    Implementation of flexible, fully connected neural networks by hand.
'''
import argparse
import re
import numpy as np
from scipy.special import expit as sigmoid


class NN:
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
            print(f"\tError: \"{f1 }\" is not a valid activation function.")
            exit()

        self.pre_activations = []
        self.post_activations = []

    def init_layers(self, D: int, L: str, C: int) -> None:
        layers = []
        layers_str = [re.split("x", layer) for layer in re.split(",", L)]

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

        self.linears = [Layer(l['input_dim'], l['output_dim']) for l in layers]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x

        y_hat = x
        for layer in self.linears:
            y_hat = layer(y_hat)
            self.pre_activations.append(y_hat)

            y_hat = self.activation(y_hat)
            self.post_activations.append(y_hat)

        return y_hat

    def backward(self, y: np.ndarray, lr: float) -> None:
        # Start computing final layer gradient
        post_activation = self.post_activations[-1]
        sensitivity = post_activation - y

        # Gradient for final and intermediary layers
        for ix in range(len(self.post_activations) - 1, 1, -1):
            prev_post_activation = self.post_activations[ix - 1]
            weight_gradient = prev_post_activation @ sensitivity.T

            # Update weights and bias
            layer = self.linears[ix]
            layer.weights -= lr * weight_gradient
            layer.bias -= lr * sensitivity

            sensitivity = self.activation_deriv(self.pre_activations[ix - 1]) * (layer.weights @ sensitivity)

        # Gradient for first layer
        weight_gradient = self.input @ sensitivity.T
        self.linears[0].weights -= lr * weight_gradient
        self.linears[0].bias -= lr * sensitivity

        self.post_activations = []
        self.input = None


# Linear layer initialization
class Layer:
    def __init__(self, input_dim: int, output_dim: int):
        self.weights = kaiming(input_dim, output_dim)
        self.bias = np.zeros(shape=[output_dim, 1])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (self.weights.T @ x) + self.bias

def kaiming(input_dim, output_dim):
    return np.random.randn(input_dim, output_dim) * np.sqrt(2./input_dim)


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


# Data Set Class
class MNISTDataset:
    def __init__(self, input_path, target_path):
        self.inputs = np.load(input_path).astype(np.float32)
        self.targets = np.load(target_path).astype(np.int64)

        n, d = self.inputs.shape
        self.dim = d
        self.len = n

    def __len__(self) -> int:
        return self.len

    def dim(self) -> int:
        return self.dim


# Utilities
def parse_all_args() -> argparse.Namespace:
    # Parses command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("C", help="The number of classes if classification or output dimension if regression (int)",
                        type=int)
    parser.add_argument("train_x", help="The training set input data (npz)")
    parser.add_argument("train_y", help="The training set target data (npz)")
    parser.add_argument("dev_x", help="The development set input data (npz)")
    parser.add_argument("dev_y", help="The development set target data (npz)")

    parser.add_argument("-f1", type=str, help="The hidden activation function: \"relu\",  \"tanh\", \"sigmoid\", or \"identity\" ("
                                              "string) [default: \"relu\"]", default="relu")
    parser.add_argument("-L", type=str, help="A comma delimited list of nunits by nlayers specifiers"
                                             "(see assignment pdf) (string) [default: \"32x1\"]", default="32x1")
    parser.add_argument("-lr", type=float, help="The learning rate (float) [default: 0.1]", default=0.1)
    parser.add_argument("-mb", type=int,
                        help="The minibatch size (int) [default: 1]", default=1)
    parser.add_argument("-report_freq", type=int,
                        help="Dev performance is reported every report_freq updates (int) [default: 128]", default=128)
    parser.add_argument("-e", type=int,
                        help="The number of training epochs (int) [default: 1000]", default=1000)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_all_args()

    # Load data
    train_data = MNISTDataset(args.train_x, args.train_y)
    test_data = MNISTDataset(args.dev_x, args.dev_y)
    breakpoint()

    # Init NN
    model = NN(train_data.dim, args.L, args.C, args.f1)

    # Train model


    # Test model