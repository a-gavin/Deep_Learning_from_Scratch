'''
    Alex Gavin
    Fall 2020

    Implementation of flexible, fully connected neural networks
    for classifying MNIST data from scratch.
'''
import argparse
import re
import numpy as np
from scipy.special import expit as sigmoid
from scipy.special import softmax
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder

# TODO: REMOVE ME, used for debugging Relu
np.seterr(all='raise')

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
            print(f"\tError: \"{f1}\" is not a valid activation function.")
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


# Linear layer initialization
class Layer:
    def __init__(self, input_dim: int, output_dim: int):
        self.weights = kaiming(input_dim, output_dim)
        self.bias = np.zeros(shape=[output_dim, 1])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = (self.weights.T @ x) + self.bias

        return z

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
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

        n, d = self.inputs.shape
        self.dim = d
        self.len = n

    def __len__(self) -> int:
        return self.len

    def dim(self) -> int:
        return self.dim


def load_data(inputs_path: str, targets_path: str, split_pctgs: str) -> (MNISTDataset, MNISTDataset, MNISTDataset):
    # Convert split percentages to floats for data sampling
    splits_str = re.split(",", split_pctgs)

    if not splits_str:
        print(f"\tError: Data splits \"{split_pctgs}\" specified incorrectly.")
        exit()

    splits_float = [float(split_pctg)/100 for split_pctg in splits_str]

    split_sum = sum(splits_float)
    if split_sum <= 0 or 1 < split_sum:
        print(f"\tError: Data splits \"{split_pctgs}\" specified incorrectly.")
        exit()

    # Load MNIST images
    # Load MNIST labels, convert to one hot enc
    inputs = np.load(inputs_path).astype(np.float32)

    int_targets = np.load(targets_path).astype(np.int64).reshape([-1, 1])
    enc = OneHotEncoder(sparse=False)
    targets = enc.fit_transform(int_targets)

    # Generate random indices for data sampling
    n, _ = inputs.shape
    rand_indices = np.arange(0, n)
    np.random.shuffle(rand_indices)

    # Sample and initialize data
    train_size = int(n * splits_float[0])
    dev_size = int(n * splits_float[1])

    train_indices = rand_indices[:train_size]
    train_inputs = np.take(inputs, train_indices, axis=0)
    train_targets = np.take(targets, train_indices, axis=0)

    dev_indices = rand_indices[train_size:(train_size + dev_size)]
    dev_inputs = np.take(inputs, dev_indices, axis=0)
    dev_targets = np.take(targets, dev_indices, axis=0)
    
    test_indices = rand_indices[(train_size + dev_size):]
    test_inputs = np.take(inputs, test_indices, axis=0)
    test_targets = np.take(targets, test_indices, axis=0)

    train_data = MNISTDataset(train_inputs, train_targets)
    dev_data = MNISTDataset(dev_inputs, dev_targets)
    test_data = MNISTDataset(test_inputs, test_targets)

    return train_data, dev_data, test_data

# Utilities
def parse_all_args() -> argparse.Namespace:
    # Parses command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("C", help="The number of classes if classification or output dimension if regression (int)",
                        type=int)
    parser.add_argument("inputs", help="Input data (npz)")
    parser.add_argument("targets", help="Target data (npz)")

    parser.add_argument("-f1", type=str, help="The hidden activation function: \"relu\",  \"tanh\", \"sigmoid\", or \"identity\" ("
                                              "string) [default: \"relu\"]", default="relu")
    parser.add_argument("-L", type=str, help="A comma delimited list of nunits by nlayers specifiers"
                                             "(string) [default: \"32x1\"]", default="32x1")
    parser.add_argument("-spli_pctgs", type=str, help="A comma delimited list denoting train, dev, and test percentages, respectively."
                                             "(string) [default: \"70,20,10\"]", default="70,20,10")
    parser.add_argument("-lr", type=float, help="The learning rate (float) [default: 0.1]", default=0.1)
    parser.add_argument("-mb", type=int,
                        help="The minibatch size (int) [default: 1]", default=1)
    parser.add_argument("-report_freq", type=int,
                        help="Dev performance is reported every report_freq updates (int) [default: 1000]", default=1000)
    parser.add_argument("-e", type=int,
                        help="The number of training epochs (int) [default: 1000]", default=1000)

    return parser.parse_args()


def train(model: NN, train_data: MNISTDataset, dev_data: MNISTDataset, 
                            mb: int, lr: float, epochs: int, report_freq: int):

    for epoch in range(1, epochs+1):
        print(f"->\tEpoch {epoch}")

        for ix in range(int(train_data.len/mb)):  # TODO: Fix iteration to ensure last data points not dropped
            bottom_bound = ix * mb
            upper_bound = bottom_bound + mb

            X = train_data.inputs[bottom_bound:upper_bound, :].T
            y = train_data.targets[bottom_bound:upper_bound, :].T

            y_pred = model.forward(X)  # TODO: Eval and print loss
            model.backward(y, lr)

            if (ix % report_freq) == 0:
                dev_acc = test(model, dev_data)
                print(f"{epoch:03d} -- dev acc: {100*dev_acc:0.1f}%")
        print()
                

def test(model, data):
    acc = 0.0
    N = data.len
    mb = 32

    for ix in range(int(data.len/mb)):
        bottom_bound = ix * mb
        upper_bound = bottom_bound + mb

        X = data.inputs[bottom_bound:upper_bound, :].T
        y = data.targets[bottom_bound:upper_bound, :].T

        y_pred = model.forward(X)

        # Reset data saved for backprop
        # as it is not needed for testing
        model.pre_activations = []
        model.post_activations = []
        model.input = None

        matched_outputs = np.argmax(y_pred, axis=0) == np.argmax(y, axis=0)
        acc += matched_outputs.sum()

    acc /= N

    return acc


if __name__ == "__main__":
    args = parse_all_args()

    # Load data
    train_data, dev_data, test_data = load_data(args.inputs, args.targets, args.spli_pctgs)

    # Init NN
    model = NN(train_data.dim, args.L, args.C, args.f1)

    # Train model
    train(model, train_data, dev_data, args.mb, args.lr, args.e, args.report_freq)

    # Test model
    test_acc = test(model, test_data)
    print(f"Model test accuracy: {100*test_acc:.1f}%")