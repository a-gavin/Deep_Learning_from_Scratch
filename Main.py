'''
    Alex Gavin
    Fall 2020

    Implementation of flexible, fully connected neural networks
    from scratch for training classifiers.
'''
import argparse
import re
import numpy as np
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder

from DataSet import DataSet
from architectures.NN import NN

# TODO: REMOVE ME, used for debugging Relu
np.seterr(all='raise')


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
    parser.add_argument("-split_pctgs", type=str, help="A comma delimited list denoting train, dev, and test percentages, respectively."
                                             "(string) [default: \"70,20,10\"]", default="70,20,10")
    parser.add_argument("-lr", type=float, help="The learning rate (float) [default: 0.1]", default=0.1)
    parser.add_argument("-mb", type=int,
                        help="The minibatch size (int) [default: 1]", default=1)
    parser.add_argument("-report_freq", type=int,
                        help="Dev performance is reported every report_freq updates (int) [default: 1000]", default=1000)
    parser.add_argument("-e", type=int,
                        help="The number of training epochs (int) [default: 1000]", default=1000)

    return parser.parse_args()


def load_data(inputs_path: str, targets_path: str, split_pctgs: str) -> (DataSet, DataSet, DataSet):
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

    # Load inputs
    # Load targets, convert to one hot enc
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

    train_data = DataSet(train_inputs, train_targets)
    dev_data = DataSet(dev_inputs, dev_targets)
    test_data = DataSet(test_inputs, test_targets)

    return train_data, dev_data, test_data


def train(model: NN, train_data: DataSet, dev_data: DataSet, 
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
                

def test(model: NN, data: DataSet):
    acc = 0.0
    N = data.len
    mb = 32

    for ix in range(int(data.len/mb)):
        bottom_bound = ix * mb
        upper_bound = bottom_bound + mb

        X = data.inputs[bottom_bound:upper_bound, :].T
        y = data.targets[bottom_bound:upper_bound, :].T

        y_pred = model.forward(X)

        # Reset data saved for backprop in forward
        # Not needed for testing
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
    train_data, dev_data, test_data = load_data(args.inputs, args.targets, args.split_pctgs)

    # Init NN
    model = NN(train_data.dim, args.L, args.C, args.f1)

    # Train model
    train(model, train_data, dev_data, args.mb, args.lr, args.e, args.report_freq)

    # Test model
    test_acc = test(model, test_data)
    print(f"Model test accuracy: {100*test_acc:.1f}%")