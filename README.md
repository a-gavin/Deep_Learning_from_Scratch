# Deep Learning from Scratch

### Description:
A simple Python3 application for building and training deep neural network based classifiers. Implementation inspired by the PyTorch API and notes from Dr. Brian Hutchinson's deep learning course at WWU, CSCI597J (now [CSCI581](https://catalog.wwu.edu/preview_course_nopop.php?catoid=16&coid=125176&)).

Modified MNIST data set for example usage located here: _______.

**Note:** Currently, only SGD is supported.

### Example invocations:

* Train 10 class classifier on neural network with two hidden layers, 128 and 64 nodes each, with learning rate 0.01, tanh as hidden activation function for 10 epochs:

> `python3 Main.py 10 <path_to_input_data> <path_to_target_data> -L 128x1,64x1 -lr 0.01 -e 10 -f1 tanh`

* Train binary classifier on neural network with one layer with 32 nodes (default) with sigmoid as hidden activation function, minibatch size of 32 inputs, and a train/dev/test split of 80%, and 10%, 10%, respectively:

> `python3 Main.py 2 <path_to_input_data> <path_to_target_data> -f1 sigmoid -mb 32 -split_pctgs 80,10,10`