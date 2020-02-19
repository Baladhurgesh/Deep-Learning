To Train a 2-layer softmax neural network to classify images of hand-written digits from the MNIST dataset. The input to the network will be
a 28 × 28-pixel image (converted into a 784-dimensional vector); the output will be a vector of 10 probabilities (one for each digit). We need to minimize the cross entropy loss.

To get started, first download the MNIST dataset (including both the training, validation, and testing
subsets) from the following web links:

• https://s3.amazonaws.com/jrwprojects/mnist_train_images.npy

• https://s3.amazonaws.com/jrwprojects/mnist_train_labels.npy

• https://s3.amazonaws.com/jrwprojects/mnist_validation_images.npy

• https://s3.amazonaws.com/jrwprojects/mnist_validation_labels.npy

• https://s3.amazonaws.com/jrwprojects/mnist_test_images.npy

• https://s3.amazonaws.com/jrwprojects/mnist_test_labels.npy

These files can be loaded into numpy using np.load.
Then implement stochastic gradient descent (SGD) to minimize the cross-entropy loss function. Regu-
larize the weights but not the bias b. Optimize the same hyperparameters as in homework 2 problem
2 (age regression). You should also use the same methodology as for the previous homework, except
that the MNIST dataset includes a dedicated validation set that you should use.
