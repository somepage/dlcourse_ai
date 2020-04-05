import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        self.layers = [ConvolutionalLayer(input_shape[2], conv1_channels, 4, 0),
                       ReLULayer(),
                       MaxPoolingLayer(2, 2),
                       ConvolutionalLayer(conv1_channels, conv2_channels, 4, 0),
                       ReLULayer(),
                       MaxPoolingLayer(2, 2),
                       Flattener(),
                       FullyConnectedLayer(int(input_shape[0] * input_shape[1] * conv2_channels / 16), n_output_classes)]
        
        

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        val = X
        
        for i in range(len(self.layers)):
            val = self.layers[i].forward(val)
            for p in self.layers[i].params().keys():
                self.layers[i].params()[p].grad = np.zeros_like(self.layers[i].params()[p].value)

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        loss, grad = softmax_with_cross_entropy(val, y)
        
        for i in reversed(range(len(self.layers))):
            grad = self.layers[i].backward(grad)
                
        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        pred = np.zeros(X.shape[0], np.int)
        output = X.copy()
        
        for i in range(len(self.layers)):
            output = self.layers[i].forward(output)
        
        pred = output.argmax(axis=1)
        return pred

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        for i in range(len(self.layers)):
            for p in self.layers[i].params().keys():
                result[p +str(i)] =  self.layers[i].params()[p]

        return result
