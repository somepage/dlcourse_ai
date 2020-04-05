import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    preds = (predictions.T - np.max(predictions, axis=1)).T
    exps = np.exp(preds)
    
    return (exps.T / exps.sum(axis=1)).T


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss
    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    return - np.log(probs[np.arange(len(probs)), target_index]).mean()


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient
    Arguments:
      W, np array - weights
      reg_strength - float value
    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    loss = reg_strength * np.power(W, 2).sum()
    grad = reg_strength * 2 * W
    
    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    probs = softmax(preds)
    loss = cross_entropy_loss(probs, target_index).mean()
    dprediction = np.zeros_like(preds)
    dprediction[np.arange(preds.shape[0]), target_index] = 1
    dprediction = - (dprediction - softmax(preds)) / dprediction.shape[0]
    
    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        self.is_positive = True

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        output = np.maximum(0, X)
        self.is_positive = output.astype(bool)
        
        return output
        

    def backward(self, d_out):
        """
        Backward pass
        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output
        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = self.is_positive.astype(int) * d_out
        
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        
        return self.X @ self.W.value + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B
        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output
        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        self.W.grad += self.X.T @ d_out
        self.B.grad += 2 * np.mean(d_out, axis=0)
        d_input = d_out @ self.W.value.T
        
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        self.X = X

        if self.padding:
            self.X = np.zeros((batch_size,
                               height + 2 * self.padding,
                               width + 2 * self.padding,
                               channels), dtype=X.dtype)
            self.X[:, self.padding: -self.padding, self.padding: -self.padding, :] = X

        _, height, width, channels = self.X.shape

        out_height = height - self.filter_size + 1
        out_width = width - self.filter_size + 1

        output = []

        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            row = []
            for x in range(out_width):
                cube = self.X[:, y: y + self.filter_size, x: x + self.filter_size, :]
                cube = np.transpose(cube, axes=[0, 3, 2, 1]).reshape((batch_size, self.filter_size ** 2 * channels))
                # cube = cube.reshape((batch_size, self.filter_size ** 2 * channels))
                W_cube = np.transpose(self.W.value, axes=[2, 0, 1, 3])
                out = cube.dot(W_cube.reshape((self.filter_size ** 2 * self.in_channels, self.out_channels)))
                # out has shape (batch_size, out_channel)
                row.append(np.array([out], dtype=self.W.value.dtype).reshape((batch_size, 1, 1, self.out_channels)))
            output.append(np.dstack(row))
        output = np.hstack(output)
        output += self.B.value

        return output
        

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        d_inp = np.zeros(self.X.shape)
        window = np.zeros(self.X.shape)
        for y in range(out_height):
            for x in range(out_width):
                # d_cube is shape (batch_size, out_channel) => (batch_size, out_features)
                d_cube = d_out[:, y, x, :]
                # X_cube is shape (batch_size, filter_size, filter_size, channels)
                X_cube = self.X[:, y: y + self.filter_size, x: x + self.filter_size, :]
                # X_cube is shape (batch_size, filter_size * filter_size * channels) => (batch_size, in_features)
                #                                   0, 1, 2, 3
                X_cube = np.transpose(X_cube, axes=[0, 3, 1, 2]).reshape((batch_size, self.filter_size ** 2 * channels))
                # W_cube is shape (filter_size * filter_size * in_channels, out_shannel) => (in_features, out_features)
                W_cube = np.transpose(self.W.value, axes=[2, 0, 1, 3])
                W_cube = W_cube.reshape((self.filter_size ** 2 * self.in_channels, self.out_channels))
                # self.W.grad = self.X.transpose().dot(d_out)
                # E = np.ones(shape=(1, self.X.shape[0]))
                # self.B.grad = E.dot(d_out)
                # d_out.dot(self.W.value.transpose())
                # gradiants for dense layer reshaped to shape of W
                d_W_cube = (X_cube.transpose().dot(d_cube)).reshape(self.in_channels,
                                                                    self.filter_size,
                                                                    self.filter_size,
                                                                    self.out_channels)

                self.W.grad += np.transpose(d_W_cube, axes=[2, 1, 0, 3])
                E = np.ones(shape=(1, batch_size))
                self.B.grad += E.dot(d_cube).reshape((d_cube.shape[1]))

                # d_cube : (batch_size, out_features) dot W_cube.transpose: (out_features, in_features)
                # d_inp_xy is shape (batch_size, in_features)
                d_inp_xy = d_cube.dot(W_cube.transpose())
                d_inp_xy = d_inp_xy.reshape((batch_size, channels, self.filter_size, self.filter_size))
                #                                       0, 1, 2, 3
                d_inp_xy = np.transpose(d_inp_xy, axes=[0, 3, 2, 1])

                d_inp[:, y: y + self.filter_size, x: x + self.filter_size, :] += d_inp_xy
                window[:, y: y + self.filter_size, x: x + self.filter_size, :] += 1

        if self.padding:
            d_inp = d_inp[:, self.padding: -self.padding, self.padding: -self.padding, :]

        return d_inp

                

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool
        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        self.X = X

        output = []
        for y in range(0, height, self.stride):
            row = []
            for x in range(0, width, self.stride):
                X_cube = X[:, y: y + self.pool_size, x: x + self.pool_size, :]
                row.append(X_cube.max(axis=1).max(axis=1).reshape(batch_size, 1, 1, channels))
            row = np.dstack(row)
            output.append(row)
        output = np.hstack(output)

        return output

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        
        output = np.zeros(self.X.shape)
        
        for y_num, y in enumerate(range(0, height, self.stride)):
            for x_num, x in enumerate(range(0, width, self.stride)):
                d_cube = d_out[:, y_num, x_num, :]
                d_cube = d_cube.reshape(batch_size, 1, 1, channels)
                X_cube = self.X[:, y: y + self.pool_size, x: x + self.pool_size, :]
                d_cube_out = (X_cube == X_cube.max(axis=1).max(axis=1).reshape(batch_size, 1, 1, channels)) * d_cube
                # print(X_cube == X_cube.max(axis=1).max(axis=1).reshape(batch_size, 1, 1, channels))
                # print(d_cube)
                output[:, y: y + self.pool_size, x: x + self.pool_size, :] += d_cube_out.astype(np.float32)
        return output
        
        
        

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        
        return X.reshape((batch_size, height * width * channels))

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X.shape)

    def params(self):
        # No params!
        return {}