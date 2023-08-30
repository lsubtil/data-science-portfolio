import numpy as np
import matplotlib.pyplot as plt

def initialize_parameters(layer_dims, initialization):
    """
    Arguments:
    layer_dims -- list containing the dimensions of each layer in our network
    initialization -- the initialization method, random or he
    
    Returns:
    parameters -- python dictionary containing the parameters
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    parameters = {}
    L = len(layer_dims)
        
    if initialization == "random":
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])     
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    elif initialization == "he":
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*np.sqrt(2/layer_dims[l-1]) 
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer, numpy array of shape (size of previous layer, number of examples)
    W -- weights matrix, numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function
    cache -- a python tuple containing "A", "W" and "b", stored for computing the backward pass efficiently
    """

    Z = np.dot(W,A) + b
    cache = (A, W, b)
    
    return Z, cache


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array; any shape
    
    Returns:
    A -- output of sigmoid(z); same shape as Z
    cache -- returns Z as well
    """
    
    A = 1/(1 + np.exp(-Z))
    cache = Z
    
    return A, cache


def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer; any shape

    Returns:
    A -- Post-activation parameter; same shape as Z
    cache -- a python dictionary containing "A"
    """
    
    A = np.maximum(0,Z)
    cache = Z 
    
    return A, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A -- activations from previous layer; numpy array of shape (size of previous layer, number of examples)
    W -- weights matrix; numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector; numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer; stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function
    cache -- a python tuple containing "linear_cache" and "activation_cache"
    """
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def forward_pass(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters()
    
    Returns:
    AL -- activation value from the output (last) layer
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                 

    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
    caches.append(cache)
          
    return AL, caches


def compute_cost(AL, Y):
    """
    Implement the the cross-entropy cost function.

    Arguments:
    AL -- activation value from the last layer, shape of (1, number of examples)
    Y -- label vector, shape of (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]
    cost = -np.sum(Y*np.log(AL) + (1 - Y)*(np.log(1 - AL)))/m   
    cost = np.squeeze(cost)

    return cost


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ,A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T,dZ)
    
    return dA_prev, dW, db


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- where we store 'Z' for computing the backward propagation

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0
    
    return dZ


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- where we store 'Z' for computing the backward propagation

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    s = 1/(1 + np.exp(-Z))
    dZ = dA*s*(1 - s)
    
    return dZ


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR -> ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we stored for computing backward propagation
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


def backward_pass(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation
    Y -- labels vector
    caches -- list of caches containing:
                - every cache of linear_activation_forward() with "relu" 
                - the cache of linear_activation_forward() with "sigmoid"
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    # Initializing the backpropagation
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L-1]
    dA_prev, dW, db = linear_activation_backward(dAL, current_cache, 'sigmoid')
    
    grads["dA" + str(L-1)] = dA_prev
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(grads["dA" + str(l+1)], current_cache, 'relu')
        
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l+1)] = dW
        grads["db" + str(l+1)] = db
        
    return grads


def update_parameters(params, grads, learning_rate):
    """
    Update the parameters using gradient descent
    
    Arguments:
    params -- python dictionary containing the parameters 
    grads -- python dictionary containing the gradients
    
    Returns:
    parameters -- python dictionary containing the updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    parameters = params.copy()
    L = len(parameters) // 2 

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]

    return parameters