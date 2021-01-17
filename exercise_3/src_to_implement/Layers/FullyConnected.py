from Layers.Base import BaseLayer
import numpy as np
import copy
from numpy.core.fromnumeric import size
from Optimization.Optimizers import Sgd

# given the input data size and output data size
class FullyConnected(BaseLayer):
    
    # a Optimizer Object
    _optimizer = False
    _optimizer_w = False
    _optimizer_b = False

    def __init__(self, input_size, output_size):
        # init father class Baslayer
        super(BaseLayer, self).__init__()
        # interface of input data
        self.input = None
        # weight and bias are random value matrix; +1 means bias vector
        self.weight_shape = (input_size, output_size)
        self.bias_shape = (output_size, )
        self.weights = np.random.uniform(0, 1, size=self.weight_shape)
        self.bias = np.random.uniform(0, 1, size=self.bias_shape)
        # transmit difference
        self.gradient_weights = None
        self.gradient_bias = None
    
    def initialize(self, weight_initializer, bias_initializer):

        fan_in = self.weight_shape[0]
        fan_out = self.weight_shape[1]

        self.weights = weight_initializer.initialize(self.weight_shape, fan_in, fan_out)

        self.bias = bias_initializer.initialize(self.bias_shape, fan_in, fan_out)
    
    """
    optimizer for weights and bias will be separate
    """
    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, Obj):
        self._optimizer = Obj
        self._optimizer_b = copy.deepcopy(self._optimizer)
        self._optimizer_w = copy.deepcopy(self._optimizer)
    
    # Input matrix is (N,x+1); Weights matrix is (x+1,m); Output matrix is (N,m); 
    def forward(self, input_tensor):
        # bias input is a unit col vector,shape(N,1)
        self.input = input_tensor
        bias_input = np.ones(input_tensor.shape[0])
        self.input_plus = np.c_[input_tensor, bias_input]

        # combine weights and bias together
        weight_bias = np.vstack((self.weights, self.bias))
        # print("weight = ", self.weights.shape, "bias = ", self.bias.shape)
        output = np.dot(self.input_plus, weight_bias)
        return output
    
    # backword with gradient; error_tensor is (N, m);
    def backward(self, error_tensor):
        # partial difference of input, drop last col of matrix self.weights; (N,x) = (N,m) * (m,x)
        gradient_input = np.dot(error_tensor, self.weights.T)
        # partial difference of weight, (x,m) = (x,N) * (N,m)
        self.gradient_weights = np.dot(self.input.T, error_tensor)
        # gradient (1, m)
        self.gradient_bias = np.sum(error_tensor, axis=0).reshape(1, error_tensor.shape[1])
        
        # cal weights update
        if self._optimizer_w:
            self.weights = self._optimizer_w.calculate_update(self.weights, self.gradient_weights)
        if self._optimizer_b:
            self.bias = self._optimizer_b.calculate_update(self.bias, self.gradient_bias)
        
        return gradient_input

