import numpy as np
from numpy.core.fromnumeric import size
from Optimization.Optimizers import Sgd

# given the input data size and output data size
class FullyConnected:
    
    # a Optimizer Object
    _optimizer = False

    def __init__(self, input_size, output_size):
        # interface of input data
        self.input = None
        # weight and bias are random value matrix; +1 means bias vector
        self.weights = np.random.uniform(0, 1, size=(input_size+1, output_size))
        # transmit difference
        self.gradient_weights = None
        
    # a interface to visit protected parameter _optimizer
    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, Obj):
        self._optimizer = Obj
    
    # Input matrix is (N,x+1); Weights matrix is (x+1,m); Output matrix is (N,m); 
    def forward(self, input_tensor):
        # bias input is a unit col vector,shape(N,1)
        self.input = input_tensor
        bias_input = np.ones(input_tensor.shape[0])
        self.input_plus = np.c_[input_tensor, bias_input]
        output = np.dot(self.input_plus, self.weights) # + self.bias
        return output
    
    # backword with gradient; error_tensor is (N, m);
    def backward(self, error_tensor):
        # partial difference of input, drop last col of matrix self.weights; (N,x) = (N,m) * (m,x)
        gradient_input = np.dot(error_tensor, self.weights.T[:,:-1])
        # partial difference of weight, (x,m) = (x,N) * (N,m)
        self.gradient_weig = np.dot(self.input.T, error_tensor)
        # gradient (1, m)
        self.gradient_bias = np.sum(error_tensor, axis=0).reshape(1, error_tensor.shape[1])
        self.gradient_weights = np.r_[self.gradient_weig, self.gradient_bias]
        # cal weights update
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        return gradient_input

