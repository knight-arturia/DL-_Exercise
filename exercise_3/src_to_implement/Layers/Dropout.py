
from Layers.Base import BaseLayer
import numpy as np 

class Dropout(BaseLayer):
    def __init__(self, probability):
        super(Dropout, self).__init__()
        # probability is the part of input that we want to keep
        self.dropout_prob = probability
        # mask is a matrix consist of {0, 1}, mask.shape = input.shape
        self.mask = None
        
    def forward(self, input_tensor):
        # testing phase means this layer in test yet 
        if self.testing_phase:
            return input_tensor
        else:
            self.mask = np.random.rand(*input_tensor.shape) < self.dropout_prob
            return (input_tensor / self.dropout_prob) * self.mask
        
    def backward(self, error_tensor):
        return (error_tensor / self.dropout_prob) * self.mask
