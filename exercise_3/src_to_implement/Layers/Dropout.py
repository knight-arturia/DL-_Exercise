
from Layers.Base import BaseLayer
import numpy as np 

class Dropout(BaseLayer):
    def __init__(self, probability):
        super(BaseLayer, self).__init__()
        # probability is the part of input that we want to keep
        self.dropout_prob = probability
        # mask is a matrix consist of {0, 1}, mask.shape = input.shape
        self.mask = None
        
    def forward(self, input_tensor):
        # testing phase means test time here
        if self.testing_phase:
            return input_tensor * self.dropout_prob
        else:
            self.mask = np.random.rand(*input_tensor.shape) < self.dropout_prob
            return input_tensor * self.mask
        
    def backward(self, backward_tensor):
        return backward_tensor * self.mask
