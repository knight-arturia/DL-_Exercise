import numpy as np

class Flatten:
    def __init__(self):
        self.ori_shape = None
        pass
    
    def forward(self, input_tensor):
        self.ori_shape = input_tensor.shape
        return input_tensor.reshape(input_tensor.shape[0], -1)
    
    def backward(self, error_tensor):
        return error_tensor.reshape(self.ori_shape)