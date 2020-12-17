# import numpy as np

class ReLU:
    # receive no parameters
    def __init__(self):
        # interface for input tensor and error tensor
        self.neg = None
    
    # change all negative num in input to 0, positive unchanged
    def forward(self, input_tensor):
        
        self.neg = (input_tensor <= 0)
        output = input_tensor.copy()
        output[self.neg] = 0
        return output
    
    # for all negative num in forward input, their err will also be 0 in backward 
    def backward(self, error_tensor):
        error_tensor[self.neg] = 0
        gradient_input = error_tensor
        return gradient_input
