import numpy as np

class Sigmoid:

    def forward(input_tensor):
        return 1 / (1 + np.exp(-input_tensor))
    
    def backward(error_tensor):
        sig = 1 / (1 + np.exp(-error_tensor))
        return sig * (1 - sig)