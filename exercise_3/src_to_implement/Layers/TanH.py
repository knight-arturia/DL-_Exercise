import numpy as np

class TanH:
    def forward(input_tensor):
        return np.tanh(input_tensor)

    def backward(error_tensor):
        return 1 - np.tanh(error_tensor)**2
