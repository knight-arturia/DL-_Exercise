import numpy as np

class Sigmoid:

    def forward(self,input_tensor):

        activation_tensor = np.exp(-input_tensor)
        activation_tensor = np.true_divide(1, 1 + activation_tensor)
        
        self.activation_tensor = activation_tensor
        
        return activation_tensor
    

    def backward(self, error_tensor):
        
        next_error_tensor = np.true_divide(1, error_tensor)
        next_error_tensor = -np.log(next_error_tensor - 1 + np.finfo(np.float32).eps)
        