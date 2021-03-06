import numpy as np

class Sigmoid:

    def forward(self,input_tensor):

        activation_tensor = np.exp(-input_tensor)
        activation_tensor = np.true_divide(1, 1 + activation_tensor)
        
        self.activation_tensor = activation_tensor
        
        return activation_tensor
    

    def backward(self, error_tensor):
        
        derivation_y_x = np.multiply(self.activation_tensor, 1 - self.activation_tensor)

        gradient_input = np.multiply(error_tensor, derivation_y_x)
        
        return gradient_input

        
