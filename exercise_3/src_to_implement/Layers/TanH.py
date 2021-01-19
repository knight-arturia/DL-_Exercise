import numpy as np

class TanH:

    def forward(self,input_tensor):

        activation_tensor = np.tanh(input_tensor)
                
        self.activation_tensor = activation_tensor
        
        return activation_tensor
    

    def backward(self, error_tensor):
        
        derivation_y_x = 1 - np.multiply(self.activation_tensor, self.activation_tensor)
        
        gradient_input = np.multiply(error_tensor, derivation_y_x)
        
        return gradient_input
        
