import numpy as np

class L2_Regularizer:
    def __init__(self, alpha):
        self.reg_weight = alpha
    
    # cal weights term for enhanced loss 
    def norm(self, weights):
        # flat the matrix to a vector , in order to use numpy function
        weights_vector = weights.flatten()
        return self.reg_weight * np.linalg.norm(weights_vector, ord=2)** 2
        # np.linalg.norm(weights_vector, ord=2)**2 == np.sum(weights_vector**2)

    # cal gradient for backward
    def calculate_gradient(self, weights):
        return self.reg_weight * weights

class L1_Regularizer:
    def __init__(self, alpha):
        self.reg_weight = alpha
    
    # cal weights term for enhanced loss 
    def norm(self, weights):
        weights_vector = weights.flatten()
        return self.reg_weight * np.linalg.norm(weights_vector, ord=1) 

    # cal gradient for backward
    def calculate_gradient(self, weights):
        return self.reg_weight * np.sign(weights)