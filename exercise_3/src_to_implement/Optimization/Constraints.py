import numpy as np

class L2_Regularizer:
    def __init__(self, alpha):
        self.reg_weight = alpha
    
    # cal weights term for enhanced loss 
    def norm(self, weights):
        return self.reg_weight * np.linalg.norm(weights, ord=2, axis=None, keepdims=False)**2 

    # cal gradient for backward
    def calculate_gradient(self, weights):
        return self.reg_weight * weights

class L1_Regularizer:
    def __init__(self, alpha):
        self.reg_weight = alpha
    
    # cal weights term for enhanced loss 
    def norm(self, weights):
        return self.reg_weight * np.linalg.norm(weights, ord=1, axis=None, keepdims=False) 

    # cal gradient for backward
    def calculate_gradient(self, weights):
        return self.reg_weight * np.sign(weights)