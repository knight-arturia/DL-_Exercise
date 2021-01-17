import numpy as np

class Constant:
    def __init__(self, num=0.1):
        self.constant = num
    
    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.ones(weights_shape) * self.constant
        return weights

class UniformRandom:
    def __init__(self):
        self.weights = None
        
    def initialize(self, weights_shape, fan_in, fan_out):
        self.weights = np.random.uniform(0, 1, size=weights_shape)
        return self.weights

class Xavier:
    def __init__(self):
        self.weights = None
    
    def initialize(self, weights_shape, fan_in, fan_out):
        stand_devi = np.sqrt(2 / (fan_in + fan_out))
        self.weights = np.random.normal(0, stand_devi, size=weights_shape)
        return self.weights

class He:
    def __init__(self):
        self.weights = None
    
    def initialize(self, weights_shape, fan_in, fan_out):
        stand_devi = np.sqrt(2 / fan_in)
        self.weights = np.random.normal(0, stand_devi, size=weights_shape)
        return self.weights
