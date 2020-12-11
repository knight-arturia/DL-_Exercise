import numpy as np

class Constant:
    def __init__(self, num=0.1):
        self.constant = num
    
    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.ones(weights_shape) * self.constant
        return weights

class UniformRandom:
    def initialize(weights_shape, fan_in, fan_out):
        weights = np.random.uniform(0, 1, size=weights_shape)
        return weights

class Xavier:
    def initialize(weights_shape, fan_in, fan_out):
        stand_devi = np.sqrt(2 / (fan_in + fan_out))
        weights = np.random.normal(0, stand_devi, size=weights_shape)
        return weights

class He:
    def initialize(weights_shape, fan_in, fan_out):
        stand_devi = np.sqrt(2 / fan_in)
        weights = np.random.normal(0, stand_devi, size=weights_shape)
        return weights
