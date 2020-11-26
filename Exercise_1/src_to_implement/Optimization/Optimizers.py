import numpy as np

class Sgd:
    # init the sgd only with a learning rate
    def __init__(self, learning_rate):
        self.lr = learning_rate
    # update weights with backprobagate gradient
    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = weight_tensor - self.lr * gradient_tensor 
        return weight_tensor
