import numpy as np

class Sgd:
    # init the sgd only with a learning rate
    def __init__(self, learning_rate):
        self.lr = learning_rate
    # update weights with backprobagate gradient
    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = weight_tensor - self.lr * gradient_tensor 
        return weight_tensor

class SgdWithMomentum:

    def __init__(self, learning_rate, momentum_rate):
        self.lr = learning_rate
        self.momr = momentum_rate
        self.mom = 0
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        self.mom = self.momr * self.mom - self.lr * gradient_tensor
        weight_tensor = weight_tensor + self.mom
        return weight_tensor

class Adam:

    def __init__(self, learing_rate, mu, rho):
        self.lr = learing_rate
        self.mu = mu
        self.rho = rho
        self.fir_mom = 0
        self.sec_mom = 0
        self.iter_num = 1

    # choose iter_num = 1 at beginning
    def calculate_update(self, weight_tensor, gradient_tensor):
        # first momentum 
        self.fir_mom = self.mu * self.fir_mom + (1 - self.mu) * gradient_tensor
        # second momentum (squared gradient)
        self.sec_mom = self.rho * self.sec_mom + (1- self.rho) * gradient_tensor * gradient_tensor
        # bias correction
        fir_mom_cor = np.true_divide(self.fir_mom, 1 - self.mu**self.iter_num)
        sec_mom_cor = np.true_divide(self.sec_mom, 1 - self.rho**self.iter_num)
        weight_tensor = weight_tensor - self.lr * (fir_mom_cor / (np.sqrt(sec_mom_cor) + np.finfo(float).eps))
        # change iter_num = 2 for next iteration
        self.iter_num += 1
        return weight_tensor