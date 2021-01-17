from Layers.Base import BaseLayer
import numpy as np

class SoftMax(BaseLayer):
    def __init__(self):
        # init father class Baslayer
        super(BaseLayer, self).__init__()
        self.output = None
    
    # return the probability 
    def forward(self, input_tensor):
        max = np.max(input_tensor)
        # in order to avoid large num overflow: input_tensor - max
        exp_in = np.exp(input_tensor - max)
        sum_exp = np.sum(exp_in, axis=1)
        sum_exp = np.transpose(np.expand_dims(sum_exp, 0).repeat(input_tensor.shape[1], axis=0))
        self.output = np.true_divide(exp_in, sum_exp)
        return self.output
    
    # return the backward with error_tensor is (-label/output)
    def  backward(self, error_tensor):
        # backward output of Softmax is (output - label)
        sum = np.sum(np.multiply(error_tensor, self.output), axis=1)
        # expend sum from (N, ) to a (N, m)
        sum = np.transpose(np.expand_dims(sum, 0).repeat(error_tensor.shape[1], axis=0))
        back_error = self.output * (error_tensor - sum)
        return back_error
