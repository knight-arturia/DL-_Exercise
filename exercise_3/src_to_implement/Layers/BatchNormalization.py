import os
import sys
folder_path, _ = os.path.split(__file__)
sys.path.append(folder_path)

import numpy as np
import Helpers

class BatchNormalization:

    running_mean = None    
    running_var = None
    
    testing_phase = False

    optimizer = None

    _image_tensor_shape = None

    gradient_bias = None
    gradient_weights = None


    def __init__(self, channel, momentum=0.8):
        
        self.channel = channel

        self.bias, self.weights = self.initialize(channel)

        self.momentum = momentum 
        


    def initialize(self,channel):
                
        bias = np.zeros((1,channel)) #PB (channel,1)? channel?
        
        weights = np.ones((1,channel))

        return bias,weights
        

    def reformat(self,tensor):

        if tensor.ndim == 4:

            self._image_tensor_shape = tensor.shape

            return self.image2vec(tensor)

        else:
            
            return self.vec2image(self._image_tensor_shape, tensor)

    
    def image2vec(self,tensor):
        
        B, C, M, N = tensor.shape

        tensor = tensor.reshape((B, C, M * N))
                
        tensor = tensor.transpose((0, 2, 1))

        tensor = tensor.reshape((B * M * N, C))
        
        return tensor


    def vec2image(self,shape, tensor):
        
        B, C, M, N = shape
        
        tensor = tensor.reshape((B, M * N, C))        
        tensor = tensor.transpose((0, 2, 1))        
        tensor = tensor.reshape((B, C, M, N))        

        return tensor


    def forward(self, input_tensor):
        

        tensor_dimension = input_tensor.ndim
        

        if tensor_dimension == 4:
                        
            input_tensor = self.reformat(input_tensor)
        
        self.input_tensor = input_tensor
        
        BMN = input_tensor.shape[0]

        if self.weights.shape[0] == 1:
            
            self.weights = self.weights.repeat(BMN, axis=0)
            self.bias = self.bias.repeat(BMN, axis=0)

        
        if not(self.testing_phase):            

            mu = input_tensor.mean(axis=0)
            mu = np.expand_dims(mu, 0).repeat(BMN, axis=0)

            xc = input_tensor - mu

            '''
            if np.sum(xc) > 0.0001:
                print("xc1")
                print(np.sum(xc))        
            '''

            var = np.mean(xc ** 2, axis=0)        
            var = np.expand_dims(var, 0).repeat(BMN, axis = 0)

            # self.var = var
        
            
            # calculate mean        
            if self.running_mean is None:

                self.running_mean = mu
                self.running_var = var

            else:
                self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu # default momentum = 0.8
                self.running_var = self.momentum * self.running_var + (1-self.momentum) * var   

        xc = input_tensor - self.running_mean
        xn = np.true_divide(xc, np.sqrt(self.running_var + np.finfo(np.float32).eps))
        
        '''
        if np.sum(xc) > 0.0001:
            print("xc2")
            print(np.sum(xc))   
        '''

        self.xn = xn
            

        output_tensor = np.multiply(xn, self.weights) + self.bias

        if tensor_dimension == 4:
                        
            output_tensor = self.reformat(output_tensor)

        return output_tensor

    def backward(self, error_tensor):
        
        tensor_dimension = error_tensor.ndim
        
        if tensor_dimension == 4:

            error_tensor = self.reformat(error_tensor)

        BMN = error_tensor.shape[0]
        
        # Gradient with respect to weights
        gradient_weights = np.multiply(error_tensor, self.xn)
        gradient_weights = np.sum(gradient_weights, axis=0)
        gradient_weights = np.expand_dims(gradient_weights, 0).repeat(BMN, axis=0)

        self.gradient_weights = gradient_weights
                
        # Gradient with respect to bias
        gradient_bias = np.sum(error_tensor, axis=0)
        gradient_bias = np.expand_dims(gradient_bias, 0).repeat(BMN, axis=0)

        self.gradient_bias = gradient_bias


        # Gradient with respect to input
        gradient_input = Helpers.compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.running_mean, self.running_var)

        # update weights and bias when optimiezer is defined
        if self.optimizer != None:

            self.weights = self.optimizer.calculate_update(self.weights, gradient_weights)
            self.weights = self.optimizer.calculate_update(self.bias, gradient_bias)
        
        if tensor_dimension == 4:

            gradient_input = self.reformat(gradient_input)
        
        return gradient_input

        

