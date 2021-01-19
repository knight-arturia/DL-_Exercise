import numpy as np
from numpy.core.fromnumeric import shape

class RNN:
    _optimizer = None
    '''
    assume that 
    input_size = (b * in)
    output_size = (b * out)
    hidden_size = h
    '''
    def __init__(self, input_size, hidden_size, output_size):
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        # init hidden state
        self.hidden_state = np.zeros(hidden_size)
        # show the sequence belonging of this cell
        self.k2 = None
        self.seq_flag = False
        # init weights and bias
        self.W_xh = np.random.uniform(0, 1, size=(input_size+1, hidden_size))
        self.W_hh = np.random.uniform(0, 1, size=(hidden_size, hidden_size))
        self.W_hy = np.random.uniform(0, 1, size=(hidden_size, output_size))
        self.B_y = np.random.uniform(0, 1, size=output_size)
    
    '''
    to initialize the weights and bias
    '''
    def initialize(self, weight_initializer, bias_initializer):
        self.W_xh = weight_initializer.initialize(self.W_xh.shape, self.input_size+1, self.hidden_size)
        self.W_hh = weight_initializer.initialize(self.W_hh.shape, self.hidden_size, self.hidden_size)
        self.W_hy = weight_initializer.initialize(self.W_hy.shape, self.hidden_size, self.output_size)
        self.B_y = bias_initializer.initialize(self.B_y.shape, self.hidden_size, self.output_size)
    
    '''
    memorize is how many steps we need to memorize, k2
    '''
    @property
    def memorize(self):
        return self.k2
    @memorize.setter
    def memorize(self, value):
        self.k2 = value
    
    def forward(self, input_tensor):
        # add one col 1 to the end of input_tensor
        col = np.ones(input_tensor.shape[0])
        input_tensor_extend = np.c_[input_tensor, col]

        output_tensor = np.zeros((input_tensor.shape[0], self.output_size))
        ht_tensor = np.zero((input_tensor.shape[0], self.hidden_size))
        for i in range(input_tensor.shape[0]):
            # if sequence flag is false means a new start of back propagation sequence 
            if self.seq_flag == False:
                self.hidden_state = np.zeros(self.hidden_size)
                # set the h_t-1 back to 0
                # ht_tensor[i-1] = 0
                self.seq_flag = True
            h1 = np.dot(self.hidden_state, self.W_hh)
            h2 = np.dot(input_tensor_extend[i], self.W_xh)
            ht = np.tanh(h1 + h2)
            # save h_t for backward propagation
            ht_tensor[i] = ht
            self.hidden_state = ht
            y1 = np.dot(self.hidden_state, self.W_hy) + self.B_y
            output_tensor[i] = 1 / (1 + np.exp(-y1))
        
        # store the input, output, ht for the backpropagation
        self.input_tensor = input_tensor_extend
        self.ht_tensor = ht_tensor
        self.output_tensor = output_tensor

        return output_tensor
    
    '''
    give the optimizer for weights and bias
    '''
    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, Obj):
        self._optimizer = Obj
    
    '''
    cal all regularization loss
    '''
    @property
    def calculate_regularization_loss(self):
        return (self.optimizer.reg.norm(self.W_xh) 
            + self.optimizer.reg.norm(self.W_hh) 
            + self.optimizer.reg.norm(self.W_hy))

    '''
    access to weights
    '''
    @property
    def gradient_weights(self):
        return self.W_hh
    @gradient_weights.setter
    def gradient_weights(self, Matrix):
        self.W_hh = Matrix
    
    '''
    the time steps of iteration is given by k2
    '''
    def backward(self, error_tensor):
        # init all gradient
        d_W_hy = np.zeros(self.W_hy.shape)
        d_W_hh = np.zeros(self.W_hh.shape)
        d_W_xh = np.zeros(self.W_xh.shape)
        d_B_y = np.zeros(self.B_y.shape)
        output_error_tensor = np.zeros(self.input_tensor.shape)
        
        # a inverse iteration from back to forward in sequence
        for i in range (error_tensor.shape[0]-1, -1, -1):
            d_sigmoid = self.output_tensor[i] * (1 - self.output_tensor[i])
            d_ht = np.dot(error_tensor[i] * d_sigmoid,  self.W_hy.T)
            d_W_hy += np.outer(self.ht_tensor[i], error_tensor[i] * d_sigmoid)
            d_B_y += error_tensor[i] * d_sigmoid
            
            # get dE/dW_hh and dE/dW_xh in k2 steps back propagation 
            for step in np.arange(max(0, i-self.k2), i+1)[::-1]:
                d_tanh = 1 - self.ht_tensor[step]**2
                d_W_hh += np.outer(d_ht * d_tanh, self.ht_tensor[step-1])
                d_W_xh += np.outer(self.input_tensor[step], d_ht * d_tanh)
                d_ht = np.dot(d_ht * d_tanh, self.W_hh.T)
            
            # regive d_ht to calculate d_x
            d_ht = np.dot(error_tensor[i] * d_sigmoid,  self.W_hy.T)
            d_tanh = 1 - self.ht_tensor[i]**2
            d_x = np.dot(d_ht * d_tanh, self.W_xh.T)
            output_error_tensor[i] = d_x
        
        # update the weights and bias
        if self._optimizer:
            self.W_hy = self._optimizer.calculate_update(self.W_hy, d_W_hy)
            self.B_y = self._optimizer.calculate_update(self.B_y, d_B_y)
            self.W_xh = self._optimizer.calculate_update(self.W_xh, d_W_xh)
            self.W_hh = self._optimizer.calculate_update(self.W_hh, d_W_hh)
        
        # delete the last col
        return output_error_tensor[:,:-1]







