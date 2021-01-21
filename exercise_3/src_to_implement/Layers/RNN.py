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
        self.k2 = 3
        self.seq_flag = False
        # init weights and bias
        self.W_y = np.random.uniform(0, 1, size=(hidden_size, output_size))
        self.B_y = np.random.uniform(0, 1, size=output_size)
        # W_hx is a combination of W_xh and W_hh
        self.W_hx = np.random.uniform(0, 1, size=(input_size+hidden_size, hidden_size))
        self.B_hx = np.random.uniform(0, 1, size=hidden_size)
    
    '''
    to initialize the W_hx and bias
    '''
    def initialize(self, weight_initializer, bias_initializer):
        # self.W_xh = weight_initializer.initialize(self.W_xh.shape, self.input_size+1, self.hidden_size)
        # self.W_hh = weight_initializer.initialize(self.W_hh.shape, self.hidden_size, self.hidden_size)
        self.W_hx = weight_initializer.initialize(self.W_hx.shape, self.W_hx.shape[0], self.hidden_size)
        self.B_hx = bias_initializer.initialize(self.B_hx.shape, self.W_hx.shape[0], self.hidden_size)
        self.W_y = weight_initializer.initialize(self.W_y.shape, self.hidden_size, self.output_size)
        self.B_y = bias_initializer.initialize(self.B_y.shape, self.hidden_size, self.output_size)
    
    '''
    memorize is how many steps we need to memorize, k2
    '''
    @property
    def memorize(self):
        return self.seq_flag
    @memorize.setter
    def memorize(self, value):
        self.seq_flag = value
    
    def forward(self, input_tensor):

        output_tensor = np.zeros((input_tensor.shape[0], self.output_size))
        ht_tensor = np.zeros((input_tensor.shape[0], self.hidden_size))
        for i in range(input_tensor.shape[0]):
            # if sequence flag is false means a new start of back propagation sequence 
            if self.seq_flag == False:
                self.hidden_state = np.zeros(self.hidden_size)
                # set the h_t-1 back to 0
                # ht_tensor[i-1] = 0
                self.seq_flag = True
            
            # attach input vector with hidden state
            state_plus_input = np.hstack((self.hidden_state, input_tensor[i]))
            ht = np.tanh(np.dot(state_plus_input, self.W_hx) + self.B_hx)
            # save h_t for backward propagation
            ht_tensor[i] = ht
            self.hidden_state = ht
            y1 = np.dot(self.hidden_state, self.W_y) + self.B_y
            output_tensor[i] = 1 / (1 + np.exp(-y1))
        
        # store the input, output, ht for the backpropagation
        self.input_tensor = input_tensor
        self.ht_tensor = ht_tensor
        self.output_tensor = output_tensor

        return output_tensor
    
    '''
    give the optimizer for W_hx and bias
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
        return (self.optimizer.reg.norm(self.W_hx) 
            + self.optimizer.reg.norm(self.W_y))

    '''
    access to W_hx
    '''
    @property
    def gradient_weights(self):
        return self.W_hx
    @gradient_weights.setter
    def gradient_weights(self, Matrix):
        self.W_hx = Matrix
    
    @property
    def weights(self):
        return self.W_hx
    @weights.setter
    def weights(self, Matrix):
        self.W_hx = Matrix
    
    
    '''
    the time steps of iteration is given by k2
    '''
    def backward(self, error_tensor):
        
        print('error_shape = ', error_tensor.shape)
        
        # init all gradient
        d_W_y = np.zeros(self.W_y.shape)
        d_B_y = np.zeros(self.B_y.shape)
        d_W_hx = np.zeros(self.W_hx.shape)
        d_B_hx = np.zeros(self.B_hx.shape)
        
        d_hx = np.zeros(self.hidden_size+self.input_size)
        
        output_error_tensor = np.zeros(self.input_tensor.shape)
        
        # a inverse iteration from back to forward in sequence
        for i in np.arange(error_tensor.shape[0])[::-1]:
            
            d_sigmoid = self.output_tensor[i] * (1 - self.output_tensor[i])
            e_h2 = np.dot(error_tensor[i] * d_sigmoid,  self.W_y.T)

            gradient_W_y = np.outer(self.ht_tensor[i], error_tensor[i] * d_sigmoid)
            gradient_B_y = error_tensor[i] * d_sigmoid

            d_tanh = 1 - self.ht_tensor[i] ** 2
            e_h1 = e_h2 * d_tanh

            d_W_xh = self.input_tensor[i].reshape((-1, 1))
            d_W_hh = self.ht_tensor[i-1].reshape((-1, 1))
            gradient_W_xh = np.outer(d_W_xh.T, e_h1)
            gradient_W_hh = np.outer(d_W_hh.T, e_h1)
            print("gradient_W_xh shape: ", gradient_W_xh.shape)
            print("gradient_W_hh shape: ", gradient_W_hh.shape)


            state_plus_input = np.hstack((self.ht_tensor[i - 1], self.input_tensor[i]))
            d_W_hx += np.outer(state_plus_input, d_ht * d_tanh)
            W_hh = self.W_hx[0:self.hidden_size]
            d_B_hx += d_ht * d_tanh
            d_hx += np.dot(d_ht * d_tanh, self.W_hx.T)
            d_ht = np.dot(d_ht * d_tanh, W_hh.T)
        return None
        '''
            # ready for the back propagation to k2 steps
            # cut W_hx and get a W_hh, for gradient d_ht  
            W_hh = self.W_hx[0:self.hidden_size]


                
                # d_W_hh += np.outer(d_ht * d_tanh, self.ht_tensor[step-1])
                # d_W_xh += np.outer(self.input_tensor[step], d_ht * d_tanh)
                

                d_hx += np.dot(d_ht * d_tanh, self.W_hx.T)

                # update d_ht to step-1
                d_ht = np.dot(d_ht * d_tanh, W_hh.T)

            d_x = d_hx[self.hidden_size: self.hidden_size+self.input_size]
            output_error_tensor[i] = d_x
            
        # update the W_hx and bias
        if self._optimizer:
            self.W_y = self._optimizer.calculate_update(self.W_y, d_W_y)
            self.B_y = self._optimizer.calculate_update(self.B_y, d_B_y)
            self.W_hx = self._optimizer.calculate_update(self.W_hx, d_W_hx)
            self.B_hx = self._optimizer.calculate_update(self.B_hx, d_B_hx)
        
        # delete the last col
        return output_error_tensor
        '''







