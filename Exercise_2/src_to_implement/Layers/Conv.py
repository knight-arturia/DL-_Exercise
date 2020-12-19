import numpy as np
import math
import copy
from numpy.core.fromnumeric import shape

class Conv:
    
    # a Optimizer Object
    _optimizer = False
    _optimizer_w = False
    _optimizer_b = False

    def __init__(self, stride_shape, convolution_shape, num_kernels):

        # get stride at row and col direction
        self.stri_h = stride_shape[0]
        self.stri_w = stride_shape[1] if type(stride_shape) == tuple else 1
        
        # add a third dimension 1 to Conv_shape
        if len(convolution_shape) == 2:
            convolution_shape = convolution_shape + (1,)
        
        # insert filter batch in to weight shape; weight_shape(kern_num, channel, height, width)
        self.weight_shape = (num_kernels,) + convolution_shape
        
        # print(self.weight_shape)

        self.weights = np.random.uniform(0, 1, size=self.weight_shape)
        self.kern_num = num_kernels
        self.bias = np.random.uniform(0, 1, size=num_kernels)
        self.gradient_weights = None
        self.gradient_bias = None
    
    """
    optimizer for weights and bias will be separate
    """
    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, Obj):
        self._optimizer = Obj
        self._optimizer_b = copy.deepcopy(self._optimizer)
        self._optimizer_w = copy.deepcopy(self._optimizer)

    def im2col(self, input, f_h, f_w, out_h, out_w):
        # input_tensor.shape = (N, C, H, W)
        # input_col.shape = (out_h * out_w * batch, channels * filter_h * filter_w )
        N, C, H, W = input.shape

        col = np.zeros((N, C, f_h, f_w, out_h, out_w))

        for y in range(f_h):
            y_max = y + self.stri_h * out_h
            for x in range(f_w):
                x_max = x + self.stri_w * out_w
                col[:, :, y, x, :, :] = input[:, :, y:y_max:self.stri_h, x:x_max:self.stri_w]
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
        return col

    """
    Transpos Variables in forward function
    self.input_shape_origin : origin shape of input_tensor
    self.input_shape : shape of input_tensor after padding
    self.out_shape : shape of output
    """
    def forward(self, input_tensor):
        
        # origin input shape
        self.input_shape_origin = input_tensor.shape
        # add a width dimension to 3D input_tensor
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor[..., np.newaxis]
        
        # extended input shape
        self.input_shape_extend = input_tensor.shape

        num, channels, filter_h, filter_w = self.weights.shape
        
        # SAME padding
        pad_h1 = int(math.floor((filter_h - 1)/2))
        pad_h2 = int(math.ceil((filter_h - 1)/2))
        pad_w1 = int(math.floor((filter_w - 1)/2))
        pad_w2 = int(math.ceil((filter_w - 1)/2))
        input_tensor = np.pad(input_tensor, ((0, 0), (0, 0), (pad_h1, pad_h2), (pad_w1, pad_w2)),
            'constant', constant_values=0)
        
        batch, channels, height, width = input_tensor.shape

        # padded input shape
        self.input_shape = input_tensor.shape
        self.input_channel = channels
        
        # cal output shape first; height and weight already padded
        out_h = int((height - filter_h) / self.stri_h) + 1
        out_w = int((width - filter_w) / self.stri_w) + 1
        
        # transform the input_tensor to a large 2D matrix
        self.input_col = self.im2col(input_tensor, filter_h, filter_w, out_h, out_w)
        
        # tranform filter to a matrix; transit to multiply
        self.weight_col = self.weights.reshape(self.kern_num, -1).T

        # add bias to the end of weight_col
        weight_col_bias = np.vstack((self.weight_col, self.bias))
        
        # add one col 1 to the end of input_col
        bias_input = np.ones((self.input_col.shape[0], 1))
        input_col_plus = np.c_[self.input_col, bias_input]
        
        output = np.dot(input_col_plus, weight_col_bias)
        output = output.reshape(batch, out_h, out_w, -1).transpose(0, 3, 1, 2)
        self.out_shape = output.shape
        
        # if input_tensor is 3D, then change output back to 3D
        if len(self.input_shape_origin) == 3:
            output = np.squeeze(output, axis=3)
        
        return output
    
    """
    Transfer 2D back to 4D 
    """
    
    def col2im(self, col, input_shape, filter_h, filter_w):
    
        N, C, H, W = input_shape

        pad_h1 = int(math.floor((filter_h - 1)/2))
        pad_h2 = int(math.ceil((filter_h - 1)/2))
        pad_w1 = int(math.floor((filter_w - 1)/2))
        pad_w2 = int(math.ceil((filter_w - 1)/2))

        out_h = (H + pad_h1 + pad_h2 - filter_h)//self.stri_h + 1
        out_w = (W + pad_w1 + pad_w2 - filter_w)//self.stri_w + 1

        # print("reshape = ", N, ",",out_h, ",",out_w, ",",C, ",",filter_h, ",",filter_w)

        col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
        
        # add the padding to image
        img = np.zeros((N, C, H + pad_h1 + pad_h2 + self.stri_h - 1, W + pad_w1 + pad_w2 + self.stri_w - 1))
        
        for y in range(filter_h):
            y_max = y + self.stri_h * out_h
            for x in range(filter_w):
                x_max = x + self.stri_w * out_w
                img[:, :, y:y_max:self.stri_h, x:x_max:self.stri_w] += col[:, :, y, x, :, :]
        
        # remove the pad from output
        return img[:, :, pad_h1:H + pad_h1, pad_w1:W + pad_w1]


    def backward(self, error_tensor):
        
        self.err_shape_origin = error_tensor.shape
        
        # add a x axis for 3D error_tensor
        if len(error_tensor.shape) == 3:
            error_tensor = error_tensor[..., np.newaxis]
        
        # error_tensor.shape(batch, kern_num, out_h, out_w)
        self.err = error_tensor
        

        # col_err.shape(out_h*out_w*batch , kern_num)
        col_err = error_tensor.transpose(0, 2, 3, 1).reshape(-1, self.kern_num)
        # cal gradient of weights and bias
        self.gradient_weights = np.dot(self.input_col.T, col_err)
        self.gradient_weights = self.gradient_weights.transpose(1, 0).reshape(self.weights.shape)
        self.gradient_bias = np.sum(col_err, axis=0)

        # cal grad of input in col form
        d_input_col = np.dot(col_err, self.weight_col.T)
        
        # transform col form back to image form
        num, channels, filter_h, filter_w = self.weights.shape
        next_err = self.col2im(d_input_col, self.input_shape_extend, filter_h, filter_w)
        
        # update parameters in optimizers
        if self._optimizer_w:
            self.weights = self._optimizer_w.calculate_update(self.weights, self.gradient_weights)
        if self._optimizer_b:
            self.bias = self._optimizer_b.calculate_update(self.bias, self.gradient_bias)
        
        if len(self.err_shape_origin) == 3:
            next_err = np.squeeze(next_err, axis=3)
        
        return next_err

    def initialize(self, weight_initializer, bias_initializer):
        fan_in = self.weight_shape[1] * self.weight_shape[2] * self.weight_shape[3]
        fan_out = self.weight_shape[0] * self.weight_shape[2] * self.weight_shape[3]
        
        self.weights = weight_initializer.initialize(self.weight_shape, fan_in, fan_out)

        self.bias = bias_initializer.initialize((1, self.kern_num), fan_in, fan_out)