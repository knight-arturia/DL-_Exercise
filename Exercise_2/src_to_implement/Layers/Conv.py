import numpy as np
import math
from numpy.core.fromnumeric import shape

class Conv:

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
        self.bias = np.random.uniform(0, 1, size=(1, num_kernels))
        self.gradient_weights = None
        self.gradient_bias = None
    
    """
    optimizer for weights and bias will be separate
    """
    @property
    def optimizer_w(self):
        return self._optimizer_w
    @optimizer_w.setter
    def optimizer_w(self, Obj):
        self._optimizer_w = Obj
    
    @property
    def optimizer_b(self):
        return self._optimizer_b
    @optimizer_b.setter
    def optimizer_b(self, Obj):
        self._optimizer_b = Obj
    

    def im2col(self, input, f_h, f_w, out_h, out_w):
        # input_tensor.shape = (N, C, H, W)
        # input_col.shape = (out_h * out_w * batch, channels * filter_h * filter_w )
        N, C, H, W = input.shape

        col = np.zeros((N, C, f_h, f_w, out_h, out_w))

        for y in range(f_h):
            y_max = y + self.stri_h*out_h
            for x in range(f_w):
                x_max = x + self.stri_w*out_w
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
        
        self.input_shape_origin = input_tensor.shape
        
        # add a width dimension to 3D input_tensor
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor[..., np.newaxis]
        # print("Input_shape = ", input_tensor.shape)

        # print("stride_h = ", self.stri_h, "stride_w = ", self.stri_w)

        num, channels, filter_h, filter_w = self.weights.shape
        
        # SAME padding
        pad_h1 = int(math.floor((filter_h - 1)/2))
        pad_h2 = int(math.ceil((filter_h - 1)/2))
        pad_w1 = int(math.floor((filter_w - 1)/2))
        pad_w2 = int(math.ceil((filter_w - 1)/2))
        input_tensor = np.pad(input_tensor, (
            (0, 0), (0, 0), (pad_h1, pad_h2), (pad_w1, pad_w2)),
            'constant', constant_values=0)
        
        # print("Input_shape_padded = ", input_tensor.shape)

        
        batch, channels, height, width = input_tensor.shape

        self.input_shape = input_tensor.shape
        self.input_channel = channels
        # cal output shape first; padding is 0
        out_h = int((height - filter_h) / self.stri_h) + 1
        out_w = int((width - filter_w) / self.stri_w) + 1
        
        # transform the input_tensor to a large 2D matrix
        self.input_col = self.im2col(input_tensor, filter_h, filter_w, out_h, out_w)
        
        # print("input_col = ", self.input_col.shape)
        
        # tranform filter to a matrix; transit to multiply
        weight_col = self.weights.reshape(self.kern_num, -1).T
        
        # print("weight_col = ", weight_col.shape)

        # add bias to the end of weight_col
        weight_col_bias = np.r_[weight_col, self.bias]

        # add one col 1 to the end of input_col
        bias_input = np.ones((self.input_col.shape[0], 1))
        input_plus = np.c_[self.input_col, bias_input]
        
        output = np.dot(input_plus, weight_col_bias)
        output = output.reshape(batch, out_h, out_w, -1).transpose(0, 3, 1, 2)
        self.out_shape = output.shape
        
        # if input_tensor is 3D, then change output back to 3D
        if len(self.input_shape_origin) == 3:
            output = np.squeeze(output, axis=3)
        
        print("output = ", output.shape)

        return output
    
    def backward(self, error_tensor):
        # add a x axis for 3D error_tensor
        if len(error_tensor.shape) == 3:
            error_tensor = error_tensor[..., np.newaxis]
        
        # error_tensor.shape(batch, kern_num, out_h, out_w)
        self.err = error_tensor
        self.err_shape = error_tensor.shape

        # col_err.shape(out_h*out_w*batch , kern_num)
        col_err = np.reshape(error_tensor.transpose(0, 2, 3, 1), [-1, self.kern_num])
        # cal gradient of weights and bias
        self.gradient_weights = np.dot(self.input_col.T, col_err).reshape(self.weights.shape)
        self.gradient_bias = np.sum(col_err, axis=0)

        num, channels, filter_h, filter_w = self.weights.shape
        # deconv of same padded error_tensor with flippd kernel to get next_error
        pad_h1 = int(math.floor((filter_h - 1)/2))
        pad_h2 = int(math.ceil((filter_h - 1)/2))
        pad_w1 = int(math.floor((filter_w - 1)/2))
        pad_w2 = int(math.ceil((filter_w - 1)/2))
        pad_err = np.pad(self.err, (
            (0, 0), (0,0), (pad_h1, pad_h2), (pad_w1, pad_w2)),
            'constant', constant_values=0)
        
        # exchange axis height and width in self.weights; self.weights.shape = (n, c, h, w)
        flip_weights = self.weights.reshape([self.kern_num, self.input_channel, -1])
        flip_weights = flip_weights[:, :, ::-1]
        flip_weights = flip_weights.swapaxes(0,1)
        # col_flip_weight.shape = (f_h * f_w * kern_num, input_channel)
        col_flip_weights = flip_weights.reshape(self.input_channel, -1).T
        print("flip_weight = ", col_flip_weights.shape)
        
        out_err_h = int((pad_err.shape[2] - filter_h) / self.stri_h) + 1
        out_err_w = int((pad_err.shape[3] - filter_w) / self.stri_w) + 1

        col_pad_err=self.im2col(pad_err, filter_h, filter_w, out_err_h, out_err_w)
        print("col_pad_err = ", col_pad_err.shape)
        next_err = np.dot(col_pad_err, col_flip_weights)
        next_err = np.reshape(next_err, self.input_shape)
        
        # update parameters in optimizers
        if self._optimizer_w:
            self.weights = self._optimizer_w.calculate_update(self.weights, self.gradient_weights)
        if self._optimizer_b:
            self.bias = self._optimizer_b.calculate_update(self.bias, self.gradient_bias)
        
        return next_err

    def initialize(self, weight_initializer, bias_initializer):
        fan_in = self.weight_shape[1] * self.weight_shape[2] * self.weight_shape[3]
        fan_out = self.weight_shape[0] * self.weight_shape[2] * self.weight_shape[3]
        
        self.weights = weight_initializer.initialize(self.weight_shape, fan_in, fan_out)

        self.bias = bias_initializer.initialize((1, self.kern_num), fan_in, fan_out)