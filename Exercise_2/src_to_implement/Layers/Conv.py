import numpy as np
from numpy.core.fromnumeric import shape

class Conv:

    _optimizer_w = False
    _optimizer_b = False

    def __init__(self, stride_shape, convolution_shape, num_kernels):

        # get stride at row and col direction
        self.stri_h = stride_shape[0] if type(stride_shape) == tuple else stride_shape
        self.stri_w = stride_shape[1] if type(stride_shape) == tuple else stride_shape
        
        if len(convolution_shape.shape) == 2:
            convolution_shape.insert(2, 1)
        # insert filter batch in to weight shape; weight_shape(kern_num, channel, height, width)
        self.weight_shape = convolution_shape.insert(0, num_kernels)
        self.weights = np.random.uniform(0, 1, size=self.weight_shape)
        self.kern_num = num_kernels
        self.bias = None
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
    

    def im2col(self, input, w, h, f_w, f_h):
        # input_col.shape = (out_h * out_w * batch, channels * filter_h * filter_w )
        input_col = []
        for b in range(input.shape[0]):
            for i in range(0, w - f_w + 1, self.stri_w):
                for j in range(0, h - f_h + 1, self.stri_h):
                    col = input[:, i:i + f_w, j:j + f_h, :].reshape([-1])
                    input_col.append(col)
        input_col = np.array(input_col)
        return input_col

    def forward(self, input_tensor):
        
        input_tensor = np.pad(input_tensor, (
            (0, 0), (0, 0), (self.weight_shape[2] // 2, self.weight_shape[2] // 2), (self.weight_shape[3] // 2, self.weight_shape[3] // 2)),
            'constant', constant_values=0)
        
        channels, filter_h, filter_w = self.weights.shape
        batch, channels, height, width = input_tensor.shape

        self.input_shape = input_tensor.shape
        self.input_channel = channels
        # cal output shape first; padding is 0
        out_h = int(1 + (height - filter_h) / self.stri_h)
        out_w = int(1 + (width - filter_w) / self.stri_w)
        
        # transform the input_tensor to a large 2D matrix
        self.input_col = self.im2col(input_tensor, width, height, filter_w, filter_h)
        
        # tranform filter to a matrix; transit to multiply
        weight_col = self.weights.reshape(self.kern_num, -1).T
        
        # initialize bias here as a col
        self.bias = np.random.uniform(0, 1, size=(batch * out_h * out_w, 1))
        self.bias_shape = self.bias.shape

        output = np.dot(self.input_col, weight_col) + self.bias
        output = output.reshape(batch, out_h, out_w, -1).transpose(0, 3, 1, 2)
        self.out_shape = output.shape

        return output
    
    def backward(self, error_tensor):

        # error_tensor.shape(batch, kern_num, out_h, out_w)
        self.err = error_tensor.transpose(0, 2, 3, 1)
        # col_err.shape(out_h*out_w*batch , kern_num)
        col_err = np.reshape(error_tensor.transpose(0, 2, 3, 1), [-1, self.kern_num])
        # cal gradient of weights and bias
        self.gradient_weights = np.dot(self.input_col.T, col_err).reshape(self.weights.shape)
        self.gradient_bias = np.sum(col_err, axis=0)

        # deconv of same padded error_tensor with flippd kernel to get next_error
        pad_err = np.pad(self.err, (
            (0, 0), (self.weight_shape[2] // 2, self.weight_shape[2] // 2), (self.weight_shape[3] // 2, self.weight_shape[3] // 2)), (0,0),
            'constant', constant_values=0)
        
        # exchange axis height and width in self.weights; self.weights.shape = (n, c, h, w)
        flip_weights = self.weights.reshape([self.kern_num, self.input_channel, -1])
        flip_weights = flip_weights[:, :, ::-1]
        col_flip_weights = flip_weights.reshape(self.kern_num, -1).T
        
        col_pad_err=self.im2col(pad_err, error_tensor.shape[3], error_tensor.shape[2], self.weight_shape[3], self.weight_shape[2])
        next_err = np.dot(col_pad_err, col_flip_weights)
        next_err = np.reshape(next_err, self.input_shape)
        
        # update parameters in optimizers
        if self._optimizer_w:
            self.weights = self._optimizer_w.calculate_update(self.weights, self.gradient_weights)
        if self._optimizer_b:
            self.bias = self._optimizer_b.calculate_update(self.bias, self.gradient_bias)
        
        return next_err

    def initialize(self, weight_initializer, bias_initializer):
        fan_in = self.input_shape[1] * self.input_shape[2] * self.input_shape[3]
        fan_out = self.out_shape[1] * self.out_shape[2] * self.out_shape[3]
        self.weights = weight_initializer.initialize(self.weight_shape, fan_in, fan_out)

        self.bias = bias_initializer.initialize(self.bias_shape, fan_in, fan_out)