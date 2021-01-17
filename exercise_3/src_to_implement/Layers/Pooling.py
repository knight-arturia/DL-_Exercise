from Layers.Base import BaseLayer
import numpy as np
import math
from numpy.core.defchararray import index
from numpy.core.fromnumeric import shape
from numpy.core.numeric import zeros_like

class Pooling(BaseLayer):
    
    def __init__(self, stride_shape, Pooling_shape):
        # init father class Baslayer
        super(BaseLayer, self).__init__()

        self.filter_shape = Pooling_shape

        self.stri_h = stride_shape[0]
        self.stri_w = stride_shape[1] if type(stride_shape) == tuple else 1
    
    def im2col(self, input_data, Pool_h, Pool_w):
        
        N, C, H, W = input_data.shape

        out_h = (H - Pool_h)//self.stri_h + 1
        out_w = (W - Pool_w)//self.stri_w + 1

        img = input_data
        col = np.zeros((N, C, Pool_h, Pool_w, out_h, out_w))

        for y in range(Pool_h):
            y_max = y + self.stri_h * out_h
            for x in range(Pool_w):
                x_max = x + self.stri_w * out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:self.stri_h, x:x_max:self.stri_w]
        
        # col.shape = (N*out_h*out_w*C, Pool_h*Pool_w)
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
        return col

    def forward(self, input_tensor):
        
        input_shape_origin = input_tensor.shape

        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor[..., np.newaxis]
        
        # extended input shape
        input_shape_extend = input_tensor.shape

        batch, channels, height, width = input_shape_extend
        Pool_h, Pool_w = self.filter_shape[0], self.filter_shape[1]
        
        # each row in input_col matrix is a filter area
        input_col = self.im2col(input_tensor, Pool_h, Pool_w)
        input_col = input_col.reshape(-1, Pool_h*Pool_w)

        # valid padding means no padded
        out_h = int(1 + (height - Pool_h) / self.stri_h)
        out_w = int(1 + (width - Pool_w) / self.stri_w)
        
        # get the position of max value 
        arg_max = np.argmax(input_col, axis=1)
        
        output = np.max(input_col, axis=1)
        output = output.reshape(batch, out_h, out_w, channels).transpose(0, 3, 1, 2)
        
        if len(input_shape_origin) == 3:
            output = np.squeeze(output, axis=3)

        # transpoe self values in forward
        self.input_shape_origin = input_shape_origin
        self.input_shape_extend = input_shape_extend
        self.pos_max = arg_max

        return output

    def col2im(self, col, input_shape, filter_h, filter_w):
    
        N, C, H, W = input_shape

        out_h = (H - filter_h)//self.stri_h + 1
        out_w = (W - filter_w)//self.stri_w + 1

        # print("reshape = ", N, ",",out_h, ",",out_w, ",",C, ",",filter_h, ",",filter_w)

        col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
        
        # add the padding to image
        img = np.zeros((N, C, H , W ))
        
        for y in range(filter_h):
            y_max = y + self.stri_h * out_h
            for x in range(filter_w):
                x_max = x + self.stri_w * out_w
                img[:, :, y:y_max:self.stri_h, x:x_max:self.stri_w] += col[:, :, y, x, :, :]
        
        return img
    

    def backward(self, error_tensor):

        err_shape_origin = error_tensor.shape

        if len(error_tensor.shape) == 3:
            error_tensor = error_tensor[..., np.newaxis]
        
        error_tensor = error_tensor.transpose(0, 2, 3, 1)
        # pool_size is Pool_h*Pool_w 
        Pool_h, Pool_w = self.filter_shape
        pool_size = Pool_h * Pool_w
        
        # init max value matrix
        dmax = np.zeros((error_tensor.size, pool_size))
        # put the max value back to its position; pos_max.shape = (N*out_h*out_w, 1)
        dmax[np.arange(self.pos_max.size), self.pos_max.flatten()] = error_tensor.flatten()

        # dmax.shape = (N, H, W, C, Pool_h*Pool_w)
        dmax = dmax.reshape(error_tensor.shape + (pool_size, ))

        err_col = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        next_err = self.col2im(err_col, self.input_shape_extend, Pool_h, Pool_w)

        if len(err_shape_origin) == 3:
            next_err = np.squeeze(next_err, axis=3)

        return next_err