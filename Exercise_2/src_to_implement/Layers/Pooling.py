import numpy as np
from numpy.core.defchararray import index
from numpy.core.numeric import zeros_like

class Pooling:
    
    def __init__(self, stride_shape, Pooling_shape):

        self.filter_shape = Pooling_shape

        self.stri_h = stride_shape[0] if type(stride_shape) == tuple else stride_shape
        self.stri_w = stride_shape[1] if type(stride_shape) == tuple else stride_shape
    
    def forward(self, input_tensor):
        
        # padding input with valid method
        pad_input = np.pad(input_tensor, (
                (0, 0), (0,0), (self.filter_shape[0] - 1, self.filter_shape[0] - 1), 
                (self.filter_shape[1] - 1, self.filter_shape[1] - 1)), 'constant', constant_values=0)
        self.input = input_tensor
        self.pad_in = pad_input

        batch, channels, height, width = pad_input.shape
        filter_h, filter_w = self.filter_shape[0], self.filter_shape[1]
        
        out_h = int(1 + (height - filter_h) / self.stri_h)
        out_w = int(1 + (width - filter_w) / self.stri_w)
        output = np.zeros((batch, channels, out_h, out_w))
        
        self.index = zeros_like(pad_input)

        for b in range(batch):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        # get a masked input_tensor in choose out i and j on axis 2 and 3
                        x_masked = pad_input[:, :, i * self.stri_h : i * self.stri_h + filter_h, 
                            j * self.stri_w : j * self.stri_w + filter_w]
                        output[:, :, i, j] = np.max(x_masked, axis=(2,3))
                        # get all the position of 
                        max_index_h = np.argmax(pad_input[b, c, i * self.stri_h : i * self.stri_h + filter_h, 
                            j * self.stri_w : j * self.stri_w + filter_w], axis= 2)
                        max_index_w = np.argmax(pad_input[b, c, i * self.stri_h : i * self.stri_h + filter_h, 
                            j * self.stri_w : j * self.stri_w + filter_w], axis= 3)
                        self.index[b, c, i * self.stri_h : i * self.stri_h + max_index_h // filter_h , 
                            j * self.stri_w : j * self.stri_w + max_index_w] = 1

        return output

    def backward(self, error_tensor):
        
        max_array = self.pad_in * self.index
        cut_h = self.filter_shape[0] - 1
        cut_w = self.filter_shape[1] - 1

        return max_array[:, :, cut_h : cut_h + self.input.shape[2], cut_w : cut_w + self.input.shape[3]]