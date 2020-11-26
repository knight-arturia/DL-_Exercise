import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.batch_size = None
        self.input_tensor = None
    # return loss function
    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        # if input_tensor is a one-dimension vector
        if input_tensor.ndim == 1:
            label_tensor = label_tensor.reshape(1, label_tensor.size)
            input_tensor = input_tensor.reshape(1, input_tensor.size)
        # find out the number of output vector in this batch
        self.batch_size = input_tensor.shape[0]
        # print(self.batch_size)
        # cal loss function
        Loss = -np.sum(label_tensor * np.log(input_tensor + np.finfo(float).eps))
        return Loss
    # return error_tensor
    def backward(self, label_tensor):
        error_tensor = - np.true_divide(label_tensor, self.input_tensor)
        return error_tensor
