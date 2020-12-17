import Layers
import numpy as np
import copy

from Layers.FullyConnected import FullyConnected
from Layers.ReLU import ReLU
from Layers.SoftMax import SoftMax
from Optimization.Loss import CrossEntropyLoss
from Optimization.Optimizers import Sgd

class NeuralNetwork:
    """
    protected members
    """
    def __init__(self, Optimizer_Obj, weight_initializer, bias_initializer):
        self.optimizer = Optimizer_Obj
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.weight_init = weight_initializer
        self.bias_init = bias_initializer
    """
    forward calculation
    the structure of layers are store in list "layers"
    return loss
    """
    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        for i in self.layers:
            self.input_tensor = i.forward(self.input_tensor)
        loss_val = self.loss_layer.forward(self.input_tensor, self.label_tensor)
        self.loss.append(loss_val)
        return loss_val
    """
    backward calculation 
    """
    def backward(self):
        # input_tensor, label_tensor = self.data_layer.next()
        # get the first error_tensor from 
        error_tensor = self.loss_layer.backward(self.label_tensor)
        # remove loss_layers from rev_layer
        for i in reversed(self.layers):
            error_tensor = i.backward(error_tensor)
    """
    optimize the parameters
    layer is the FullyConnected layer
    """
    def append_trainable_layer(self, layer):
        layer.initialize(self.weight_init, self.bias_init)
        layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)
    """
    train the network
    iterations is the times of iteration
    """
    def train(self, iterations):
        for i in range(iterations):
            self.forward()
            self.backward()
    """
    test
    get the output of final layer before loss_layer
    """
    def test(self, input_tensor):
        for i in self.layers:
            input_tensor = i.forward(input_tensor)
        return input_tensor