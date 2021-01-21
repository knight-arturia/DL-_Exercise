class FullyConnected:

    def __init__(self, weights):

        self.weights = weights
    
    def forward(self, input_tensor)

        self


class RNNElement:
    
    
    def __init__(self, Optimizer_Obj, weight_initializer, bias_initializer):
        
        #layers
        self.FClayer_xt
        # object of class optimizer
        self.optimizer = Optimizer_Obj
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.weight_init = weight_initializer
        self.bias_init = bias_initializer
        # show the layer is at testing model or not
        self._phase = False

    '''
    use _phase to change testing_phase for all layers
    '''
    @property
    def phase(self):
        return self._phase
    @phase.setter
    def phase(self, value):
        self._phase = value
        for i in self.layers:
            i.testing_phase = self._phase
    
    """
    forward calculation
    the structure of layers are store in list "layers"
    return loss
    """
    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        regular_loss = 0
        for i in self.layers:
            # check if layer i has optimizer, cal regular loss for each layer
            if i.optimizer_flag:
                # check if optimizer got a regularizer
                if i.optimizer.reg:
                    regular_loss += i.optimizer.reg.norm(i.weights)
            self.input_tensor = i.forward(self.input_tensor)
        loss_val = self.loss_layer.forward(self.input_tensor, self.label_tensor) + regular_loss
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
    layer is the FullyConnected layer or conv layer
    """
    def append_trainable_layer(self, layer):
        # layer BatchNormalization need a different initialization
        if isinstance(layer, BatchNormalization):
            layer.initialize(layer.channel)
        else:
            layer.initialize(self.weight_init, self.bias_init)
        layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)
    """
    train the network
    iterations is the times of iteration
    """
    def train(self, iterations):
        self.phase = False
        for i in range(iterations):
            self.forward()
            self.backward()
    """
    test
    get the output of final layer before loss_layer
    """
    def test(self, input_tensor):
        self.phase = True
        for i in self.layers:
            input_tensor = i.forward(input_tensor)
        return input_tensor