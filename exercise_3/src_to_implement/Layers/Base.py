
'''
set a base class for all layers
'''

class BaseLayer:
    def __init__(self):

        # properties
        self.testing_phase = False
        self.optimizer_flag = False