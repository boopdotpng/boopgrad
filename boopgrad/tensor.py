import numpy as np

class Tensor:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self._graph = None
    # tensor operations are not evaluated until you call realize()
    def realize():
        pass
    # return a new tensor, detached from autograd graph
    def detach(self): return Tensor(self.data)
    # ** methods that generate new tensors ** 
    @staticmethod
    # a tensor of all zeros
    def zeros(*dims):
        pass
    @staticmethod
    # the default way to initialize linear layers
    def kaiming_uniform(*dims):
        pass
