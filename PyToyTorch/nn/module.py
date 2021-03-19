import numpy as np


class Module:
    def build_params(self, params):
        self.saved_input = None
        self.params = tuple(params)
        self.grads = tuple(np.zeros_like(p) for p in self.params)

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)
