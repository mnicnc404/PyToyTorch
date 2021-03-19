import numpy as np
try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None


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

    def export(self):
        raise NotImplementedError


class Sequential(Module):
    def __init__(self, *modules):
        self.modules = list(modules)

    def append(self, module):
        self.modules.append(module)

    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x

    def backward(self, d_y):
        for module in self.modules:
            d_y = module.backward(d_y)
        return d_y

    def export(self):
        if torch is not None:
            return torch.nn.Sequential(
                *[module.export() for module in self.modules]
            )
