import numpy as np
try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None


class Module:
    params = []
    grads = []

    def build_params(self, params):
        self.saved_input = None
        self.params = list(params)
        self.grads = list(np.zeros_like(p) for p in self.params)

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
        for module in modules:
            assert isinstance(module, Module)
        self.modules = list(modules)
        self.params = [param for m in modules for param in m.params]
        self.grads = [grad for m in modules for grad in m.grads]

    def append(self, module):
        assert isinstance(module, Module)
        self.modules.append(module)
        self.params.extend(module.params)
        self.grads.extend(module.grads)

    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x

    def backward(self, d_y):
        for module in reversed(self.modules):
            d_y = module.backward(d_y)
        return d_y

    def export(self):
        if torch is not None:
            return torch.nn.Sequential(
                *[module.export() for module in self.modules]
            )
