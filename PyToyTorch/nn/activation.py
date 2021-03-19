"""
There certainly are several more clever ways to implement backward
for activations and other basic operators... but the code here
works anyways
"""
import numpy as np
try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None

from .module import Module


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _softplus(x):
    return np.log(np.exp(x) + 1)


class BaseActivation(Module):
    def self(self):
        self.saved_output = None

    def act(self, x):
        raise NotImplementedError

    def d_act(self, x):
        raise NotImplementedError

    def forward(self, x):
        self.saved_input = x
        self.saved_output = self.act(x)
        return self.saved_output

    def backward(self, d_y):
        return self.d_act(self.saved_input) * d_y


class ReLU(BaseActivation):
    def act(self, x):
        res = x.copy()
        res[res <= 0] = 0.
        return res

    def d_act(self, x):
        res = x.copy()
        res[res > 0] = 1.
        res[res <= 0] = 0.
        return res

    def export(self):
        if torch is not None:
            return torch.nn.ReLU()


class LeakyReLU(BaseActivation):
    def __init__(self, slope=0.01):
        self.slope = slope

    def act(self, x):
        res = x.copy()
        res[res <= 0] *= self.slope
        return res

    def d_act(self, x):
        res = x.copy()
        res[res > 0] = 1.
        res[res <= 0] = self.slope
        return res

    def export(self):
        if torch is not None:
            return torch.nn.LeakyReLU(self.slope)


class Tanh(BaseActivation):
    def act(self, x):
        return np.tanh(x)

    def d_act(self, x):
        return 1 - np.power(self.saved_output, 2)

    def export(self):
        if torch is not None:
            return torch.nn.Tanh()


class Sigmoid(BaseActivation):
    def act(self, x):
        return _sigmoid(x)

    def d_act(self, x):
        return self.saved_output * (1 - self.saved_output)

    def export(self):
        if torch is not None:
            return torch.nn.Sigmoid()


class Softplus(BaseActivation):
    def act(self, x):
        return _softplus(x)

    def d_act(self, x):
        return _sigmoid(x)

    def export(self):
        if torch is not None:
            return torch.nn.Softplus()


class Mish(BaseActivation):
    def act(self, x):
        return x * np.tanh(_softplus(x))

    def d_act(self, x):
        # NOTE: tanh_s_x can be saved as grad buffer
        tanh_s_x = np.tanh(_softplus(x))
        return tanh_s_x + x * (1 - np.power(tanh_s_x, 2)) * _sigmoid(x)

    def export(self):
        if torch is not None:
            class _TorchMish(torch.nn.Module):
                def forward(self, x):
                    return x * torch.nn.functional.softplus(x).tanh()
            return _TorchMish()
