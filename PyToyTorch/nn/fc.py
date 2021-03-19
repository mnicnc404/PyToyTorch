import numpy as np

from .module import Module


class FC(Module):
    def __init__(self, in_size, out_size, bias=True):
        self.bias = bias
        # NOTE: weight
        params = [
            np.random.randn(in_size, out_size).astype(np.float32)
        ]
        if bias:
            # NOTE: bias
            params.append(np.random.randn(1, out_size))
        self.build_params(params)

    def forward(self, x):
        self.saved_input = x
        res = x.dot(self.params[0])
        if self.bias:
            res += self.params[1]
        return res

    def backward(self, d_y):
        previous_d_y = d_y.dot(self.params[0].T)
        self.grads[0][:] = self.saved_input.T.dot(d_y)
        if self.bias:
            self.grads[1][:] = d_y.sum(0, keepdims=True)
        return previous_d_y
