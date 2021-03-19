import numpy as np
try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None


from .module import Module


class FC(Module):
    def __init__(self, in_size, out_size, bias=True):
        self.bias = bias
        # NOTE: weight
        # NOTE: clip to avoid nan
        params = [
            np.random.randn(
                in_size, out_size
            ).astype(np.float32).clip(-1., 1.)
        ]
        if bias:
            # NOTE: bias
            params.append(
                np.random.randn(
                    1, out_size
                ).astype(np.float32).clip(-1, 1)
            )
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

    def export(self):
        # NOTE: grads are not being copied
        if torch is not None:
            in_size, out_size = self.params[0].shape
            m = torch.nn.Linear(in_size, out_size, self.bias)
            with torch.no_grad():
                m.weight.copy_(torch.from_numpy(self.params[0].T))
                if self.bias:
                    m.bias.copy_(torch.from_numpy(self.params[1].squeeze(0)))
            return m
