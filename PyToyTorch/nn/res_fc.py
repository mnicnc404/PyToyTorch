try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None

from .module import Module
from .fc import FC


class ResFC(Module):
    def __init__(self, size):
        # NOTE: input and output should be the same
        self.fc = FC(size, size)

    def forward(self, x):
        return x + self.fc.forward(x)

    def backward(self, d_y):
        return d_y + self.fc.backward(d_y)

    def export(self):
        if torch is not None:
            class _TorchResFC(torch.nn.Module):
                def __init__(self, fc):
                    self.fc = fc

                def forward(self, x):
                    return x + self.fc(x)

            m_fc = self.fc.export()
            m = _TorchResFC(m_fc)
            return m
