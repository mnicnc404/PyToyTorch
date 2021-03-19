import numpy as np


class BaseLoss:
    def __init__(self, reduction='mean'):
        assert reduction in ('mean', 'sum'), \
                "assert reduction in (\'mean\', \'sum\')"
        self.reduction = reduction

    def forward(pred, target):
        raise NotImplementedError

    def backward(pred, target):
        raise NotImplementedError

    def __call__(self, pred, target):
        return self.forward(pred, target)


# NOTE: it should not be called "mean" se if reduction == sum, but anyways
class MSE(BaseLoss):
    def forward(self, pred, target):
        se = np.power(pred - target, 2)
        return se.sum() if self.reduction == 'sum' else se.mean()

    def backward(self, pred, target):
        d_diff = 2 * (pred - target)
        return d_diff if self.reduction == 'sum' else d_diff / target.size


class BCE(BaseLoss):
    def forward(self, pred, target):
        e = -(target * np.log(pred) + (1 - target) * np.log(1 - pred))
        return e.sum() if self.reduction == 'sum' else e.mean()

    def backward(self, pred, target):
        e = (-target / pred + (1 - target) / (1 - pred))
        return e if self.reduction == 'sum' else e / target.size
