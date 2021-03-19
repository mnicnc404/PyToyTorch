from ..nn.module import Module


class Optim:
    def __init__(self, module, lr):
        assert isinstance(module, Module)
        self.lr = lr
        self.params = module.params
        self.grads = module.grads
        assert len(self.params) == len(self.grads)


# TODO: momentum
class SGD(Optim):
    def __init__(self, module, lr, momentum=False):
        super().__init__(module, lr)
        assert not momentum, "momentum not supported yet"
        self.momentum = momentum

    def update(self):
        for param, grad in zip(self.params, self.grads):
            param -= grad * self.lr
            grad.fill(0)
