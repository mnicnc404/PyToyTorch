# PyToyTorch

PyToyTorch, or ptt, a toy neural network framework which depends merely on numpy.

If you have pytorch installed, you can export ptt model to pytorch model (but no one would do that apparently).

This project is only an exercise. Do not rely this on your project.


# Installation

```
git clone https://github.com/mnicnc404/PyToyTorch.git
cd PyToyTorch
python setup.py install

# NOTE: to run test(s) in test/, you shoulf have pytorch and sklearn installed.
```

# Example


```
>>> import PyToyTorch as ptt
>>> import numpy as np
>>> model = ptt.nn.Sequential(
...     ptt.nn.FC(4, 5),
...     ptt.nn.Mish(),
...     ptt.nn.FC(5, 1),
...     ptt.nn.Sigmoid(),
... )
>>> x = np.random.rand(2, 4).astype(np.float32)
>>> out = model(x)
>>> out
array([[0.7493436],
       [0.8790313]], dtype=float32)


# train with ptt
>>> y = np.array([[1], [0]], dtype=np.float32)
>>> crit = ptt.loss.BCE()
>>> optim = ptt.optim.SGD(model, 0.1)
>>> out = model(x)
>>> loss = crit(out, y)
>>> loss
1.2003906
>>> grad = crit.backward(out, y)
>>> model.backward(grad)
# optim.update() also cleans grad, so no need to clean grad manually
>>> optim.update()  
# see test_with_pytorch.py for training phase example


# if you have pytorch installed:
>>> pytorch_model = model.export()
>>> pytorch_model
Sequential(
  (0): Linear(in_features=4, out_features=5, bias=True)
  (1): _TorchMish()
  (2): Linear(in_features=5, out_features=1, bias=True)
  (3): Sigmoid()
)
>>> tout = pytorch_model(tx)
>>> tout
tensor([[0.7493],
        [0.8790]], grad_fn=<SigmoidBackward>)
>>> out
array([[0.7493436],
       [0.8790313]], dtype=float32)
>>> (np.power(out - tout.detach().numpy(), 2)).mean()
0.0
```
