import torch
import numpy as np
import PyToyTorch as ptt
from sklearn import datasets


def rmse(x, y):
    return np.sqrt(np.power(x - y, 2).mean())


def main():
    toy_dataset = datasets.load_breast_cancer()
    X = toy_dataset['data'].astype(np.float32)
    X = (X - X.min()) / X.max()
    Y = np.expand_dims(toy_dataset['target'], 1).astype(np.float32)
    first_size = X.shape[1]

    b_size = 16
    n_epoch = 100
    lr = 0.01
    ptt_model = ptt.nn.Sequential(
        ptt.nn.FC(first_size, 25),
        ptt.nn.Mish(),
        ptt.nn.ResFC(25),
        ptt.nn.Mish(),
        ptt.nn.FC(25, 1),
        ptt.nn.Sigmoid(),
    )
    torch_model = ptt_model.export()

    crit = ptt.loss.BCE()
    t_crit = torch.nn.BCELoss()
    optim = ptt.optim.SGD(ptt_model, lr)
    t_optim = torch.optim.SGD(torch_model.parameters(), lr, momentum=0.)
    for e in range(1, 1 + n_epoch):
        idxs = np.random.choice(X.shape[0], b_size, replace=False)
        batch_x = X[idxs]
        batch_y = Y[idxs]
        out = ptt_model(batch_x)
        loss = crit(out, batch_y)
        loss_grad = crit.backward(out, batch_y)
        ptt_model.backward(loss_grad)
        optim.update()  # NOTE: grads cleaned here
        t_batch_x = torch.from_numpy(batch_x)
        t_batch_y = torch.from_numpy(batch_y)
        t_out = torch_model(t_batch_x)
        t_loss = t_crit(t_out, t_batch_y)
        t_loss.backward()
        t_optim.step()
        t_optim.zero_grad()
        print(
            loss,
            t_loss.detach().item(),
            'diff: ',
            rmse(out, t_out.detach().numpy()),
            rmse(loss, t_loss.detach().numpy()),
        )


if __name__ == '__main__':
    main()
