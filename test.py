import torch
import numpy as np
import PyToyTorch as ptt
from sklearn import datasets
from tqdm import tqdm


def rmse(x, y):
    return np.sqrt(np.power(x - y, 2).mean())


def cal_accuracy(scores, Y, threshold=0.5):
    pred = scores > threshold
    same = pred == Y
    if torch.is_tensor(same):
        return same.float().mean()
    else:
        return same.astype(np.float32).mean()


def main():
    np.random.seed(7777)
    torch.manual_seed(7777)
    toy_dataset = datasets.load_breast_cancer()
    X_all = toy_dataset['data'].astype(np.float32)
    X_all = (X_all - X_all.min()) / X_all.max()  # To avoid overflow
    Y_all = np.expand_dims(toy_dataset['target'], 1).astype(np.float32)
    first_size = X_all.shape[1]
    eval_idxs = np.random.choice(
        X_all.shape[0], X_all.shape[0] // 10, replace=False
    )
    mask = np.zeros(X_all.shape[0], np.bool)
    mask[eval_idxs] = True
    X_eval = X_all[mask]
    X = X_all[~mask]
    Y_eval = Y_all[mask]
    Y = Y_all[~mask]
    TX_eval = torch.from_numpy(X_eval)
    TY_eval = torch.from_numpy(Y_eval)

    b_size = 8
    n_epoch = 2000
    lr = 0.1
    ptt_model = ptt.nn.Sequential(
        ptt.nn.FC(first_size, 25),
        ptt.nn.Mish(),
        ptt.nn.ResFC(25),
        ptt.nn.Mish(),
        ptt.nn.FC(25, 1),
        ptt.nn.Sigmoid(),
    )
    ptt_model.append(ptt.nn.FC(1, 1))
    ptt_model.append(ptt.nn.Sigmoid())
    out = ptt_model(X_eval)
    torch_model = ptt_model.export()
    tout = torch_model(TX_eval)
    ptt_old_acc = cal_accuracy(out, Y_eval)
    pytorch_old_acc = cal_accuracy(tout, TY_eval)
    print('ptt eval accuracy before training: {}'.format(ptt_old_acc))
    print('pytorch eval accuracy before training: {}'.format(
        pytorch_old_acc))
    print('output diff: {}'.format(rmse(out, tout.detach().numpy())))

    crit = ptt.loss.BCE()
    t_crit = torch.nn.BCELoss()
    optim = ptt.optim.SGD(ptt_model, lr)
    t_optim = torch.optim.SGD(torch_model.parameters(), lr, momentum=0.)
    with tqdm(range(1, 1 + n_epoch)) as pbar:
        for e in tqdm(range(1, 1 + n_epoch)):
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
            pbar.set_postfix(
                ptt_loss="{:.4f}".format(loss),
                pytorch_loss="{:.4f}".format(t_loss.item())
            )
    out = ptt_model(X_eval)
    tout = torch_model(TX_eval)
    ptt_new_acc = cal_accuracy(out, Y_eval)
    pytorch_new_acc = cal_accuracy(tout, TY_eval)
    print('ptt eval accuracy after training: {}'.format(ptt_new_acc))
    print('pytorch eval accuracy after training: {}'.format(
        pytorch_new_acc))
    print('output diff: {}'.format(rmse(out, tout.detach().numpy())))


if __name__ == '__main__':
    main()
