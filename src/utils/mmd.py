import torch
import numpy as np
from src.utils.startup import exp_configs

device = exp_configs.device


def MMD(x, y, kernel='linear'):

    # Need x and y to be of same size. Sample points from the larger array, to have equal number of datapoints in both.
    if x.shape[0] > y.shape[0]:
        x = x[np.random.choice(x.shape[0], y.shape[0], replace=False), :]
    if x.shape[0] < y.shape[0]:
        y = y[np.random.choice(y.shape[0], x.shape[0], replace=False), :]

    # Cast to torch tensors
    x = torch.tensor(x) if type(x) == np.ndarray else x
    y = torch.tensor(y) if type(y) == np.ndarray else y


    if kernel == 'linear':
        X_bar = x.mean(axis=0)
        Y_bar = y.mean(axis=0)
        Z_bar = X_bar - Y_bar
        mmd2 = Z_bar.dot(Z_bar)
        return mmd2

    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50] # Change bandwidth?
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)

