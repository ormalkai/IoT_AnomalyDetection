import numpy as np


def calc_mse(x_mat, x_mat_hat, axis=1):
    return (np.square(x_mat - x_mat_hat)).mean(axis=axis)
