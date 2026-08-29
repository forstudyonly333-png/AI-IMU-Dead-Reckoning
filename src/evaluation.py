import numpy as np


def rmse(predicted, actual):

    error = predicted - actual

    mse = np.mean(
        np.sum(error ** 2, axis=1)
    )

    return np.sqrt(mse)


def position_error(predicted, actual):

    error = np.linalg.norm(
        predicted - actual,
        axis=1
    )

    return error