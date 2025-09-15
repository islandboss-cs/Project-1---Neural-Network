import numpy as np

# Mean Squared Error (MSE)
def f_mse(real, prediction):
    return np.mean((real - prediction) ** 2)

def f_mse_der(real, prediction):
    return 2 * (prediction - real) / real.size

# MSE with factor 2m
def f_mse2(real, prediction):
    return np.mean((real - prediction) ** 2) / 2

def f_mse2_der(real, prediction):
    return (prediction - real) / real.size