# src/optim/sgd.py
import numpy as np

def sgd_update(theta: np.ndarray, grad: np.ndarray, lr: float) -> np.ndarray:
    return theta - lr * grad
