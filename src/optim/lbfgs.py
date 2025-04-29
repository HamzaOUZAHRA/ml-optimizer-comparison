# src/optim/lbfgs.py
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

def lbfgs(
    func_grad,
    x0: np.ndarray,
    max_iter: int = 15000,
    tol: float = 1e-6
) -> np.ndarray:
    """
    Wrapper around L-BFGS-B to minimize a function that returns (loss, grad).
    """
    def _loss(x):
        loss, _ = func_grad(x)
        return loss

    def _grad(x):
        _, grad = func_grad(x)
        return grad

    theta_opt, _, _ = fmin_l_bfgs_b(
        func=_loss,
        x0=x0,
        fprime=_grad,
        maxiter=max_iter,
        pgtol=tol,
    )
    return theta_opt
