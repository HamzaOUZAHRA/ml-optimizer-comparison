# src/optim/momentum.py
import numpy as np

def momentum_update(
    theta: np.ndarray,
    grad: np.ndarray,
    velocity: np.ndarray,
    lr: float,
    beta: float = 0.9
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs a single momentum update.

    θₜ ← θₜ₋₁ – α · vₜ
    vₜ ← β·vₜ₋₁ + (1–β)·∇ℓ(θₜ₋₁)

    Returns
    -------
    theta_new : np.ndarray
    velocity_new : np.ndarray
    """
    velocity_new = beta * velocity + (1 - beta) * grad
    theta_new   = theta - lr * velocity_new
    return theta_new, velocity_new
