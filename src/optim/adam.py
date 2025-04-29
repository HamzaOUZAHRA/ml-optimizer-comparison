# src/optim/adam.py
import numpy as np

def adam_update(
    theta: np.ndarray,
    grad: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    t: int,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs a single Adam update.

    mₜ ← β1·mₜ₋₁ + (1–β1)·gₜ
    vₜ ← β2·vₜ₋₁ + (1–β2)·gₜ²
    m̂ₜ ← mₜ / (1–β1ᵗ)
    v̂ₜ ← vₜ / (1–β2ᵗ)
    θₜ ← θₜ₋₁ – α·m̂ₜ/(√v̂ₜ + ε)

    Returns
    -------
    theta_new : np.ndarray
    m_new     : np.ndarray
    v_new     : np.ndarray
    """
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)
    theta_new = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
    return theta_new, m_new, v_new
