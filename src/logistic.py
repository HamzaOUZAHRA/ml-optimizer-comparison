import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from optim.lbfgs import lbfgs
from optim.momentum import momentum_update
from optim.adam import adam_update


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def compute_loss_and_grad(theta: np.ndarray, X: np.ndarray, y: np.ndarray):
    m, n = X.shape
    w = theta[:n]
    b = theta[n]
    z = X.dot(w) + b
    h = sigmoid(z)
    eps = 1e-8
    loss = -np.mean(y * np.log(h + eps) + (1 - y) * np.log(1 - h + eps))
    dw = (1 / m) * X.T.dot(h - y)
    db = (1 / m) * np.sum(h - y)
    grad = np.concatenate([dw, [db]])
    return loss, grad


class LogisticRegression:
    def __init__(
        self,
        optimizer: str = 'gd',
        lr: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
        beta: float = 0.9
    ):
        self.optimizer = optimizer
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.beta = beta
        self.w = None
        self.b = None
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        m, n = X.shape
        # Pack θ = [w, b]
        theta = np.zeros(n + 1)
        self.loss_history = []

        if self.optimizer == 'gd':
            for i in range(self.max_iter):
                loss, grad = compute_loss_and_grad(theta, X, y)
                self.loss_history.append(loss)
                theta -= self.lr * grad
                if np.linalg.norm(grad) < self.tol:
                    break

        elif self.optimizer == 'momentum':
            velocity = np.zeros_like(theta)
            for i in range(self.max_iter):
                loss, grad = compute_loss_and_grad(theta, X, y)
                self.loss_history.append(loss)
                theta, velocity = momentum_update(
                    theta, grad, velocity, lr=self.lr, beta=self.beta
                )
                if np.linalg.norm(grad) < self.tol:
                    break

        elif self.optimizer == 'adam':
            m_t = np.zeros_like(theta)
            v_t = np.zeros_like(theta)
            for t in range(1, self.max_iter + 1):
                loss, grad = compute_loss_and_grad(theta, X, y)
                self.loss_history.append(loss)
                theta, m_t, v_t = adam_update(
                    theta, grad, m=m_t, v=v_t, t=t,
                    lr=self.lr, beta1=self.beta, beta2=0.999
                )
                if np.linalg.norm(grad) < self.tol:
                    break

        elif self.optimizer == 'lbfgs':
            theta0 = np.zeros(n + 1)

            def fg(th):
                loss, grad = compute_loss_and_grad(th, X, y)
                self.loss_history.append(loss)
                return loss, grad

            theta = lbfgs(
                func_grad=fg,
                x0=theta0,
                max_iter=self.max_iter,
                tol=self.tol
            )

        else:
            raise ValueError(f"Unknown optimizer '{self.optimizer}'")

        # unpack
        self.w = theta[:n]
        self.b = theta[n]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X.dot(self.w) + self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


if __name__ == '__main__':
    # load data
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    proc = os.path.join(base, 'data', 'processed')
    X_train = np.load(os.path.join(proc, 'X_train.npy'))
    y_train = np.load(os.path.join(proc, 'y_train.npy'))
    X_test  = np.load(os.path.join(proc, 'X_test.npy'))
    y_test  = np.load(os.path.join(proc, 'y_test.npy'))

    for opt in ['gd', 'momentum', 'adam', 'lbfgs']:
        print(f"\n>>> Training with {opt.upper()} <<<")
        lr = 0.1 if opt in ('gd', 'momentum', 'adam') else None
        model = LogisticRegression(optimizer=opt, lr=lr, max_iter=500)
        model.fit(X_train, y_train)
        print(f"{opt:8} Acc: {accuracy_score(y_test, model.predict(X_test)):.4f} "
              f"AUC: {roc_auc_score(y_test, model.predict_proba(X_test)):.4f}")
