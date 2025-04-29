import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from optim.sgd import sgd_update
from optim.momentum import momentum_update
from optim.adam import adam_update
from optim.lbfgs import lbfgs

# ── Activations ────────────────────────────────────────────────────────────────

def relu(Z: np.ndarray) -> np.ndarray:
    return np.maximum(0, Z)

def drelu(Z: np.ndarray) -> np.ndarray:
    return (Z > 0).astype(float)

def sigmoid(Z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-Z))

def dsigmoid(A: np.ndarray) -> np.ndarray:
    return A * (1 - A)

# ── MLP Class ──────────────────────────────────────────────────────────────────

class MLP:
    def __init__(
        self,
        layer_sizes: list[int],
        optimizer: str = "gd",
        lr: float = 0.01,
        max_iter: int = 500,
        tol: float = 1e-6,
        beta: float = 0.9
    ):
        """
        layer_sizes: sizes of layers, e.g. [30, 16, 1].
        optimizer: 'gd', 'momentum', 'adam', or 'lbfgs'.
        """
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1
        self.optimizer = optimizer
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.beta = beta

        self.params = {}
        self.velocity = {}
        self.m = {}
        self.v = {}
        self.loss_history = []
        # storage for packing/unpacking
        self.shapes = {}
        self.sizes = {}
        self.total_size = 0

        self._init_params()

    def _init_params(self):
        # initialize weights and optimizer state
        for l in range(1, self.L + 1):
            n_in = self.layer_sizes[l - 1]
            n_out = self.layer_sizes[l]
            self.params[f"W{l}"] = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
            self.params[f"b{l}"] = np.zeros((1, n_out))
            self.velocity[f"W{l}"] = np.zeros((n_in, n_out))
            self.velocity[f"b{l}"] = np.zeros((1, n_out))
            self.m[f"W{l}"] = np.zeros((n_in, n_out))
            self.m[f"b{l}"] = np.zeros((1, n_out))
            self.v[f"W{l}"] = np.zeros((n_in, n_out))
            self.v[f"b{l}"] = np.zeros((1, n_out))

        # record shapes and sizes for L-BFGS
        self.shapes = {k: v.shape for k, v in self.params.items()}
        self.sizes = {k: v.size for k, v in self.params.items()}
        self.total_size = sum(self.sizes.values())

    def _pack(self) -> np.ndarray:
        """Flatten all parameters into a single 1-D array."""
        return np.concatenate([self.params[k].ravel() for k in self.params])

    def _unpack(self, theta: np.ndarray):
        """Unpack 1-D array back into parameter matrices."""
        idx = 0
        for k in self.params:
            size = self.sizes[k]
            shape = self.shapes[k]
            self.params[k] = theta[idx:idx+size].reshape(shape)
            idx += size

    def _forward(self, X: np.ndarray) -> np.ndarray:
        self.cache = {"A0": X}
        A = X
        for l in range(1, self.L + 1):
            W = self.params[f"W{l}"]
            b = self.params[f"b{l}"]
            Z = A.dot(W) + b
            self.cache[f"Z{l}"] = Z
            if l < self.L:
                A = relu(Z)
            else:
                A = sigmoid(Z)
            self.cache[f"A{l}"] = A
        return A

    def _compute_loss(self, A_last: np.ndarray, y: np.ndarray) -> float:
        m = y.shape[0]
        eps = 1e-8
        return -np.mean(
            y.reshape(-1,1) * np.log(A_last + eps) +
            (1 - y).reshape(-1,1) * np.log(1 - A_last + eps)
        )

    def _backward(self, y: np.ndarray) -> dict:
        grads = {}
        m = y.shape[0]
        # output layer gradient
        AL = self.cache[f"A{self.L}"]  # shape (m, 1)
        dZ = AL - y.reshape(-1, 1)
        for l in reversed(range(1, self.L + 1)):
            A_prev = self.cache[f"A{l-1}"]
            grads[f"dW{l}"] = (A_prev.T.dot(dZ)) / m
            grads[f"db{l}"] = np.sum(dZ, axis=0, keepdims=True) / m
            if l > 1:
                W = self.params[f"W{l}"]
                dA_prev = dZ.dot(W.T)
                Z_prev = self.cache[f"Z{l-1}"]
                dZ = dA_prev * drelu(Z_prev)
        return grads

    def _update_params(self, grads: dict, t: int):
        for l in range(1, self.L + 1):
            Wk, bk = f"W{l}", f"b{l}"
            dW, db = grads[f"d{Wk}"], grads[f"d{bk}"]
            if self.optimizer == "gd":
                self.params[Wk] = sgd_update(self.params[Wk], dW, self.lr)
                self.params[bk] = sgd_update(self.params[bk], db, self.lr)
            elif self.optimizer == "momentum":
                self.params[Wk], self.velocity[Wk] = momentum_update(
                    self.params[Wk], dW, self.velocity[Wk], lr=self.lr, beta=self.beta
                )
                self.params[bk], self.velocity[bk] = momentum_update(
                    self.params[bk], db, self.velocity[bk], lr=self.lr, beta=self.beta
                )
            elif self.optimizer == "adam":
                self.params[Wk], self.m[Wk], self.v[Wk] = adam_update(
                    self.params[Wk], dW, self.m[Wk], self.v[Wk], t,
                    lr=self.lr, beta1=self.beta, beta2=0.999, eps=1e-8
                )
                self.params[bk], self.m[bk], self.v[bk] = adam_update(
                    self.params[bk], db, self.m[bk], self.v[bk], t,
                    lr=self.lr, beta1=self.beta, beta2=0.999, eps=1e-8
                )
            else:
                raise ValueError(f"Unknown optimizer '{self.optimizer}'")

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.loss_history = []
        # L-BFGS branch
        if self.optimizer == 'lbfgs':
            theta0 = self._pack()
            self.loss_history = []
            def fg(theta):
                self._unpack(theta)
                AL = self._forward(X)
                loss = self._compute_loss(AL, y)
                grads = self._backward(y)
                grad_theta = np.concatenate([grads[f'd{k}'].ravel() for k in self.params])
                self.loss_history.append(loss)
                return loss, grad_theta
            theta_opt = lbfgs(fg, theta0, max_iter=self.max_iter, tol=self.tol)
            self._unpack(theta_opt)
            return

        # SGD/momentum/adam branches
        for t in range(1, self.max_iter + 1):
            AL = self._forward(X)
            loss = self._compute_loss(AL, y)
            self.loss_history.append(loss)
            grads = self._backward(y)
            self._update_params(grads, t)
            if t > 1 and abs(self.loss_history[-2] - loss) < self.tol:
                break

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


# ── Script to run MLP ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    proc = os.path.join(base, "data", "processed")
    X_train = np.load(os.path.join(proc, "X_train.npy"))
    y_train = np.load(os.path.join(proc, "y_train.npy"))
    X_test  = np.load(os.path.join(proc, "X_test.npy"))
    y_test  = np.load(os.path.join(proc, "y_test.npy"))

    layer_sizes = [X_train.shape[1], 16, 1]
    for opt in ["gd", "momentum", "adam", "lbfgs"]:
        print(f"\n>>> Training MLP with {opt.upper()} <<<")
        mlp = MLP(layer_sizes, optimizer=opt, lr=0.01, max_iter=1000, tol=1e-6, beta=0.9)
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        y_prob = mlp.predict_proba(X_test)
        print(f"{opt:8} Acc: {accuracy_score(y_test, y_pred):.4f} AUC: {roc_auc_score(y_test, y_prob):.4f}")