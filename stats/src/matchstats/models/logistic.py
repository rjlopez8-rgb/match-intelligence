"""Small, dependency-free logistic and softmax regression (NumPy only).

Kept deliberately simple and transparent: every trained model is a list of named
coefficients saved as JSON, so anyone can read exactly what the model learned.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit_logistic(
    X: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 60, weights: np.ndarray | None = None
) -> np.ndarray:
    """Binary logistic regression by Newton-Raphson (IRLS) with L2 regularisation.

    ``X`` must already contain an intercept column. The intercept is not penalised.
    """
    n, d = X.shape
    w = np.zeros(d)
    sw = np.ones(n) if weights is None else weights
    reg = np.full(d, l2)
    reg[0] = 0.0
    for _ in range(iters):
        p = sigmoid(X @ w)
        g = X.T @ ((p - y) * sw) / n + reg * w
        r = p * (1 - p) * sw
        H = (X.T * r) @ X / n + np.diag(reg + 1e-9)
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def predict_logistic(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return sigmoid(X @ w)


def softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def fit_softmax(
    X: np.ndarray, y: np.ndarray, k: int, l2: float = 1e-3, iters: int = 3000, lr: float = 0.05
) -> np.ndarray:
    """Multinomial logistic regression trained with Adam. Returns W of shape (d, k)."""
    n, d = X.shape
    W = np.zeros((d, k))
    Y = np.eye(k)[y]
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    reg = np.full((d, 1), l2)
    reg[0] = 0.0
    for t in range(1, iters + 1):
        P = softmax(X @ W)
        G = X.T @ (P - Y) / n + reg * W
        m = 0.9 * m + 0.1 * G
        v = 0.999 * v + 0.001 * G * G
        W -= lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    return W


def predict_softmax(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    return softmax(X @ W)
