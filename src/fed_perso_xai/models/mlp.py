"""Explicit MLP classifier with NumPy CPU and optional PyTorch GPU training paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised through optional dependency path
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def initialize_parameters(
    n_features: int,
    hidden_dim: int,
    *,
    seed: int = 0,
) -> list[np.ndarray]:
    """Return seeded parameters for a one-hidden-layer binary MLP."""

    if n_features < 1:
        raise ValueError("n_features must be >= 1.")
    if hidden_dim < 1:
        raise ValueError("hidden_dim must be >= 1.")
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0.0, np.sqrt(2.0 / float(n_features)), size=(n_features, hidden_dim))
    b1 = np.zeros(hidden_dim, dtype=np.float64)
    w2 = rng.normal(0.0, np.sqrt(2.0 / float(hidden_dim)), size=(hidden_dim, 1))
    b2 = np.zeros(1, dtype=np.float64)
    return [
        np.asarray(w1, dtype=np.float64),
        b1,
        np.asarray(w2, dtype=np.float64),
        b2,
    ]


@dataclass
class MLPClassifierModel:
    """A one-hidden-layer binary MLP trained with mini-batch SGD or Adam."""

    n_features: int
    hidden_dim: int
    activation: str
    optimizer: str
    device: str
    learning_rate: float
    batch_size: int
    local_epochs: int
    l2_regularization: float = 1e-4

    def __post_init__(self) -> None:
        self.activation = self._normalize_activation(self.activation)
        self.optimizer = self._normalize_optimizer(self.optimizer)
        self.device = self._normalize_device(self.device)
        self.W1, self.b1, self.W2, self.b2 = [
            parameter.copy()
            for parameter in initialize_parameters(self.n_features, self.hidden_dim)
        ]

    def get_parameters(self) -> list[np.ndarray]:
        return [
            self.W1.copy(),
            self.b1.copy(),
            self.W2.copy(),
            self.b2.copy(),
        ]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        if len(parameters) != 4:
            raise ValueError("MLPClassifierModel expects [W1, b1, W2, b2].")
        W1 = np.asarray(parameters[0], dtype=np.float64)
        b1 = np.asarray(parameters[1], dtype=np.float64).reshape(-1)
        W2 = np.asarray(parameters[2], dtype=np.float64)
        b2 = np.asarray(parameters[3], dtype=np.float64).reshape(1)
        if W1.shape != (self.n_features, self.hidden_dim):
            raise ValueError(
                f"Expected W1 shape {(self.n_features, self.hidden_dim)}, received {W1.shape}."
            )
        if b1.shape != (self.hidden_dim,):
            raise ValueError(f"Expected b1 shape {(self.hidden_dim,)}, received {b1.shape}.")
        if W2.shape != (self.hidden_dim, 1):
            raise ValueError(f"Expected W2 shape {(self.hidden_dim, 1)}, received {W2.shape}.")
        self.W1 = W1.copy()
        self.b1 = b1.copy()
        self.W2 = W2.copy()
        self.b2 = b2.copy()

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int) -> float:
        if self.device == "gpu":
            return self._fit_torch(X, y, seed)
        return self._fit_numpy(X, y, seed)

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        if self.device == "gpu":
            return self._predict_logits_torch(X)
        _, _, logits, _ = self._forward_numpy(X)
        return logits.reshape(-1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.device == "gpu":
            return self._predict_proba_torch(X)
        _, _, _, probabilities = self._forward_numpy(X)
        return probabilities.reshape(-1)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.device == "gpu":
            return self._loss_torch(X, y)
        probabilities = np.clip(self.predict_proba(X), 1e-8, 1.0 - 1e-8)
        targets = np.asarray(y, dtype=np.float64).reshape(-1)
        data_loss = -np.mean(
            targets * np.log(probabilities) + (1.0 - targets) * np.log(1.0 - probabilities)
        )
        regularization = 0.5 * self.l2_regularization * (
            float(np.sum(self.W1**2)) + float(np.sum(self.W2**2))
        )
        return float(data_loss + regularization)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "parameter_count": np.asarray([4], dtype=np.int64),
            "parameter_0": self.W1.astype(np.float64, copy=False),
            "parameter_1": self.b1.astype(np.float64, copy=False),
            "parameter_2": self.W2.astype(np.float64, copy=False),
            "parameter_3": self.b2.astype(np.float64, copy=False),
            "n_features": np.asarray([self.n_features], dtype=np.int64),
            "hidden_dim": np.asarray([self.hidden_dim], dtype=np.int64),
            "activation": np.asarray([self.activation]),
            "optimizer": np.asarray([self.optimizer]),
            "device": np.asarray([self.device]),
            "learning_rate": np.asarray([self.learning_rate], dtype=np.float64),
            "batch_size": np.asarray([self.batch_size], dtype=np.int64),
            "local_epochs": np.asarray([self.local_epochs], dtype=np.int64),
            "l2_regularization": np.asarray([self.l2_regularization], dtype=np.float64),
        }
        np.savez_compressed(path, **payload)
        return path

    def _fit_numpy(self, X: np.ndarray, y: np.ndarray, seed: int) -> float:
        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n_samples = X.shape[0]
        batch_size = max(1, min(self.batch_size, n_samples))
        adam_state = self._initialize_adam_state() if self.optimizer == "adam" else None
        step = 0
        for _ in range(self.local_epochs):
            indices = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch_indices = indices[start : start + batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                hidden_linear, hidden_activation, _, probabilities = self._forward_numpy(X_batch)
                errors = probabilities - y_batch.reshape(-1, 1)
                grad_W2 = (hidden_activation.T @ errors) / X_batch.shape[0]
                grad_W2 += self.l2_regularization * self.W2
                grad_b2 = np.mean(errors, axis=0)
                hidden_grad = (errors @ self.W2.T) * self._activation_derivative_numpy(
                    hidden_linear,
                    hidden_activation,
                )
                grad_W1 = (X_batch.T @ hidden_grad) / X_batch.shape[0]
                grad_W1 += self.l2_regularization * self.W1
                grad_b1 = np.mean(hidden_grad, axis=0)
                if adam_state is None:
                    self.W1 -= self.learning_rate * grad_W1
                    self.b1 -= self.learning_rate * grad_b1
                    self.W2 -= self.learning_rate * grad_W2
                    self.b2 -= self.learning_rate * grad_b2
                else:
                    step += 1
                    self.W1 = self._adam_update(self.W1, grad_W1, state=adam_state["W1"], step=step)
                    self.b1 = self._adam_update(self.b1, grad_b1, state=adam_state["b1"], step=step)
                    self.W2 = self._adam_update(self.W2, grad_W2, state=adam_state["W2"], step=step)
                    self.b2 = self._adam_update(self.b2, grad_b2, state=adam_state["b2"], step=step)
        return self.loss(X, y)

    def _fit_torch(self, X: np.ndarray, y: np.ndarray, seed: int) -> float:
        torch_module = self._require_torch()
        device = self._resolve_torch_device()
        torch_module.manual_seed(int(seed))
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(int(seed))
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        X_tensor = torch_module.as_tensor(X_np, dtype=torch_module.float32, device=device)
        y_tensor = torch_module.as_tensor(y_np, dtype=torch_module.float32, device=device)
        n_samples = X_tensor.shape[0]
        batch_size = max(1, min(self.batch_size, int(n_samples)))

        W1, b1, W2, b2 = self._parameters_to_torch(device)
        params = [W1, b1, W2, b2]
        if self.optimizer == "adam":
            optimizer = torch_module.optim.Adam(
                params,
                lr=self.learning_rate,
                weight_decay=self.l2_regularization,
            )
        else:
            optimizer = torch_module.optim.SGD(
                params,
                lr=self.learning_rate,
                weight_decay=self.l2_regularization,
            )

        for _ in range(self.local_epochs):
            indices = torch_module.randperm(int(n_samples), device=device)
            for start in range(0, int(n_samples), batch_size):
                batch_indices = indices[start : start + batch_size]
                X_batch = X_tensor[batch_indices]
                y_batch = y_tensor[batch_indices]
                optimizer.zero_grad(set_to_none=True)
                logits = self._forward_torch(X_batch, W1, b1, W2, b2)
                loss = F.binary_cross_entropy_with_logits(logits, y_batch)
                loss.backward()
                optimizer.step()

        self.W1, self.b1, self.W2, self.b2 = self._parameters_from_torch(W1, b1, W2, b2)
        return self.loss(X, y)

    def _predict_logits_torch(self, X: np.ndarray) -> np.ndarray:
        torch_module = self._require_torch()
        device = self._resolve_torch_device()
        X_tensor = torch_module.as_tensor(np.asarray(X, dtype=np.float32), dtype=torch_module.float32, device=device)
        W1, b1, W2, b2 = self._parameters_to_torch(device, requires_grad=False)
        with torch_module.no_grad():
            logits = self._forward_torch(X_tensor, W1, b1, W2, b2).reshape(-1)
        return logits.detach().cpu().numpy().astype(np.float64, copy=False)

    def _predict_proba_torch(self, X: np.ndarray) -> np.ndarray:
        torch_module = self._require_torch()
        logits = torch_module.as_tensor(
            self._predict_logits_torch(X),
            dtype=torch_module.float32,
        )
        probabilities = torch_module.sigmoid(logits)
        return probabilities.detach().cpu().numpy().astype(np.float64, copy=False)

    def _loss_torch(self, X: np.ndarray, y: np.ndarray) -> float:
        torch_module = self._require_torch()
        device = self._resolve_torch_device()
        X_tensor = torch_module.as_tensor(np.asarray(X, dtype=np.float32), dtype=torch_module.float32, device=device)
        y_tensor = torch_module.as_tensor(
            np.asarray(y, dtype=np.float32).reshape(-1, 1),
            dtype=torch_module.float32,
            device=device,
        )
        W1, b1, W2, b2 = self._parameters_to_torch(device, requires_grad=False)
        with torch_module.no_grad():
            logits = self._forward_torch(X_tensor, W1, b1, W2, b2)
            data_loss = F.binary_cross_entropy_with_logits(logits, y_tensor)
            regularization = 0.5 * self.l2_regularization * (
                torch_module.sum(W1**2) + torch_module.sum(W2**2)
            )
            total_loss = data_loss + regularization
        return float(total_loss.detach().cpu().item())

    def _forward_numpy(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        features = np.asarray(X, dtype=np.float64)
        hidden_linear = features @ self.W1 + self.b1
        hidden_activation = self._apply_activation_numpy(hidden_linear)
        logits = hidden_activation @ self.W2 + self.b2[0]
        clipped_logits = np.clip(logits, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-clipped_logits))
        return hidden_linear, hidden_activation, logits, probabilities

    @staticmethod
    def _normalize_activation(value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"relu", "tanh"}:
            raise ValueError("activation must be one of: relu, tanh.")
        return normalized

    @staticmethod
    def _normalize_optimizer(value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"sgd", "adam"}:
            raise ValueError("optimizer must be one of: sgd, adam.")
        return normalized

    @staticmethod
    def _normalize_device(value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"cpu", "gpu"}:
            raise ValueError("device must be one of: cpu, gpu.")
        return normalized

    def _apply_activation_numpy(self, hidden_linear: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(hidden_linear, 0.0)
        return np.tanh(hidden_linear)

    def _activation_derivative_numpy(
        self,
        hidden_linear: np.ndarray,
        hidden_activation: np.ndarray,
    ) -> np.ndarray:
        if self.activation == "relu":
            return (hidden_linear > 0.0).astype(np.float64)
        return 1.0 - hidden_activation**2

    def _initialize_adam_state(self) -> dict[str, dict[str, np.ndarray]]:
        return {
            "W1": {"m": np.zeros_like(self.W1), "v": np.zeros_like(self.W1)},
            "b1": {"m": np.zeros_like(self.b1), "v": np.zeros_like(self.b1)},
            "W2": {"m": np.zeros_like(self.W2), "v": np.zeros_like(self.W2)},
            "b2": {"m": np.zeros_like(self.b2), "v": np.zeros_like(self.b2)},
        }

    def _adam_update(
        self,
        parameter: np.ndarray,
        gradient: np.ndarray,
        *,
        state: dict[str, np.ndarray],
        step: int,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> np.ndarray:
        grad = np.asarray(gradient, dtype=np.float64)
        state["m"] = beta1 * state["m"] + (1.0 - beta1) * grad
        state["v"] = beta2 * state["v"] + (1.0 - beta2) * (grad**2)
        m_hat = state["m"] / (1.0 - beta1**step)
        v_hat = state["v"] / (1.0 - beta2**step)
        return parameter - self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    def _require_torch(self):
        if torch is None or F is None:
            raise ImportError(
                "PyTorch is required for MLP device='gpu'. Install the optional torch dependency first."
            )
        return torch

    def _resolve_torch_device(self):
        torch_module = self._require_torch()
        if self.device != "gpu":
            return torch_module.device("cpu")
        if not torch_module.cuda.is_available():
            raise RuntimeError(
                "MLP device='gpu' was requested, but CUDA is not available in this environment."
            )
        return torch_module.device("cuda")

    def _parameters_to_torch(self, device, *, requires_grad: bool = True):
        torch_module = self._require_torch()
        W1 = torch_module.tensor(self.W1, dtype=torch_module.float32, device=device, requires_grad=requires_grad)
        b1 = torch_module.tensor(self.b1, dtype=torch_module.float32, device=device, requires_grad=requires_grad)
        W2 = torch_module.tensor(self.W2, dtype=torch_module.float32, device=device, requires_grad=requires_grad)
        b2 = torch_module.tensor(self.b2, dtype=torch_module.float32, device=device, requires_grad=requires_grad)
        return W1, b1, W2, b2

    @staticmethod
    def _parameters_from_torch(W1, b1, W2, b2) -> list[np.ndarray]:
        return [
            W1.detach().cpu().numpy().astype(np.float64, copy=True),
            b1.detach().cpu().numpy().astype(np.float64, copy=True),
            W2.detach().cpu().numpy().astype(np.float64, copy=True),
            b2.detach().cpu().numpy().astype(np.float64, copy=True),
        ]

    def _forward_torch(self, X, W1, b1, W2, b2):
        hidden_linear = X @ W1 + b1
        if self.activation == "relu":
            hidden_activation = F.relu(hidden_linear)
        else:
            hidden_activation = torch.tanh(hidden_linear)
        return hidden_activation @ W2 + b2.reshape(1)
