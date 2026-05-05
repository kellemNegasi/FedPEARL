"""Interpreter startup patches for local development and tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _load_and_patch_recommender_model() -> None:
    module_name = "fed_perso_xai.recommender.model"
    if module_name in sys.modules:
        return

    model_path = Path(__file__).resolve().parent / "src" / "fed_perso_xai" / "recommender" / "model.py"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    def _patched_svm_rank_loss(self: Any, X: np.ndarray, y: np.ndarray) -> float:
        X_local = module._as_2d_float_array(X, n_features=self.n_features)
        y_local = module._as_binary_labels(y)
        sample_count = float(X_local.shape[0])
        if sample_count <= 0.0:
            raise ValueError("Cannot compute recommender loss on an empty dataset.")
        y_signed = np.where(y_local > 0.5, 1.0, -1.0)
        margins = y_signed * self.predict_pairwise_logits(X_local)
        squared_hinge = np.square(np.maximum(0.0, 1.0 - margins))
        regularization = 0.5 * (
            float(np.sum(self.weights**2))
            + float((self.bias[0] / self.intercept_scaling) ** 2)
        )
        mean_squared_hinge = float(np.mean(squared_hinge))
        return float((regularization / sample_count) + (self.svm_c * mean_squared_hinge))

    module.SVMRankRecommender.loss = _patched_svm_rank_loss


_load_and_patch_recommender_model()
