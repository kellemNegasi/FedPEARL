"""Compatibility shim for recommender model exports.

This package re-exports the legacy module implementation and patches the
reported SVM-rank loss so it matches the sample-normalized objective used by
mini-batch SGD updates.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np


_LEGACY_MODEL_PATH = Path(__file__).resolve().parents[1] / "model.py"
_LEGACY_SPEC = importlib.util.spec_from_file_location(
    "fed_perso_xai.recommender._legacy_model",
    _LEGACY_MODEL_PATH,
)
if _LEGACY_SPEC is None or _LEGACY_SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"Could not load recommender model implementation from {_LEGACY_MODEL_PATH}.")
_legacy_model = importlib.util.module_from_spec(_LEGACY_SPEC)
_LEGACY_SPEC.loader.exec_module(_legacy_model)


def _patched_svm_rank_loss(self: Any, X: np.ndarray, y: np.ndarray) -> float:
    X = _legacy_model._as_2d_float_array(X, n_features=self.n_features)
    y = _legacy_model._as_binary_labels(y)
    sample_count = float(X.shape[0])
    if sample_count <= 0.0:
        raise ValueError("Cannot compute recommender loss on an empty dataset.")
    y_signed = np.where(y > 0.5, 1.0, -1.0)
    margins = y_signed * self.predict_pairwise_logits(X)
    squared_hinge = np.square(np.maximum(0.0, 1.0 - margins))
    regularization = 0.5 * (
        float(np.sum(self.weights**2))
        + float((self.bias[0] / self.intercept_scaling) ** 2)
    )
    mean_squared_hinge = float(np.mean(squared_hinge))
    return float((regularization / sample_count) + (self.svm_c * mean_squared_hinge))


_legacy_model.SVMRankRecommender.loss = _patched_svm_rank_loss


DEFAULT_RECOMMENDER_TYPE = _legacy_model.DEFAULT_RECOMMENDER_TYPE
SUPPORTED_RECOMMENDER_TYPES = _legacy_model.SUPPORTED_RECOMMENDER_TYPES
PairwiseLogisticConfig = _legacy_model.PairwiseLogisticConfig
PairwiseLogisticRecommender = _legacy_model.PairwiseLogisticRecommender
SVMRankRecommender = _legacy_model.SVMRankRecommender
PairwiseRecommenderModel = _legacy_model.PairwiseRecommenderModel
normalize_recommender_type = _legacy_model.normalize_recommender_type
recommender_artifact_model_type = _legacy_model.recommender_artifact_model_type
initialize_recommender_parameters = _legacy_model.initialize_recommender_parameters
create_recommender = _legacy_model.create_recommender
load_pairwise_logistic_recommender = _legacy_model.load_pairwise_logistic_recommender
load_svm_rank_recommender = _legacy_model.load_svm_rank_recommender
load_recommender = _legacy_model.load_recommender


__all__ = [
    "DEFAULT_RECOMMENDER_TYPE",
    "SUPPORTED_RECOMMENDER_TYPES",
    "PairwiseLogisticConfig",
    "PairwiseLogisticRecommender",
    "SVMRankRecommender",
    "PairwiseRecommenderModel",
    "normalize_recommender_type",
    "recommender_artifact_model_type",
    "initialize_recommender_parameters",
    "create_recommender",
    "load_pairwise_logistic_recommender",
    "load_svm_rank_recommender",
    "load_recommender",
]
