"""Models used in the predictive baseline."""

from .base import TabularClassifier
from .logistic_regression import LogisticRegressionModel, initialize_parameters as initialize_logistic_parameters
from .mlp import MLPClassifierModel, initialize_parameters as initialize_mlp_parameters
from .persistence import (
    LoadedGlobalModelArtifact,
    load_global_model,
    load_global_model_parameters,
    save_global_model_parameters,
)
from .registry import (
    DEFAULT_MODEL_REGISTRY,
    ModelRegistry,
    ModelSpec,
    build_model_config,
    create_model,
    initialize_model_parameters,
)

__all__ = [
    "DEFAULT_MODEL_REGISTRY",
    "LoadedGlobalModelArtifact",
    "ModelRegistry",
    "ModelSpec",
    "build_model_config",
    "TabularClassifier",
    "LogisticRegressionModel",
    "MLPClassifierModel",
    "create_model",
    "initialize_model_parameters",
    "initialize_logistic_parameters",
    "initialize_mlp_parameters",
    "load_global_model",
    "load_global_model_parameters",
    "save_global_model_parameters",
]
