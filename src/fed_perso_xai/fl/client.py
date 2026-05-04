"""Flower client adapters for the federated baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import logging
import numpy as np

FLOWER_IMPORT_ERROR_MESSAGE = (
    "Flower support is not installed. Install the optional federated extras with "
    "`pip install -e .[fl]` for debug runtime support or `pip install -e .[ray]` "
    "for Ray-backed simulation."
)

try:
    import flwr as fl
except ImportError:  # pragma: no cover - exercised via optional dependency paths
    fl = None  # type: ignore[assignment]

from fed_perso_xai.evaluation.metrics import compute_classification_metrics
from fed_perso_xai.models import create_model
from fed_perso_xai.recommender import (
    DEFAULT_RECOMMENDER_TYPE,
    PairwiseLogisticConfig,
    create_recommender,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClientData:
    """Local client arrays used in training and evaluation."""

    client_id: int
    X_train: np.ndarray
    y_train: np.ndarray
    row_ids_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    row_ids_test: np.ndarray

    def get_split(self, split_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return one local split in a consistent order."""

        normalized = split_name.strip().lower()
        if normalized in {"train", "local_train", "client_local_train"}:
            return self.X_train, self.y_train, self.row_ids_train
        if normalized in {"test", "local_test", "client_local_test"}:
            return self.X_test, self.y_test, self.row_ids_test
        raise ValueError(f"Unsupported client split '{split_name}'. Expected 'train' or 'test'.")


@dataclass(frozen=True)
class RecommenderClientData:
    """Local pairwise arrays and held-out context for recommender FL."""

    client_id: int
    client_name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_eval: np.ndarray
    y_eval: np.ndarray


@dataclass(frozen=True)
class SharedParameterPayload:
    """Subset of model parameters that participates in server aggregation.

    The current baseline has no personalized server-excluded tensors yet, so all model
    parameters are currently treated as shared/global. The helper is kept
    explicit so later iterations can leave local tensors on the client.
    """

    shared_parameters: list[np.ndarray]
    shared_parameter_indices: tuple[int, ...]
    total_parameter_count: int


@dataclass(frozen=True)
class SecureAggregationClientSpec:
    """Client-local secure aggregation settings for encoded helper-share output."""

    enabled: bool = False
    num_helpers: int = 5
    privacy_threshold: int = 2
    reconstruction_threshold: int | None = None
    field_modulus: int = 2_147_483_647
    quantization_scale: int = 1 << 16
    seed: int = 0


@dataclass(frozen=True)
class ClusteredRecommenderClientUpdate:
    """Client-side clustered training outputs consumed by the coordinator."""

    client_id: str
    num_examples: int
    train_loss: float
    encoded_model_update: Any
    weighted_payload_max_abs: float
    raw_clustering_vector: np.ndarray | None = None


SECURE_PAYLOAD_ENCODING_KEY = "secure_payload_encoding"
SECURE_PAYLOAD_ENCODING_VALUE = "lcc_helper_shares_v1"
SECURE_PAYLOAD_LAYOUT_KEY = "secure_payload_layout"
SECURE_WEIGHTED_PAYLOAD_MAX_ABS_KEY = "secure_weighted_payload_max_abs"


def extract_shared_parameter_payload(parameters: list[np.ndarray]) -> SharedParameterPayload:
    """Return the model tensors that should be aggregated by the server."""

    normalized = [np.asarray(parameter, dtype=np.float64).copy() for parameter in parameters]
    return SharedParameterPayload(
        shared_parameters=normalized,
        shared_parameter_indices=tuple(range(len(normalized))),
        total_parameter_count=len(normalized),
    )


def apply_shared_parameter_payload(
    current_parameters: list[np.ndarray],
    shared_parameters: list[np.ndarray],
    shared_parameter_indices: tuple[int, ...] | None = None,
) -> list[np.ndarray]:
    """Merge aggregated shared tensors back into a full local model state."""

    merged = [np.asarray(parameter, dtype=np.float64).copy() for parameter in current_parameters]
    indices = shared_parameter_indices or tuple(range(len(shared_parameters)))
    if len(shared_parameters) != len(indices):
        raise ValueError("shared_parameters and shared_parameter_indices must align.")
    if len(indices) > len(merged):
        raise ValueError("shared_parameter_indices exceeds the local parameter count.")

    for index, parameter in zip(indices, shared_parameters, strict=True):
        merged[index] = np.asarray(parameter, dtype=np.float64).copy()
    return merged


def build_secure_aggregation_client_spec(
    training_config: Any,
    *,
    force_enabled: bool = False,
) -> SecureAggregationClientSpec:
    """Build the secure client spec from a training config-like object."""

    return SecureAggregationClientSpec(
        enabled=bool(force_enabled or getattr(training_config, "secure_aggregation", False)),
        num_helpers=int(getattr(training_config, "secure_num_helpers")),
        privacy_threshold=int(getattr(training_config, "secure_privacy_threshold")),
        reconstruction_threshold=getattr(training_config, "secure_reconstruction_threshold"),
        field_modulus=int(getattr(training_config, "secure_field_modulus")),
        quantization_scale=int(getattr(training_config, "secure_quantization_scale")),
        seed=int(getattr(training_config, "secure_seed")),
    )


def serialize_secure_payload_layout(layout: Any) -> str:
    """Serialize `lcc-lib` flattened tensor layout metadata into a metrics-safe string."""

    shapes = [list(entry.shape) for entry in getattr(layout, "entries")]
    return json.dumps(shapes, separators=(",", ":"))


def deserialize_secure_payload_layout(payload: str) -> Any:
    """Deserialize secure payload layout metadata produced by `serialize_secure_payload_layout`."""

    from lcc_lib.aggregation.flattening import FlattenedTensorLayout, TensorLayoutEntry

    shape_rows = json.loads(payload)
    entries = tuple(
        TensorLayoutEntry(
            shape=tuple(int(value) for value in shape_row),
            size=int(np.prod(shape_row, dtype=np.int64)),
        )
        for shape_row in shape_rows
    )
    return FlattenedTensorLayout(entries=entries)


def _build_client_secure_encoder(spec: SecureAggregationClientSpec) -> Any:
    from lcc_lib.aggregation import ClientPayloadEncoder, SecureAggregationConfig
    from lcc_lib.coding.field_ops import FieldConfig
    from lcc_lib.coding.share_codec import ShareEncodingConfig
    from lcc_lib.quantization.quantizer import QuantizationConfig

    return ClientPayloadEncoder(
        SecureAggregationConfig(
            field_config=FieldConfig(modulus=spec.field_modulus),
            quantization=QuantizationConfig(
                field_modulus=spec.field_modulus,
                scale=spec.quantization_scale,
            ),
            encoding=ShareEncodingConfig(
                num_helpers=spec.num_helpers,
                privacy_threshold=spec.privacy_threshold,
                reconstruction_threshold=spec.reconstruction_threshold,
                seed=spec.seed,
            ),
            compute_mean=False,
        )
    )


def compute_weighted_payload_max_abs(
    parameters: list[np.ndarray],
    weight: int | float,
) -> float:
    """Return a safe per-client absolute bound for the weighted secure payload."""

    scaled = [np.asarray(parameter, dtype=np.float64) * float(weight) for parameter in parameters]
    if not scaled:
        raise ValueError("parameters must contain at least one tensor.")
    max_abs = max(float(np.max(np.abs(parameter))) for parameter in scaled)
    if not np.isfinite(max_abs):
        raise ValueError("weighted secure payload must be finite.")
    return max_abs


if fl is not None:

    class FederatedLogisticRegressionClient(fl.client.NumPyClient):
        """Flower NumPy client backed by the explicit NumPy logistic regression model."""

        def __init__(
            self,
            data: ClientData,
            model_name: str,
            model_config: Any,
            seed: int,
            prediction_threshold: float = 0.5,
            secure_aggregation: SecureAggregationClientSpec | None = None,
        ) -> None:
            self.data = data
            self.seed = seed
            self.prediction_threshold = float(prediction_threshold)
            self._secure_aggregation = secure_aggregation or SecureAggregationClientSpec()
            self._secure_encoder = (
                _build_client_secure_encoder(self._secure_aggregation)
                if self._secure_aggregation.enabled
                else None
            )
            self.model = create_model(
                model_name,
                n_features=data.X_train.shape[1],
                config=model_config,
            )

        def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
            return extract_shared_parameter_payload(self.model.get_parameters()).shared_parameters

        def fit(
            self,
            parameters: list[np.ndarray],
            config: dict[str, Any],
        ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
            merged_parameters = apply_shared_parameter_payload(
                self.model.get_parameters(),
                parameters,
            )
            self.model.set_parameters(merged_parameters)
            train_loss = self.model.fit(
                self.data.X_train,
                self.data.y_train,
                seed=self.seed + self.data.client_id,
            )
            shared_payload = extract_shared_parameter_payload(self.model.get_parameters())
            metrics: dict[str, Any] = {
                "train_loss": float(train_loss),
                "client_id": str(self.data.client_id),
                # The current baseline aggregates the full predictive model. Future versions
                # may introduce explicit shared/local parameter splits.
                "aggregation_scope": "full_model",
                "shared_parameter_count": int(len(shared_payload.shared_parameters)),
                "shared_parameter_indices": ",".join(
                    str(index) for index in shared_payload.shared_parameter_indices
                ),
            }
            if self._secure_encoder is not None:
                weighted_payload_max_abs = compute_weighted_payload_max_abs(
                    shared_payload.shared_parameters,
                    int(self.data.y_train.shape[0]),
                )
                round_id = int(config.get("server_round", 0))
                encoded_update = self._secure_encoder.encode(
                    shared_payload.shared_parameters,
                    client_id=str(self.data.client_id),
                    round_id=round_id,
                    weight=int(self.data.y_train.shape[0]),
                )
                metrics[SECURE_PAYLOAD_ENCODING_KEY] = SECURE_PAYLOAD_ENCODING_VALUE
                metrics[SECURE_PAYLOAD_LAYOUT_KEY] = serialize_secure_payload_layout(encoded_update.layout)
                metrics[SECURE_WEIGHTED_PAYLOAD_MAX_ABS_KEY] = float(weighted_payload_max_abs)
                return (
                    [
                        np.asarray(share.payload, dtype=np.int64).copy()
                        for share in encoded_update.helper_shares
                    ],
                    int(self.data.y_train.shape[0]),
                    metrics,
                )
            return (
                shared_payload.shared_parameters,
                int(self.data.y_train.shape[0]),
                metrics,
            )

        def evaluate(
            self,
            parameters: list[np.ndarray],
            config: dict[str, Any],
        ) -> tuple[float, int, dict[str, Any]]:
            merged_parameters = apply_shared_parameter_payload(
                self.model.get_parameters(),
                parameters,
            )
            self.model.set_parameters(merged_parameters)
            loss = self.model.loss(self.data.X_test, self.data.y_test)
            probabilities = self.model.predict_proba(self.data.X_test)
            metrics = compute_classification_metrics(
                self.data.y_test,
                probabilities,
                loss,
                threshold=self.prediction_threshold,
            )
            metrics["client_id"] = str(self.data.client_id)
            return float(loss), int(self.data.y_test.shape[0]), metrics


    class FederatedPairwiseRecommenderClient(fl.client.NumPyClient):
        """Flower NumPy client backed by the pairwise logistic recommender."""

        def __init__(
            self,
            data: RecommenderClientData,
            model_config: PairwiseLogisticConfig,
            seed: int,
            recommender_type: str = DEFAULT_RECOMMENDER_TYPE,
            secure_aggregation: SecureAggregationClientSpec | None = None,
        ) -> None:
            self.data = data
            self.seed = int(seed)
            self._secure_aggregation = secure_aggregation or SecureAggregationClientSpec()
            self._secure_encoder = (
                _build_client_secure_encoder(self._secure_aggregation)
                if self._secure_aggregation.enabled
                else None
            )
            self.model = create_recommender(
                recommender_type=recommender_type,
                n_features=data.X_train.shape[1],
                config=model_config,
            )
            self._last_clustering_vector: np.ndarray | None = None

        def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
            return extract_shared_parameter_payload(self.model.get_parameters()).shared_parameters

        def _train_shared_payload(
            self,
            parameters: list[np.ndarray],
        ) -> tuple[SharedParameterPayload, float]:
            merged_parameters = apply_shared_parameter_payload(
                self.model.get_parameters(),
                parameters,
            )
            self.model.set_parameters(merged_parameters)
            train_loss = self.model.fit(
                self.data.X_train,
                self.data.y_train,
                seed=self.seed + self.data.client_id,
            )
            return extract_shared_parameter_payload(self.model.get_parameters()), float(train_loss)

        def _encode_shared_payload(
            self,
            shared_parameters: list[np.ndarray],
            *,
            round_id: int,
            num_examples: int,
        ) -> tuple[Any, float]:
            if self._secure_encoder is None:
                raise RuntimeError(
                    "Clustered recommender training requires client-side secure aggregation encoding."
                )
            weighted_payload_max_abs = compute_weighted_payload_max_abs(
                shared_parameters,
                int(num_examples),
            )
            encoded_update = self._secure_encoder.encode(
                shared_parameters,
                client_id=self.data.client_name,
                round_id=round_id,
                weight=int(num_examples),
            )
            return encoded_update, float(weighted_payload_max_abs)

        def _compute_clustering_vector(
            self,
            *,
            shared_parameters: list[np.ndarray],
            base_parameters: list[np.ndarray],
            representation: str,
        ) -> np.ndarray:
            from fed_perso_xai.recommender.clustering import RecommenderWeightVectorExtractor

            extractor = RecommenderWeightVectorExtractor()
            fitted_vector = extractor.flatten(shared_parameters)
            normalized_representation = str(representation).strip().lower()
            if normalized_representation == "model":
                return fitted_vector
            if normalized_representation == "delta":
                return fitted_vector - extractor.flatten(base_parameters)
            raise ValueError(f"Unsupported clustering representation {representation!r}.")

        def fit(
            self,
            parameters: list[np.ndarray],
            config: dict[str, Any],
        ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
            LOGGER.info(
                "Recommender fit start client=%s train_pairs=%s",
                self.data.client_name,
                int(self.data.y_train.shape[0]),
            )
            shared_payload, train_loss = self._train_shared_payload(parameters)
            metrics: dict[str, Any] = {
                "train_loss": float(train_loss),
                "client_id": self.data.client_name,
                "aggregation_scope": "full_recommender",
                "shared_parameter_count": int(len(shared_payload.shared_parameters)),
                "shared_parameter_indices": ",".join(
                    str(index) for index in shared_payload.shared_parameter_indices
                ),
            }
            LOGGER.info(
                "Recommender fit complete client=%s train_pairs=%s train_loss=%.6f",
                self.data.client_name,
                int(self.data.y_train.shape[0]),
                float(train_loss),
            )
            if self._secure_encoder is not None:
                round_id = int(config.get("server_round", 0))
                encoded_update, weighted_payload_max_abs = self._encode_shared_payload(
                    shared_payload.shared_parameters,
                    round_id=round_id,
                    num_examples=int(self.data.y_train.shape[0]),
                )
                metrics[SECURE_PAYLOAD_ENCODING_KEY] = SECURE_PAYLOAD_ENCODING_VALUE
                metrics[SECURE_PAYLOAD_LAYOUT_KEY] = serialize_secure_payload_layout(encoded_update.layout)
                metrics[SECURE_WEIGHTED_PAYLOAD_MAX_ABS_KEY] = float(weighted_payload_max_abs)
                return (
                    [
                        np.asarray(share.payload, dtype=np.int64).copy()
                        for share in encoded_update.helper_shares
                    ],
                    int(self.data.y_train.shape[0]),
                    metrics,
                )
            return (
                shared_payload.shared_parameters,
                int(self.data.y_train.shape[0]),
                metrics,
            )

        def fit_clustered(
            self,
            parameters: list[np.ndarray],
            config: dict[str, Any],
            *,
            representation: str,
            include_raw_clustering_vector: bool,
        ) -> ClusteredRecommenderClientUpdate:
            LOGGER.info(
                "Clustered recommender fit start client=%s train_pairs=%s include_raw_clustering_vector=%s",
                self.data.client_name,
                int(self.data.y_train.shape[0]),
                bool(include_raw_clustering_vector),
            )
            shared_payload, train_loss = self._train_shared_payload(parameters)
            round_id = int(config.get("server_round", 0))
            num_examples = int(self.data.y_train.shape[0])
            encoded_update, weighted_payload_max_abs = self._encode_shared_payload(
                shared_payload.shared_parameters,
                round_id=round_id,
                num_examples=num_examples,
            )
            self._last_clustering_vector = self._compute_clustering_vector(
                shared_parameters=shared_payload.shared_parameters,
                base_parameters=parameters,
                representation=representation,
            )
            LOGGER.info(
                "Clustered recommender fit complete client=%s train_pairs=%s train_loss=%.6f",
                self.data.client_name,
                num_examples,
                float(train_loss),
            )
            raw_clustering_vector = (
                np.asarray(self._last_clustering_vector, dtype=np.float64).copy()
                if include_raw_clustering_vector
                else None
            )
            return ClusteredRecommenderClientUpdate(
                client_id=self.data.client_name,
                num_examples=num_examples,
                train_loss=float(train_loss),
                encoded_model_update=encoded_update,
                weighted_payload_max_abs=float(weighted_payload_max_abs),
                raw_clustering_vector=raw_clustering_vector,
            )

        def build_last_private_clustering_vector(
            self,
            *,
            projection_spec: Any,
            round_id: int,
        ) -> Any:
            from fed_perso_xai.recommender.clustering import ClientSideRandomProjector

            if self._last_clustering_vector is None:
                raise RuntimeError(
                    f"No clustering representation is available for client {self.data.client_name}."
                )
            projector = ClientSideRandomProjector(self._secure_aggregation)
            return projector.build_private_reduced_vector_from_flat_vector(
                client_id=self.data.client_name,
                flat_vector=self._last_clustering_vector,
                projection_spec=projection_spec,
                round_id=int(round_id),
            )

        def evaluate(
            self,
            parameters: list[np.ndarray],
            config: dict[str, Any],
        ) -> tuple[float, int, dict[str, Any]]:
            eval_pairs = int(self.data.y_eval.shape[0])
            LOGGER.info(
                "Recommender eval start client=%s eval_pairs=%s",
                self.data.client_name,
                eval_pairs,
            )
            if eval_pairs == 0:
                LOGGER.info(
                    "Recommender eval skipped client=%s eval_pairs=0",
                    self.data.client_name,
                )
                return 0.0, 0, {"client_id": self.data.client_name}
            merged_parameters = apply_shared_parameter_payload(
                self.model.get_parameters(),
                parameters,
            )
            self.model.set_parameters(merged_parameters)
            loss = self.model.loss(self.data.X_eval, self.data.y_eval)
            predictions = self.model.predict_pairwise(self.data.X_eval)
            accuracy = float(np.mean(predictions == self.data.y_eval))
            metrics: dict[str, Any] = {
                "client_id": self.data.client_name,
                "pairwise_accuracy": accuracy,
            }
            LOGGER.info(
                "Recommender eval complete client=%s eval_pairs=%s eval_loss=%.6f pairwise_accuracy=%.6f",
                self.data.client_name,
                eval_pairs,
                float(loss),
                accuracy,
            )
            return float(loss), eval_pairs, metrics


else:

    class FederatedLogisticRegressionClient:
        """Placeholder used when Flower is not installed."""

        def __init__(
            self,
            data: ClientData,
            model_name: str,
            model_config: Any,
            seed: int,
        ) -> None:
            raise ImportError(FLOWER_IMPORT_ERROR_MESSAGE)

    class FederatedPairwiseRecommenderClient:
        """Placeholder used when Flower is not installed."""

        def __init__(
            self,
            data: RecommenderClientData,
            model_config: PairwiseLogisticConfig,
            seed: int,
            recommender_type: str = DEFAULT_RECOMMENDER_TYPE,
            secure_aggregation: SecureAggregationClientSpec | None = None,
        ) -> None:
            raise ImportError(FLOWER_IMPORT_ERROR_MESSAGE)
