"""Flower strategy helpers with optional protocol-first secure aggregation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

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

from fed_perso_xai.evaluation.metrics import aggregate_weighted_metrics
from fed_perso_xai.fl.client import (
    SECURE_PAYLOAD_ENCODING_KEY,
    SECURE_PAYLOAD_ENCODING_VALUE,
    SECURE_PAYLOAD_LAYOUT_KEY,
    SECURE_WEIGHTED_PAYLOAD_CLIP_SUMMARY_KEY,
    SECURE_WEIGHTED_PAYLOAD_MAX_ABS_KEY,
    deserialize_secure_payload_clipping_summary,
    deserialize_secure_payload_layout,
    extract_shared_parameter_payload,
)
from fed_perso_xai.utils.config import FederatedTrainingConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class FederatedRunRecorder:
    """Mutable recorder populated by the Flower strategy during training."""

    backend: str
    round_history: list[dict[str, object]] = field(default_factory=list)
    final_parameters: list[np.ndarray] | None = None


@dataclass(frozen=True)
class SecureQuantizationPlan:
    """Resolved per-round quantization settings for secure aggregation."""

    requested_scale: int
    effective_scale: int
    signed_bound: int
    max_component_l1: float


class StrategyFactory(Protocol):
    """Protocol for pluggable aggregation strategies in later stages."""

    def create(
        self,
        initial_parameters: list[np.ndarray],
        recorder: FederatedRunRecorder,
    ) -> Any:
        """Build a Flower strategy."""


StrategyFactoryBuilder = Callable[[FederatedTrainingConfig], StrategyFactory]


@dataclass(frozen=True)
class StrategySpec:
    """Declarative strategy-factory entry."""

    key: str
    display_name: str
    build_factory: StrategyFactoryBuilder


class StrategyRegistry:
    """Registry of supported federated aggregation strategies."""

    def __init__(self, specs: list[StrategySpec] | None = None) -> None:
        self._specs: dict[str, StrategySpec] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: StrategySpec) -> None:
        if spec.key in self._specs:
            raise ValueError(f"Strategy '{spec.key}' is already registered.")
        self._specs[spec.key] = spec

    def get(self, key: str) -> StrategySpec:
        try:
            return self._specs[key]
        except KeyError as exc:
            supported = ", ".join(sorted(self._specs))
            raise ValueError(
                f"Unsupported strategy '{key}'. Supported strategies: {supported}."
            ) from exc

    def list_keys(self) -> list[str]:
        return sorted(self._specs)


def _weighted_average_parameter_sets(
    parameter_sets: Sequence[Sequence[np.ndarray]],
    weights: Sequence[int | float],
) -> list[np.ndarray]:
    if not parameter_sets:
        raise ValueError("parameter_sets must contain at least one payload.")
    if len(parameter_sets) != len(weights):
        raise ValueError("weights must align with parameter_sets.")

    reference_shapes = tuple(tuple(np.asarray(array).shape) for array in parameter_sets[0])
    weighted_sums = [
        np.zeros_like(np.asarray(parameter, dtype=np.float64), dtype=np.float64)
        for parameter in parameter_sets[0]
    ]
    total_weight = float(sum(float(weight) for weight in weights))
    if total_weight <= 0.0:
        raise ValueError("weights must sum to a positive value.")

    for payload, weight in zip(parameter_sets, weights, strict=True):
        payload_shapes = tuple(tuple(np.asarray(array).shape) for array in payload)
        if payload_shapes != reference_shapes:
            raise ValueError("All shared parameter payloads must have the same shapes.")
        for index, parameter in enumerate(payload):
            weighted_sums[index] += np.asarray(parameter, dtype=np.float64) * float(weight)

    return [parameter / total_weight for parameter in weighted_sums]


def _scale_parameter_set(
    parameters: Sequence[np.ndarray],
    factor: int | float,
) -> list[np.ndarray]:
    return [np.asarray(parameter, dtype=np.float64) * float(factor) for parameter in parameters]


def _validate_secure_config(training_config: FederatedTrainingConfig) -> None:
    reconstruction_threshold = (
        training_config.secure_reconstruction_threshold
        if training_config.secure_reconstruction_threshold is not None
        else training_config.secure_privacy_threshold + 1
    )
    if reconstruction_threshold > training_config.secure_num_helpers:
        raise ValueError(
            "Invalid secure aggregation config: secure_reconstruction_threshold "
            f"({reconstruction_threshold}) must be less than or equal to "
            f"secure_num_helpers ({training_config.secure_num_helpers})."
        )
    if training_config.secure_privacy_threshold >= reconstruction_threshold:
        raise ValueError(
            "Invalid secure aggregation config: secure_privacy_threshold "
            f"({training_config.secure_privacy_threshold}) must be strictly less than "
            f"secure_reconstruction_threshold ({reconstruction_threshold})."
        )
    if training_config.secure_num_helpers < reconstruction_threshold:
        raise ValueError(
            "Invalid secure aggregation config: secure_num_helpers "
            f"({training_config.secure_num_helpers}) must be greater than or equal to "
            f"secure_reconstruction_threshold ({reconstruction_threshold})."
        )


def _flatten_secure_payload(payload: Sequence[np.ndarray]) -> np.ndarray:
    flattened = [
        np.asarray(parameter, dtype=np.float64).reshape(-1)
        for parameter in payload
    ]
    if not flattened:
        raise ValueError("Secure aggregation payloads must contain at least one tensor.")
    return np.concatenate(flattened)


def _plan_secure_quantization(
    payloads: Sequence[Sequence[np.ndarray]],
    *,
    requested_scale: int,
    field_modulus: int,
) -> SecureQuantizationPlan:
    if requested_scale < 1:
        raise ValueError("requested_scale must be at least 1.")
    if field_modulus <= 2:
        raise ValueError("field_modulus must be greater than 2.")
    if not payloads:
        raise ValueError("payloads must contain at least one client payload.")

    stacked = np.stack([_flatten_secure_payload(payload) for payload in payloads], axis=0)
    max_component_l1 = float(np.max(np.sum(np.abs(stacked), axis=0))) if stacked.size else 0.0
    if not np.isfinite(max_component_l1):
        raise ValueError("Secure aggregation payloads must be finite.")

    signed_bound = (field_modulus - 1) // 2
    if max_component_l1 <= 0.0:
        effective_scale = int(requested_scale)
    else:
        max_safe_scale = int(signed_bound // max_component_l1)
        if max_safe_scale < 1:
            raise ValueError(
                "Secure aggregation payload magnitude exceeds the finite-field range even "
                "at scale=1. Reduce model magnitude or use a larger field modulus."
            )
        effective_scale = min(int(requested_scale), max_safe_scale)

    return SecureQuantizationPlan(
        requested_scale=int(requested_scale),
        effective_scale=int(effective_scale),
        signed_bound=int(signed_bound),
        max_component_l1=max_component_l1,
    )


def _validate_encoded_secure_aggregate_bound(
    results: Sequence[tuple[Any, Any]],
    *,
    quantization_scale: int,
    field_modulus: int,
) -> float:
    if quantization_scale < 1:
        raise ValueError("quantization_scale must be at least 1.")
    if field_modulus <= 2:
        raise ValueError("field_modulus must be greater than 2.")
    weighted_payload_bounds: list[float] = []
    for _, fit_res in results:
        bound = fit_res.metrics.get(SECURE_WEIGHTED_PAYLOAD_MAX_ABS_KEY)
        if not isinstance(bound, (int, float)):
            raise ValueError(
                "Secure aggregation expected each client to report secure_weighted_payload_max_abs."
            )
        bound_value = float(bound)
        if not np.isfinite(bound_value):
            raise ValueError("Secure aggregation weighted payload bounds must be finite.")
        if bound_value < 0.0:
            raise ValueError("Secure aggregation weighted payload bounds must be non-negative.")
        weighted_payload_bounds.append(bound_value)
    total_max_abs_bound = float(sum(weighted_payload_bounds))
    signed_bound = (field_modulus - 1) // 2
    if total_max_abs_bound * float(quantization_scale) > float(signed_bound):
        raise ValueError(
            "Secure aggregation aggregate may overflow the finite field under the current "
            f"bound estimate (sum_client_max_abs={total_max_abs_bound:.6g}, "
            f"scale={quantization_scale}, signed_bound={signed_bound})."
        )
    return total_max_abs_bound


def _build_secure_clipping_report(results: Sequence[tuple[Any, Any]]) -> dict[str, Any] | None:
    clipping_clients: list[dict[str, Any]] = []
    total_clients = 0
    for _, fit_res in results:
        total_clients += 1
        payload = fit_res.metrics.get(SECURE_WEIGHTED_PAYLOAD_CLIP_SUMMARY_KEY)
        if not isinstance(payload, str):
            continue
        summary = deserialize_secure_payload_clipping_summary(payload)
        if not summary.clip_applied:
            continue
        clipping_clients.append(
            {
                "client_id": str(fit_res.metrics.get("client_id", "")),
                "num_examples": int(getattr(fit_res, "num_examples", 0)),
                "clip_threshold_abs": (
                    None
                    if summary.clip_threshold_abs is None
                    else float(summary.clip_threshold_abs)
                ),
                "raw_weighted_payload_max_abs": float(summary.raw_weighted_payload_max_abs),
                "effective_weighted_payload_max_abs": float(
                    summary.effective_weighted_payload_max_abs
                ),
                "raw_max_component_value": float(summary.raw_max_component_value),
                "clipped_max_component_value": float(summary.clipped_max_component_value),
                "clipped_component_count": int(summary.clipped_component_count),
                "total_component_count": int(summary.total_component_count),
                "clipped_fraction": float(summary.clipped_fraction),
                "total_clipping_l1": float(summary.total_clipping_l1),
                "max_clipping_delta_abs": float(summary.max_clipping_delta_abs),
            }
        )
    if not clipping_clients:
        return None
    return {
        "applied_client_count": int(len(clipping_clients)),
        "total_client_count": int(total_clients),
        "max_raw_weighted_payload_abs": float(
            max(item["raw_weighted_payload_max_abs"] for item in clipping_clients)
        ),
        "max_effective_weighted_payload_abs": float(
            max(item["effective_weighted_payload_max_abs"] for item in clipping_clients)
        ),
        "total_clipped_components": int(
            sum(item["clipped_component_count"] for item in clipping_clients)
        ),
        "total_components": int(
            sum(item["total_component_count"] for item in clipping_clients)
        ),
        "total_clipping_l1": float(
            sum(item["total_clipping_l1"] for item in clipping_clients)
        ),
        "clients": clipping_clients,
    }


def _build_secure_aggregator(
    training_config: FederatedTrainingConfig,
    *,
    scale_override: int | None = None,
) -> Any:
    _validate_secure_config(training_config)
    try:
        from lcc_lib.aggregation.secure_aggregator import (
            SecureAggregationConfig,
            SecureAggregator,
        )
        from lcc_lib.coding.field_ops import FieldConfig
        from lcc_lib.coding.share_codec import ShareEncodingConfig
        from lcc_lib.quantization.quantizer import QuantizationConfig
    except ImportError as exc:  # pragma: no cover - depends on local sibling install
        raise ImportError(
            "Secure aggregation requires `lcc-lib`. Install the sibling package, for example "
            "with `python3 -m pip install ../lcc-lib`, before enabling `secure_aggregation`."
        ) from exc

    return SecureAggregator(
        SecureAggregationConfig(
            field_config=FieldConfig(modulus=training_config.secure_field_modulus),
            quantization=QuantizationConfig(
                field_modulus=training_config.secure_field_modulus,
                scale=(
                    int(scale_override)
                    if scale_override is not None
                    else training_config.secure_quantization_scale
                ),
            ),
            encoding=ShareEncodingConfig(
                num_helpers=training_config.secure_num_helpers,
                privacy_threshold=training_config.secure_privacy_threshold,
                reconstruction_threshold=training_config.secure_reconstruction_threshold,
                seed=training_config.secure_seed,
            ),
            # `compute_mean=False` is intentional: `lcc-lib` returns a sum.
            # The strategy owns weighting and normalization. If that contract
            # changes in `lcc-lib`, this strategy must be updated as well.
            compute_mean=False,
        )
    )


def _build_pre_encoded_secure_aggregator(training_config: FederatedTrainingConfig) -> Any:
    _validate_secure_config(training_config)
    try:
        from lcc_lib.aggregation import EncodedShareAggregator, SecureAggregationConfig
        from lcc_lib.coding.field_ops import FieldConfig
        from lcc_lib.coding.share_codec import ShareEncodingConfig
        from lcc_lib.quantization.quantizer import QuantizationConfig
    except ImportError as exc:  # pragma: no cover - depends on local sibling install
        raise ImportError(
            "Secure aggregation requires `lcc-lib`. Install the sibling package, for example "
            "with `python3 -m pip install ../lcc-lib`, before enabling `secure_aggregation`."
        ) from exc

    return EncodedShareAggregator(
        SecureAggregationConfig(
            field_config=FieldConfig(modulus=training_config.secure_field_modulus),
            quantization=QuantizationConfig(
                field_modulus=training_config.secure_field_modulus,
                scale=training_config.secure_quantization_scale,
            ),
            encoding=ShareEncodingConfig(
                num_helpers=training_config.secure_num_helpers,
                privacy_threshold=training_config.secure_privacy_threshold,
                reconstruction_threshold=training_config.secure_reconstruction_threshold,
                seed=training_config.secure_seed,
            ),
            compute_mean=False,
        )
    )


if fl is not None:

    class TrackingFedAvg(fl.server.strategy.FedAvg):
        """FedAvg strategy with optional secure aggregation for shared parameters."""

        def __init__(
            self,
            *args,
            recorder: FederatedRunRecorder,
            training_config: FederatedTrainingConfig,
            **kwargs,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.recorder = recorder
            self.training_config = training_config
            self._secure_share_aggregator = (
                _build_pre_encoded_secure_aggregator(training_config)
                if training_config.secure_aggregation
                else None
            )
            LOGGER.info(
                "Aggregation mode=%s secure_num_helpers=%s secure_quantization_scale=%s secure_field_modulus=%s",
                "secure" if training_config.secure_aggregation else "plain",
                training_config.secure_num_helpers,
                training_config.secure_quantization_scale,
                training_config.secure_field_modulus,
            )

        def aggregate_fit(self, server_round, results, failures):  # type: ignore[override]
            if not results:
                return None, {}
            if failures and not self.accept_failures:
                return None, {}

            if self.training_config.secure_aggregation:
                aggregated_parameters, aggregation_record = self._aggregate_secure_shared(
                    server_round,
                    results,
                )
            else:
                aggregated_parameters, aggregation_record = self._aggregate_plain_shared(
                    server_round,
                    results,
                )

            metrics: dict[str, Any] = {}
            if self.fit_metrics_aggregation_fn is not None:
                metrics = self.fit_metrics_aggregation_fn(
                    [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
                )

            round_record = self._ensure_round_record(server_round)
            round_record["fit_metrics"] = metrics
            round_record["aggregation"] = aggregation_record
            self.recorder.final_parameters = aggregated_parameters
            LOGGER.info(
                "Round %s/%s fit summary contributors=%s metrics=%s",
                server_round,
                self.training_config.rounds,
                aggregation_record.get("num_contributors", 0),
                _format_scalar_metrics(metrics),
            )
            if float(self.training_config.evaluate_fraction) <= 0.0:
                round_record["evaluate_loss"] = None
                round_record["evaluate_metrics"] = {}
                round_record["evaluate_skipped"] = True
            return fl.common.ndarrays_to_parameters(aggregated_parameters), metrics

        def aggregate_evaluate(self, server_round, results, failures):  # type: ignore[override]
            if results and all(getattr(evaluate_res, "num_examples", 0) <= 0 for _, evaluate_res in results):
                round_record = self._ensure_round_record(server_round)
                round_record["evaluate_loss"] = None
                round_record["evaluate_metrics"] = {}
                round_record["evaluate_skipped"] = True
                LOGGER.info(
                    "Round %s/%s eval summary skipped=no_eval_examples",
                    server_round,
                    self.training_config.rounds,
                )
                return None, {}
            loss, metrics = super().aggregate_evaluate(server_round, results, failures)
            round_record = self._ensure_round_record(server_round)
            round_record["evaluate_loss"] = loss
            round_record["evaluate_metrics"] = metrics
            LOGGER.info(
                "Round %s/%s eval summary loss=%s metrics=%s",
                server_round,
                self.training_config.rounds,
                "None" if loss is None else f"{float(loss):.6f}",
                _format_scalar_metrics(metrics),
            )
            return loss, metrics

        def _aggregate_plain_shared(
            self,
            server_round: int,
            results: list[tuple[Any, Any]],
        ) -> tuple[list[np.ndarray], dict[str, object]]:
            shared_payloads = [
                self._extract_shared_payload_from_fitres(fit_res) for _, fit_res in results
            ]
            weights = [fit_res.num_examples for _, fit_res in results]
            aggregated = self._compose_updated_parameters(
                _weighted_average_parameter_sets(shared_payloads, weights)
            )
            LOGGER.info(
                "Round %s plain aggregation contributors=%s",
                server_round,
                len(shared_payloads),
            )
            return aggregated, {
                "mode": "plain",
                "num_contributors": len(shared_payloads),
                "helper_count": 0,
            }

        def _aggregate_secure_shared(
            self,
            server_round: int,
            results: list[tuple[Any, Any]],
        ) -> tuple[list[np.ndarray], dict[str, object]]:
            if self._secure_share_aggregator is None:
                raise RuntimeError("Secure aggregation was requested but no aggregator is configured.")

            weights = [fit_res.num_examples for _, fit_res in results]
            total_weight = float(sum(weights))
            if total_weight <= 0.0:
                raise ValueError("Secure aggregation requires a positive total example count.")
            max_component_l1 = _validate_encoded_secure_aggregate_bound(
                results,
                quantization_scale=self.training_config.secure_quantization_scale,
                field_modulus=self.training_config.secure_field_modulus,
            )
            clipping_report = _build_secure_clipping_report(results)
            secure_aggregator = self._secure_share_aggregator
            if secure_aggregator is None:
                raise RuntimeError("Secure aggregation was requested but no aggregator is configured.")
            encoded_updates = [
                self._extract_secure_update_from_fitres(fit_res, default_client_id=str(index))
                for index, (_, fit_res) in enumerate(results)
            ]
            secure_result = secure_aggregator.aggregate_encoded(
                encoded_updates,
                round_id=server_round,
            )
            aggregated = self._compose_updated_parameters(
                _scale_parameter_set(secure_result.aggregated_tensors, 1.0 / total_weight)
            )
            max_abs_error = (
                float(secure_result.max_abs_error)
                if np.isfinite(secure_result.max_abs_error)
                else None
            )
            LOGGER.info(
                "Round %s secure aggregation contributors=%s helpers=%s scale=%s modulus=%s max_abs_error=%s",
                server_round,
                secure_result.num_contributors,
                len(secure_result.helper_ids),
                self.training_config.secure_quantization_scale,
                self.training_config.secure_field_modulus,
                "n/a" if max_abs_error is None else f"{max_abs_error:.6g}",
            )
            if clipping_report is not None:
                LOGGER.warning(
                    "Round %s secure clipping summary applied_clients=%s/%s "
                    "max_raw_weighted_abs=%.6g max_effective_weighted_abs=%.6g "
                    "total_clipped_components=%s total_clipping_l1=%.6g",
                    server_round,
                    int(clipping_report["applied_client_count"]),
                    int(clipping_report["total_client_count"]),
                    float(clipping_report["max_raw_weighted_payload_abs"]),
                    float(clipping_report["max_effective_weighted_payload_abs"]),
                    int(clipping_report["total_clipped_components"]),
                    float(clipping_report["total_clipping_l1"]),
                )
            return aggregated, {
                "mode": "secure",
                "num_contributors": secure_result.num_contributors,
                "helper_count": len(secure_result.helper_ids),
                "field_modulus": self.training_config.secure_field_modulus,
                "quantization_scale": self.training_config.secure_quantization_scale,
                "requested_quantization_scale": self.training_config.secure_quantization_scale,
                "max_component_l1": max_component_l1,
                "max_abs_error": max_abs_error,
                "clipping": clipping_report,
            }

        def _extract_shared_payload_from_fitres(self, fit_res: Any) -> list[np.ndarray]:
            parameters = fl.common.parameters_to_ndarrays(fit_res.parameters)
            shared_payload = extract_shared_parameter_payload(parameters)
            return shared_payload.shared_parameters

        def _extract_secure_update_from_fitres(
            self,
            fit_res: Any,
            *,
            default_client_id: str,
        ) -> Any:
            try:
                from lcc_lib.aggregation import EncodedClientUpdate
                from lcc_lib.coding.share_codec import EncodedShare, ShareEncodingConfig
            except ImportError as exc:  # pragma: no cover - depends on local sibling install
                raise ImportError(
                    "Secure aggregation requires `lcc-lib`. Install the sibling package, for example "
                    "with `python3 -m pip install ../lcc-lib`, before enabling `secure_aggregation`."
                ) from exc

            encoding_mode = fit_res.metrics.get(SECURE_PAYLOAD_ENCODING_KEY)
            if encoding_mode != SECURE_PAYLOAD_ENCODING_VALUE:
                raise ValueError(
                    "Secure aggregation expected client-encoded helper shares in fit metrics."
                )
            layout_payload = fit_res.metrics.get(SECURE_PAYLOAD_LAYOUT_KEY)
            if not isinstance(layout_payload, str):
                raise ValueError("Secure aggregation expected a serialized payload layout in fit metrics.")
            layout = deserialize_secure_payload_layout(layout_payload)
            helper_payloads = fl.common.parameters_to_ndarrays(fit_res.parameters)
            encoding_config = ShareEncodingConfig(
                num_helpers=self.training_config.secure_num_helpers,
                privacy_threshold=self.training_config.secure_privacy_threshold,
                reconstruction_threshold=self.training_config.secure_reconstruction_threshold,
                seed=self.training_config.secure_seed,
            )
            helper_points = encoding_config.helper_evaluation_points
            if len(helper_payloads) != len(helper_points):
                raise ValueError(
                    "Secure aggregation expected one encoded helper share per configured helper."
                )
            helper_shares = tuple(
                EncodedShare(
                    helper_id=helper_id,
                    evaluation_point=evaluation_point,
                    payload=np.asarray(payload, dtype=np.int64),
                )
                for helper_id, (evaluation_point, payload) in enumerate(
                    zip(helper_points, helper_payloads, strict=True)
                )
            )
            return EncodedClientUpdate(
                client_id=str(fit_res.metrics.get("client_id", default_client_id)),
                layout=layout,
                helper_shares=helper_shares,
            )

        def _compose_updated_parameters(
            self,
            aggregated_shared_parameters: Sequence[np.ndarray],
        ) -> list[np.ndarray]:
            """Return server parameters after aggregation.

            The current baseline uses full-model aggregation. This function currently acts as
            a pass-through. It exists to support future partial/shared-only
            recomposition when local parameters are introduced.
            """
            return [
                np.asarray(parameter, dtype=np.float64).copy()
                for parameter in aggregated_shared_parameters
            ]

        def _ensure_round_record(self, server_round: int) -> dict[str, object]:
            while len(self.recorder.round_history) < server_round:
                self.recorder.round_history.append(
                    {"round": len(self.recorder.round_history) + 1}
                )
            return self.recorder.round_history[server_round - 1]

else:

    class TrackingFedAvg:
        """Placeholder used when Flower is not installed."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(FLOWER_IMPORT_ERROR_MESSAGE)


@dataclass(frozen=True)
class FedAvgStrategyFactory:
    """Default strategy factory for the predictive baseline."""

    training_config: FederatedTrainingConfig

    def create(
        self,
        initial_parameters: list[np.ndarray],
        recorder: FederatedRunRecorder,
    ) -> Any:
        if fl is None:  # pragma: no cover - depends on optional deps
            raise ImportError(FLOWER_IMPORT_ERROR_MESSAGE)
        minimum_clients = min(
            self.training_config.min_available_clients,
            self.training_config.num_clients,
        )
        min_evaluate_clients = minimum_clients if self.training_config.evaluate_fraction > 0.0 else 0
        return TrackingFedAvg(
            recorder=recorder,
            training_config=self.training_config,
            fraction_fit=self.training_config.fit_fraction,
            fraction_evaluate=self.training_config.evaluate_fraction,
            min_fit_clients=minimum_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=minimum_clients,
            initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
            fit_metrics_aggregation_fn=_aggregate_scalar_metrics,
            evaluate_metrics_aggregation_fn=_aggregate_scalar_metrics,
            on_fit_config_fn=lambda server_round: {"server_round": int(server_round)},
        )


DEFAULT_STRATEGY_REGISTRY = StrategyRegistry(
    specs=[
        StrategySpec(
            key="fedavg",
            display_name="FedAvg",
            build_factory=lambda training_config: FedAvgStrategyFactory(training_config),
        )
    ]
)


def create_strategy_factory(
    strategy_name: str,
    *,
    training_config: FederatedTrainingConfig,
    registry: StrategyRegistry | None = None,
) -> StrategyFactory:
    """Build one configured strategy factory from the registry."""

    spec = (registry or DEFAULT_STRATEGY_REGISTRY).get(strategy_name)
    return spec.build_factory(training_config)


def _format_scalar_metrics(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return "none"
    items: list[str] = []
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, (int, float)):
            items.append(f"{key}={float(value):.6f}")
    return ", ".join(items) if items else "none"


def _aggregate_scalar_metrics(
    metrics: list[tuple[int, dict[str, Any]]],
) -> dict[str, Any]:
    numeric_rows: list[tuple[int, dict[str, float]]] = []
    for num_examples, row in metrics:
        numeric_row = {
            key: float(value)
            for key, value in row.items()
            if isinstance(value, (int, float)) and key != "client_id"
        }
        if numeric_row:
            numeric_rows.append((num_examples, numeric_row))
    return aggregate_weighted_metrics(numeric_rows) if numeric_rows else {}
