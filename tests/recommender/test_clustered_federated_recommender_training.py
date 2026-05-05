from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fed_perso_xai.fl.client import normalize_clustering_vector
from fed_perso_xai.fl.recommender_simulation import (
    _align_cluster_labels_to_previous_round,
    _apply_assignment_hysteresis,
)
from fed_perso_xai.orchestration.recommender_training import train_federated_recommender
from fed_perso_xai.recommender.clustering import (
    ClientSideRandomProjector,
    IdentityProjectionSpec,
    PCAProjectionSpec,
    RandomProjectionSpec,
    SecretSharedReducedVector,
    SecureClusterAssignments,
    SecureKMeansClusterer,
    build_centered_pca_projection_spec,
    build_identity_projection_spec,
    build_random_projection_spec,
)
from fed_perso_xai.utils.config import (
    ArtifactPaths,
    RecommenderClusteringConfig,
    RecommenderFederatedTrainingConfig,
)

FLOWER_AVAILABLE = importlib.util.find_spec("flwr") is not None
PYARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None
LCC_AVAILABLE = importlib.util.find_spec("lcc_lib") is not None


def test_align_cluster_labels_to_previous_round_matches_by_maximum_overlap() -> None:
    previous_assignments = {
        "client_000": 0,
        "client_001": 0,
        "client_002": 1,
        "client_003": 2,
    }
    current_assignments = {
        "client_000": 2,
        "client_001": 2,
        "client_002": 0,
        "client_003": 1,
    }

    mapping, overlap = _align_cluster_labels_to_previous_round(
        previous_assignments=previous_assignments,
        current_assignments=current_assignments,
        cluster_count=3,
    )

    assert mapping == {0: 1, 1: 2, 2: 0}
    assert overlap == 4


def test_align_cluster_labels_to_previous_round_returns_identity_without_shared_clients() -> None:
    mapping, overlap = _align_cluster_labels_to_previous_round(
        previous_assignments={"client_000": 0},
        current_assignments={"client_111": 2},
        cluster_count=3,
    )

    assert mapping == {0: 0, 1: 1, 2: 2}
    assert overlap == 0


def test_align_cluster_labels_to_previous_round_scales_beyond_small_permutations() -> None:
    previous_assignments = {
        f"client_{cluster_id:03d}": cluster_id
        for cluster_id in range(8)
    }
    current_assignments = {
        f"client_{cluster_id:03d}": (cluster_id + 3) % 8
        for cluster_id in range(8)
    }

    mapping, overlap = _align_cluster_labels_to_previous_round(
        previous_assignments=previous_assignments,
        current_assignments=current_assignments,
        cluster_count=8,
    )

    assert mapping == {
        0: 5,
        1: 6,
        2: 7,
        3: 0,
        4: 1,
        5: 2,
        6: 3,
        7: 4,
    }
    assert overlap == 8


def test_align_cluster_labels_to_previous_round_breaks_ties_deterministically() -> None:
    previous_assignments = {
        "client_000": 0,
        "client_001": 1,
    }
    current_assignments = {
        "client_000": 1,
        "client_001": 0,
    }

    mapping, overlap = _align_cluster_labels_to_previous_round(
        previous_assignments=previous_assignments,
        current_assignments=current_assignments,
        cluster_count=3,
    )

    assert mapping == {0: 1, 1: 0, 2: 2}
    assert overlap == 2


def _paths(tmp_path: Path) -> ArtifactPaths:
    return ArtifactPaths(
        prepared_root=tmp_path / "prepared",
        partition_root=tmp_path / "datasets",
        centralized_root=tmp_path / "centralized",
        federated_root=tmp_path / "federated",
        comparison_root=tmp_path / "comparisons",
        cache_dir=tmp_path / "cache",
    )


def _prepare_recommender_run(
    tmp_path: Path,
    *,
    client_count: int = 4,
    feature_columns: tuple[str, ...] = ("metric_quality_z", "metric_stability_z"),
) -> tuple[ArtifactPaths, str, str, str]:
    paths = _paths(tmp_path)
    run_id = "unit-run"
    selection = "test__max-2__seed-9"
    persona = "lay"
    run_dir = paths.federated_root / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")

    for client_idx in range(client_count):
        client_id = f"client_{client_idx:03d}"
        client_dir = run_dir / "clients" / client_id
        context_dir = client_dir / "recommender_context" / selection
        label_dir = client_dir / "recommender_labels" / selection / persona
        context_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        candidates = pd.DataFrame(
            {
                "client_id": [client_id] * 4,
                "dataset_index": [0, 0, 1, 1],
                "instance_id": ["i0", "i0", "i1", "i1"],
                "method_variant": ["a", "b", "a", "b"],
                feature_columns[0]: [
                    1.0 + client_idx,
                    -1.0 - client_idx,
                    1.5 + client_idx,
                    -1.5 - client_idx,
                ],
                feature_columns[1]: [
                    0.2 * (client_idx + 1),
                    -0.2 * (client_idx + 1),
                    0.3 * (client_idx + 1),
                    -0.3 * (client_idx + 1),
                ],
                "candidate_index_within_instance": [0, 1, 0, 1],
            }
        )
        labels = pd.DataFrame(
            {
                "client_id": [client_id] * 2,
                "dataset_index": [0, 1],
                "pair_1": ["a", "a"],
                "pair_2": ["b", "b"],
                "label": [0, 0],
                "split": ["train", "test"],
            }
        )
        candidates.to_parquet(context_dir / "candidate_context.parquet", index=False)
        labels.to_parquet(label_dir / "pairwise_labels.parquet", index=False)
        (label_dir / "simulation_metadata.json").write_text(
            json.dumps(
                {
                    "instance_split": {
                        "train_dataset_indices": [0],
                        "test_dataset_indices": [1],
                    }
                }
            ),
            encoding="utf-8",
        )
    return paths, run_id, selection, persona


@pytest.mark.skipif(not FLOWER_AVAILABLE, reason="Flower is required for non-clustered recommender FL tests.")
@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow is required for Parquet artifact tests.")
def test_recommender_clustering_disabled_keeps_existing_behavior(tmp_path: Path) -> None:
    paths, run_id, selection, persona = _prepare_recommender_run(tmp_path, client_count=2)

    artifacts, metadata = train_federated_recommender(
        RecommenderFederatedTrainingConfig(
            run_id=run_id,
            selection_id=selection,
            persona=persona,
            paths=paths,
            rounds=2,
            epochs=5,
            batch_size=2,
            learning_rate=0.2,
            simulation_backend="debug-sequential",
            min_available_clients=2,
            top_k=(1, 2),
            clustering=RecommenderClusteringConfig(enabled=False),
        )
    )

    assert artifacts.model_artifact_path.exists()
    assert not artifacts.cluster_manifest_path.exists()
    assert metadata["clustered"] is False
    assert metadata["cluster_model_artifact_paths"] == {}


@pytest.mark.skipif(not LCC_AVAILABLE, reason="lcc-lib is required for clustered recommender tests.")
def test_client_side_projector_secret_shares_reduced_representation_only() -> None:
    config = RecommenderFederatedTrainingConfig(
        run_id="unit-run",
        selection_id="selection-0",
        persona="lay",
        clustering=RecommenderClusteringConfig(enabled=True),
    )
    projector = ClientSideRandomProjector(config)
    projection_spec = RandomProjectionSpec(
        projection_matrix=np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]],
            dtype=np.float64,
        ),
        requested_components=2,
        seed=11,
    )
    private_vector = projector.build_private_reduced_vector(
        client_id="client_000",
        parameters=[np.asarray([1.0, 2.0], dtype=np.float64), np.asarray([0.25], dtype=np.float64)],
        projection_spec=projection_spec,
        round_id=1,
    )

    assert isinstance(private_vector, SecretSharedReducedVector)
    assert private_vector.dimension == 2
    assert not hasattr(private_vector, "reduced_vector")
    assert len(private_vector.helper_vector_shares) == config.secure_num_helpers
    assert len(private_vector.helper_squared_norm_shares) == config.secure_num_helpers
    assert all(np.asarray(share.payload).ndim == 1 for share in private_vector.helper_vector_shares)
    assert all(np.asarray(share.payload).shape == (1,) for share in private_vector.helper_squared_norm_shares)


def test_normalize_clustering_vector_l2_returns_unit_vector() -> None:
    vector = np.asarray([3.0, 4.0], dtype=np.float64)
    normalized = normalize_clustering_vector(vector, enabled=True, mode="l2")

    assert np.allclose(normalized, np.asarray([0.6, 0.8], dtype=np.float64))
    assert np.isclose(np.linalg.norm(normalized), 1.0)


def test_normalize_clustering_vector_keeps_zero_vector_stable() -> None:
    vector = np.zeros(3, dtype=np.float64)
    normalized = normalize_clustering_vector(vector, enabled=True, mode="l2")

    assert np.allclose(normalized, vector)


def test_normalize_clustering_vector_can_use_base_vector_norm() -> None:
    vector = np.asarray([3.0, 4.0], dtype=np.float64)
    base_vector = np.asarray([6.0, 8.0], dtype=np.float64)
    normalized = normalize_clustering_vector(
        vector,
        enabled=True,
        mode="l2",
        reference_vector=base_vector,
    )

    assert np.allclose(normalized, np.asarray([0.3, 0.4], dtype=np.float64))


def test_apply_assignment_hysteresis_keeps_previous_cluster_without_clear_margin() -> None:
    adjusted_assignments, retained_count = _apply_assignment_hysteresis(
        previous_assignments={"client_000": 0, "client_001": 1},
        proposed_assignments={"client_000": 1, "client_001": 1},
        aligned_distance_matrix=np.asarray(
            [
                [1.0, 0.97],
                [2.0, 1.0],
            ],
            dtype=np.float64,
        ),
        ordered_client_ids=("client_000", "client_001"),
        assignment_margin=0.05,
    )

    assert adjusted_assignments == {"client_000": 0, "client_001": 1}
    assert retained_count == 1


def test_apply_assignment_hysteresis_allows_switch_when_new_cluster_is_meaningfully_closer() -> None:
    adjusted_assignments, retained_count = _apply_assignment_hysteresis(
        previous_assignments={"client_000": 0},
        proposed_assignments={"client_000": 1},
        aligned_distance_matrix=np.asarray([[1.0, 0.7]], dtype=np.float64),
        ordered_client_ids=("client_000",),
        assignment_margin=0.05,
    )

    assert adjusted_assignments == {"client_000": 1}
    assert retained_count == 0


def test_secure_kmeans_clusterer_selects_best_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    import fed_perso_xai.recommender.clustering as clustering_module

    config = RecommenderFederatedTrainingConfig(
        run_id="unit-run",
        selection_id="selection-0",
        persona="lay",
        clustering=RecommenderClusteringConfig(enabled=True, k=2, num_restarts=3, max_iterations=1),
    )
    clusterer = SecureKMeansClusterer(config)
    shared_vectors = [
        SecretSharedReducedVector(
            client_id=f"client_{index:03d}",
            helper_vector_shares=tuple(),
            helper_squared_norm_shares=tuple(),
            dimension=2,
        )
        for index in range(2)
    ]
    projection_spec = IdentityProjectionSpec(input_dimension_value=2)
    restart_counter = {"value": -1}
    restart_distance_matrices = (
        np.asarray([[9.0, 1.0], [8.0, 2.0]], dtype=np.float64),
        np.asarray([[0.1, 5.0], [0.2, 6.0]], dtype=np.float64),
        np.asarray([[3.0, 2.0], [4.0, 1.0]], dtype=np.float64),
    )

    monkeypatch.setattr(
        clustering_module,
        "_build_private_clustering_protocol",
        lambda training_config: type(
            "DummyProtocol",
            (),
            {
                "helper_ids": (0, 1),
                "helper_evaluation_points": (11, 12),
                "encoding_config": type(
                    "DummyEncoding",
                    (),
                    {"privacy_threshold": 2, "resolved_reconstruction_threshold": 3},
                )(),
                "field_config": type("DummyField", (), {"modulus": 2_147_483_647})(),
                "vector_scale": 256,
                "distance_scale": 65_536,
            },
        )(),
    )

    def fake_initialize_centroids(*, seed, **kwargs):
        restart_counter["value"] += 1
        restart_index = restart_counter["value"]
        return (
            np.asarray(
                [
                    [float(restart_index), 0.0],
                    [0.0, float(restart_index)],
                ],
                dtype=np.float64,
            ),
            (restart_index,),
        )

    monkeypatch.setattr(clustering_module, "_initialize_centroids", fake_initialize_centroids)

    def fake_reconstruct_distances(self, shared_reduced_vectors, centroids, protocol):
        restart_index = int(round(float(centroids[0, 0])))
        return restart_distance_matrices[restart_index], protocol.helper_ids, protocol.helper_evaluation_points

    monkeypatch.setattr(SecureKMeansClusterer, "_reconstruct_distances", fake_reconstruct_distances)
    monkeypatch.setattr(
        SecureKMeansClusterer,
        "_recompute_centroids",
        lambda self, shared_reduced_vectors, labels, previous_centroids, protocol, n_clusters, round_seed: previous_centroids,
    )

    result = clusterer.cluster(
        shared_vectors,
        projection_spec=projection_spec,
        seed=7,
        clustering_config=config.clustering,
        initial_vectors=None,
    )

    assert result.initial_centroid_indices == (1,)
    assert result.secure_metadata["num_restarts"] == 3
    assert result.secure_metadata["best_restart_index"] == 1
    assert result.secure_metadata["best_restart_seed"] == 8
    assert np.isclose(result.secure_metadata["best_objective"], 0.3)
    assert np.array_equal(result.labels, np.asarray([0, 0], dtype=np.int64))


def test_random_projection_spec_is_seeded_and_deterministic() -> None:
    spec_a = build_random_projection_spec(input_dimension=3, requested_components=8, seed=13)
    spec_b = build_random_projection_spec(input_dimension=3, requested_components=8, seed=13)
    spec_c = build_random_projection_spec(input_dimension=3, requested_components=8, seed=14)

    assert spec_a.projection_matrix.shape == (3, 8)
    assert np.allclose(spec_a.projection_matrix, spec_b.projection_matrix)
    assert not np.allclose(spec_a.projection_matrix, spec_c.projection_matrix)


def test_centered_pca_projection_spec_uses_compact_svd_for_tall_parameter_space() -> None:
    flattened_vectors = np.asarray(
        [
            [1.0, 0.0, 2.0, 0.0, 3.0],
            [0.0, 1.0, 1.0, 0.0, 2.0],
            [2.0, 1.0, 3.0, 1.0, 4.0],
        ],
        dtype=np.float64,
    )

    spec = build_centered_pca_projection_spec(
        flattened_vectors=flattened_vectors,
        requested_components=4,
        seed=17,
    )

    assert spec.projection_matrix.shape == (5, 3)
    assert spec.actual_components == 3
    assert spec.fit_client_count == 3
    assert spec.explained_variance.shape == (3,)
    assert spec.explained_variance_ratio.shape == (3,)
    assert np.isclose(np.sum(spec.explained_variance_ratio), 1.0)


def test_identity_projection_spec_returns_raw_flattened_space() -> None:
    spec = build_identity_projection_spec(input_dimension=5)
    vector = np.asarray([1.0, -2.0, 3.0, 0.5, 4.0], dtype=np.float64)

    transformed = spec.transform(vector)

    assert isinstance(spec, IdentityProjectionSpec)
    assert spec.actual_components == 5
    assert np.allclose(transformed, vector)
    assert spec.to_metadata()["projection_type"] == "identity_no_projection"


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow is required for Parquet artifact tests.")
@pytest.mark.skipif(not LCC_AVAILABLE, reason="lcc-lib is required for clustered recommender tests.")
def test_clustered_recommender_training_uses_seeded_random_projection_and_secure_cluster_aggregation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, run_id, selection, persona = _prepare_recommender_run(tmp_path, client_count=4)

    fake_local_parameters = {
        "client_000": [np.asarray([1.0, 2.0], dtype=np.float64), np.asarray([0.1], dtype=np.float64)],
        "client_001": [np.asarray([1.1, 2.1], dtype=np.float64), np.asarray([0.2], dtype=np.float64)],
        "client_002": [np.asarray([-1.0, -2.0], dtype=np.float64), np.asarray([-0.1], dtype=np.float64)],
        "client_003": [np.asarray([3.0, 3.5], dtype=np.float64), np.asarray([0.4], dtype=np.float64)],
    }
    projection_spec_calls: list[tuple[tuple[int, int], int, int]] = []
    captured_projector_calls: list[tuple[str, int]] = []
    secure_aggregate_calls: list[tuple[int, tuple[str, ...]]] = []
    import fed_perso_xai.fl.client as client_module
    import fed_perso_xai.fl.recommender_simulation as recommender_simulation
    import fed_perso_xai.recommender.clustering as clustering_module
    from lcc_lib.aggregation import EncodedShareAggregator as LCCEncodedShareAggregator

    def fake_train_shared_payload(self, parameters):
        return (
            client_module.extract_shared_parameter_payload(fake_local_parameters[self.data.client_name]),
            0.01,
        )

    original_build = ClientSideRandomProjector.build_private_reduced_vector_from_flat_vector
    original_projection_builder = recommender_simulation.build_centered_pca_projection_spec

    def spy_projection_builder(*, flattened_vectors, requested_components, seed):
        projection_spec_calls.append((tuple(int(value) for value in flattened_vectors.shape), int(requested_components), int(seed)))
        return original_projection_builder(
            flattened_vectors=flattened_vectors,
            requested_components=requested_components,
            seed=seed,
        )

    def spy_build(self, *, client_id, flat_vector, projection_spec, round_id):
        captured_projector_calls.append((str(client_id), int(round_id)))
        return original_build(
            self,
            client_id=client_id,
            flat_vector=flat_vector,
            projection_spec=projection_spec,
            round_id=round_id,
        )

    def fake_cluster(self, shared_reduced_vectors, *, projection_spec, seed, clustering_config, initial_vectors=None):
        assert clustering_config.k == 3
        assert clustering_config.pca_components == 8
        assert isinstance(projection_spec, PCAProjectionSpec)
        assert projection_spec.seed in {13, 14}
        assert projection_spec.projection_matrix.shape == (3, 3)
        assert projection_spec.mean_vector.shape == (3,)
        assert all(isinstance(item, SecretSharedReducedVector) for item in shared_reduced_vectors)
        assert all(not hasattr(item, "reduced_vector") for item in shared_reduced_vectors)
        assert all(item.dimension == 3 for item in shared_reduced_vectors)
        labels = np.asarray([0, 0, 1, 2], dtype=np.int64)
        return SecureClusterAssignments(
            labels=labels,
            centroids=np.zeros((3, shared_reduced_vectors[0].dimension), dtype=np.float64),
            iterations=2,
            initial_centroid_indices=tuple(),
            secure_metadata={
                "method": clustering_config.method,
                "seed": int(seed),
                "iterations": 2,
                "n_clusters": 3,
                "helper_count": 5,
                "helper_ids": [0, 1, 2, 3, 4],
                "helper_evaluation_points": [11, 12, 13, 14, 15],
                "server_observes_raw_weights": False,
                "server_observes_reduced_vectors": False,
                "server_observes_reconstructed_distances": True,
            },
        )

    original_secure_aggregate = LCCEncodedShareAggregator.aggregate_encoded

    def spy_secure_aggregate(self, encoded_updates, round_id):
        secure_aggregate_calls.append(
            (int(round_id), tuple(str(update.client_id) for update in encoded_updates))
        )
        return original_secure_aggregate(self, encoded_updates, round_id=round_id)

    monkeypatch.setattr(client_module.FederatedPairwiseRecommenderClient, "_train_shared_payload", fake_train_shared_payload)
    monkeypatch.setattr(recommender_simulation, "build_centered_pca_projection_spec", spy_projection_builder)
    monkeypatch.setattr(clustering_module.ClientSideRandomProjector, "build_private_reduced_vector_from_flat_vector", spy_build)
    monkeypatch.setattr(clustering_module.SecureKMeansClusterer, "cluster", fake_cluster)
    monkeypatch.setattr(LCCEncodedShareAggregator, "aggregate_encoded", spy_secure_aggregate)

    artifacts, metadata = train_federated_recommender(
        RecommenderFederatedTrainingConfig(
            run_id=run_id,
            selection_id=selection,
            persona=persona,
            paths=paths,
            recommender_type="svm_rank",
            rounds=2,
            epochs=2,
            batch_size=2,
            learning_rate=0.1,
            seed=13,
            top_k=(1, 2),
            clustering=RecommenderClusteringConfig(enabled=True),
        )
    )

    assert metadata["clustered"] is True
    assert metadata["training_variant"] == "clustered"
    assert artifacts.run_dir.name == "clustered"
    assert projection_spec_calls == [((4, 3), 8, 13), ((4, 3), 8, 14)]
    assert len(captured_projector_calls) == 8
    assert {call[0] for call in captured_projector_calls} == {
        "client_000",
        "client_001",
        "client_002",
        "client_003",
    }
    assert {call[1] for call in captured_projector_calls} == {1, 2}
    assert len(secure_aggregate_calls) == 2
    assert {call[1] for call in secure_aggregate_calls} == {
        ("client_000", "client_001"),
    }

    manifest = json.loads(artifacts.cluster_manifest_path.read_text(encoding="utf-8"))
    assert manifest["k"] == 3
    assert manifest["pca_components"] == 8
    assert manifest["normalize_clustering_vector"] is True
    assert manifest["clustering_normalization_mode"] == "l2"
    assert manifest["delta_over_base_norm"] is True
    assert np.isclose(manifest["assignment_margin"], 0.05)
    assert manifest["num_restarts"] == 5
    assert set(manifest["final_cluster_model_checkpoint_paths"]) == {"0", "1", "2"}

    round_one = json.loads((artifacts.cluster_rounds_dir / "round_0001.json").read_text(encoding="utf-8"))
    assert round_one["assignments"] == {
        "client_000": 0,
        "client_001": 0,
        "client_002": 1,
        "client_003": 2,
    }
    assert round_one["cluster_sizes"] == {"0": 2, "1": 1, "2": 1}
    assert round_one["projection"]["projection_applied"] == "client_side"
    assert round_one["projection"]["projection_type"] == "pca_covariance_eigh"
    assert round_one["projection"]["projection_seed"] == 13
    assert round_one["projection"]["data_dependent_fit"] is True
    assert round_one["projection"]["centering_applied"] is True
    assert round_one["projection"]["fit_client_count"] == 4
    assert round_one["projection"]["actual_components"] == 3
    assert round_one["projection"]["normalize_clustering_vector"] is True
    assert round_one["projection"]["clustering_normalization_mode"] == "l2"
    assert round_one["projection"]["delta_over_base_norm"] is True
    assert np.isclose(round_one["projection"]["assignment_margin"], 0.05)
    assert round_one["secure_clustering"]["server_observes_raw_weights"] is False
    assert round_one["secure_clustering"]["server_observes_reduced_vectors"] is False
    assert np.isclose(round_one["secure_clustering"]["assignment_margin"], 0.05)
    assert round_one["secure_clustering"]["hysteresis_retained_client_count"] == 0
    assert round_one["secure_aggregation_per_cluster"]["0"]["mode"] == "secure"
    assert round_one["secure_aggregation_per_cluster"]["0"]["num_contributors"] == 2
    assert round_one["secure_aggregation_per_cluster"]["1"]["mode"] == "carry_forward_underpopulated_cluster"
    assert round_one["secure_aggregation_per_cluster"]["1"]["num_contributors"] == 1
    assert round_one["secure_aggregation_per_cluster"]["1"]["client_ids"] == ["client_002"]
    assert round_one["secure_aggregation_per_cluster"]["2"]["mode"] == "carry_forward_underpopulated_cluster"
    assert round_one["secure_aggregation_per_cluster"]["2"]["num_contributors"] == 1
    assert round_one["secure_aggregation_per_cluster"]["2"]["client_ids"] == ["client_003"]
    assert len(round_one["cluster_model_checkpoint_paths"]) == 3
    for relative_path in round_one["cluster_model_checkpoint_paths"].values():
        assert (artifacts.run_dir / relative_path).exists()

    evaluation = json.loads(artifacts.evaluation_summary_path.read_text(encoding="utf-8"))
    assert evaluation["status"] == "skipped_no_validation_pairs"
    assert evaluation["training_variant"] == "clustered"
    assert evaluation["aggregate"] == {}
    assert evaluation["clients"] == []


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow is required for Parquet artifact tests.")
@pytest.mark.skipif(not LCC_AVAILABLE, reason="lcc-lib is required for clustered recommender tests.")
def test_clustered_recommender_training_can_skip_pca(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, run_id, selection, persona = _prepare_recommender_run(tmp_path, client_count=3)

    import fed_perso_xai.fl.recommender_simulation as recommender_simulation
    import fed_perso_xai.recommender.clustering as clustering_module

    def fail_if_pca_builder_called(**kwargs):
        raise AssertionError("PCA builder should not be called when clustering.enable_pca is False.")

    def fake_cluster(self, shared_reduced_vectors, *, projection_spec, seed, clustering_config, initial_vectors=None):
        assert clustering_config.enable_pca is False
        assert isinstance(projection_spec, IdentityProjectionSpec)
        assert all(item.dimension == 3 for item in shared_reduced_vectors)
        labels = np.asarray([0, 1, 2], dtype=np.int64)
        return SecureClusterAssignments(
            labels=labels,
            centroids=np.zeros((3, shared_reduced_vectors[0].dimension), dtype=np.float64),
            iterations=1,
            initial_centroid_indices=tuple(),
            secure_metadata={
                "method": clustering_config.method,
                "seed": int(seed),
                "iterations": 1,
                "n_clusters": 3,
                "helper_count": 5,
                "helper_ids": [0, 1, 2, 3, 4],
                "helper_evaluation_points": [11, 12, 13, 14, 15],
                "server_observes_raw_weights": False,
                "server_observes_reduced_vectors": False,
                "server_observes_reconstructed_distances": True,
            },
        )

    monkeypatch.setattr(recommender_simulation, "build_centered_pca_projection_spec", fail_if_pca_builder_called)
    monkeypatch.setattr(clustering_module.SecureKMeansClusterer, "cluster", fake_cluster)

    artifacts, metadata = train_federated_recommender(
        RecommenderFederatedTrainingConfig(
            run_id=run_id,
            selection_id=selection,
            persona=persona,
            paths=paths,
            rounds=2,
            epochs=2,
            batch_size=2,
            learning_rate=0.1,
            seed=13,
            top_k=(1, 2),
            clustering=RecommenderClusteringConfig(enabled=True, enable_pca=False),
        )
    )

    assert metadata["clustered"] is True
    manifest = json.loads(artifacts.cluster_manifest_path.read_text(encoding="utf-8"))
    assert manifest["enable_pca"] is False
    runtime_report = json.loads(artifacts.runtime_report_path.read_text(encoding="utf-8"))
    assert runtime_report["server_observes_raw_weights_during_clustering"] is False
    round_one = json.loads((artifacts.cluster_rounds_dir / "round_0001.json").read_text(encoding="utf-8"))
    assert round_one["projection"]["enable_pca"] is False
    assert round_one["projection"]["projection_type"] == "identity_no_projection"
    assert round_one["projection"]["data_dependent_fit"] is False
    assert round_one["projection"]["projection_fit_round_id"] is None


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow is required for Parquet artifact tests.")
@pytest.mark.skipif(not LCC_AVAILABLE, reason="lcc-lib is required for clustered recommender tests.")
def test_clustered_recommender_training_reports_raw_weight_visibility_when_pca_is_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, run_id, selection, persona = _prepare_recommender_run(tmp_path, client_count=3)

    import fed_perso_xai.recommender.clustering as clustering_module

    def fake_cluster(self, shared_reduced_vectors, *, projection_spec, seed, clustering_config, initial_vectors=None):
        assert clustering_config.enable_pca is True
        assert isinstance(projection_spec, PCAProjectionSpec)
        labels = np.asarray([0, 1, 2], dtype=np.int64)
        return SecureClusterAssignments(
            labels=labels,
            centroids=np.zeros((3, shared_reduced_vectors[0].dimension), dtype=np.float64),
            iterations=1,
            initial_centroid_indices=tuple(),
            secure_metadata={
                "method": clustering_config.method,
                "seed": int(seed),
                "iterations": 1,
                "n_clusters": 3,
                "helper_count": 5,
                "helper_ids": [0, 1, 2, 3, 4],
                "helper_evaluation_points": [11, 12, 13, 14, 15],
                "server_observes_raw_weights": False,
                "server_observes_reduced_vectors": False,
                "server_observes_reconstructed_distances": True,
            },
        )

    monkeypatch.setattr(clustering_module.SecureKMeansClusterer, "cluster", fake_cluster)

    artifacts, metadata = train_federated_recommender(
        RecommenderFederatedTrainingConfig(
            run_id=run_id,
            selection_id=selection,
            persona=persona,
            paths=paths,
            rounds=2,
            epochs=2,
            batch_size=2,
            learning_rate=0.1,
            seed=13,
            top_k=(1, 2),
            clustering=RecommenderClusteringConfig(enabled=True, enable_pca=True),
        )
    )

    assert metadata["clustered"] is True
    runtime_report = json.loads(artifacts.runtime_report_path.read_text(encoding="utf-8"))
    assert runtime_report["server_observes_raw_weights_during_clustering"] is True
    round_one = json.loads((artifacts.cluster_rounds_dir / "round_0001.json").read_text(encoding="utf-8"))
    assert round_one["projection"]["server_observes_raw_weights_during_projection_fit"] is True


@pytest.mark.parametrize("recommender_type", ["svm_rank", "pairwise_logistic"])
@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow is required for Parquet artifact tests.")
@pytest.mark.skipif(not LCC_AVAILABLE, reason="lcc-lib is required for clustered recommender tests.")
def test_clustered_recommender_training_supports_both_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recommender_type: str,
) -> None:
    paths, run_id, selection, persona = _prepare_recommender_run(tmp_path, client_count=3)

    import fed_perso_xai.recommender.clustering as clustering_module

    def fake_cluster(self, shared_reduced_vectors, *, projection_spec, seed, clustering_config, initial_vectors=None):
        labels = np.asarray([0, 1, 2], dtype=np.int64)
        return SecureClusterAssignments(
            labels=labels,
            centroids=np.zeros((3, shared_reduced_vectors[0].dimension), dtype=np.float64),
            iterations=1,
            initial_centroid_indices=tuple(),
            secure_metadata={
                "method": clustering_config.method,
                "seed": int(seed),
                "iterations": 1,
                "n_clusters": 3,
                "helper_count": 5,
                "helper_ids": [0, 1, 2, 3, 4],
                "helper_evaluation_points": [11, 12, 13, 14, 15],
                "server_observes_raw_weights": False,
                "server_observes_reduced_vectors": False,
                "server_observes_reconstructed_distances": True,
            },
        )

    monkeypatch.setattr(clustering_module.SecureKMeansClusterer, "cluster", fake_cluster)

    artifacts, metadata = train_federated_recommender(
        RecommenderFederatedTrainingConfig(
            run_id=run_id,
            selection_id=selection,
            persona=persona,
            recommender_type=recommender_type,
            paths=paths,
            rounds=2,
            epochs=3,
            batch_size=2,
            learning_rate=0.2,
            seed=7,
            top_k=(1, 2),
            clustering=RecommenderClusteringConfig(enabled=True),
        )
    )

    assert artifacts.cluster_manifest_path.exists()
    assert metadata["status"] == "completed"
    assert metadata["clustered"] is True
    assert metadata["training_variant"] == "clustered"
    assert artifacts.run_dir.name == "clustered"
    assert metadata["recommender_type"] == recommender_type


def test_recommender_clustering_config_defaults_and_validation() -> None:
    config = RecommenderClusteringConfig(enabled=True)
    assert config.method == "secure_kmeans"
    assert config.k == 3
    assert config.normalize_clustering_vector is True
    assert config.clustering_normalization_mode == "l2"
    assert config.delta_over_base_norm is True
    assert np.isclose(config.assignment_margin, 0.05)
    assert config.num_restarts == 5
    assert config.enable_pca is True
    assert config.pca_components == 8

    with pytest.raises(ValueError, match="Unsupported clustering.method"):
        RecommenderClusteringConfig(enabled=True, method="missing")
    with pytest.raises(ValueError, match="Unsupported clustering.normalization_mode"):
        RecommenderClusteringConfig(enabled=True, clustering_normalization_mode="invalid")
    with pytest.raises(ValueError, match="assignment_margin must be less than 1"):
        RecommenderClusteringConfig(enabled=True, assignment_margin=1.0)
