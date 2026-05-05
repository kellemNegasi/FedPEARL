from __future__ import annotations

import numpy as np
import pytest

from fed_perso_xai.recommender.model import (
    PairwiseLogisticConfig,
    PairwiseLogisticRecommender,
    SVMRankRecommender,
    load_pairwise_logistic_recommender,
    load_svm_rank_recommender,
)
from fed_perso_xai.utils.config import RecommenderFederatedTrainingConfig


def _training_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(
        [
            [2.0, 1.0],
            [1.5, 0.5],
            [1.0, 1.0],
            [-1.0, -1.0],
            [-1.5, -0.5],
            [-2.0, -1.0],
        ],
        dtype=np.float64,
    )
    y = np.asarray([1, 1, 1, 0, 0, 0], dtype=np.int64)
    return X, y


@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam"])
def test_pairwise_logistic_supported_optimizers_reduce_loss(optimizer: str) -> None:
    X, y = _training_data()
    config = PairwiseLogisticConfig(
        epochs=30,
        batch_size=2,
        optimizer=optimizer,
        learning_rate=0.05 if optimizer != "adam" else 0.01,
    )
    model = PairwiseLogisticRecommender.from_config(n_features=X.shape[1], config=config)

    initial_loss = model.loss(X, y)
    trained_loss = model.fit(X, y, seed=7)

    assert trained_loss < initial_loss
    assert model.optimizer == optimizer


@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam"])
def test_svm_rank_supported_optimizers_train_without_error(optimizer: str) -> None:
    X, y = _training_data()
    config = PairwiseLogisticConfig(
        epochs=20,
        batch_size=2,
        optimizer=optimizer,
        learning_rate=0.05 if optimizer != "adam" else 0.01,
        svm_c=0.5,
    )
    model = SVMRankRecommender.from_config(n_features=X.shape[1], config=config)

    initial_loss = model.loss(X, y)
    trained_loss = model.fit(X, y, seed=11)

    assert trained_loss < initial_loss
    assert model.optimizer == optimizer


def test_recommender_config_rejects_unknown_optimizer() -> None:
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        PairwiseLogisticConfig(optimizer="rmsprop")

    with pytest.raises(ValueError, match="Unsupported optimizer"):
        RecommenderFederatedTrainingConfig(
            run_id="run",
            selection_id="sel",
            persona="persona",
            optimizer="rmsprop",
        )


def test_pairwise_logistic_roundtrip_persists_optimizer(tmp_path) -> None:
    X, y = _training_data()
    config = PairwiseLogisticConfig(
        epochs=10,
        batch_size=2,
        optimizer="adagrad",
        learning_rate=0.05,
    )
    model = PairwiseLogisticRecommender.from_config(n_features=X.shape[1], config=config)
    model.fit(X, y, seed=5)

    path = tmp_path / "pairwise_logistic.npz"
    model.save(path)
    loaded = load_pairwise_logistic_recommender(path)

    assert loaded.optimizer == "adagrad"
    np.testing.assert_allclose(loaded.get_parameters()[0], model.get_parameters()[0])
    np.testing.assert_allclose(loaded.get_parameters()[1], model.get_parameters()[1])


def test_svm_rank_roundtrip_persists_optimizer(tmp_path) -> None:
    X, y = _training_data()
    config = PairwiseLogisticConfig(
        epochs=10,
        batch_size=2,
        optimizer="adam",
        learning_rate=0.01,
        svm_c=0.5,
    )
    model = SVMRankRecommender.from_config(n_features=X.shape[1], config=config)
    model.fit(X, y, seed=13)

    path = tmp_path / "svm_rank.npz"
    model.save(path)
    loaded = load_svm_rank_recommender(path)

    assert loaded.optimizer == "adam"
    np.testing.assert_allclose(loaded.get_parameters()[0], model.get_parameters()[0])
    np.testing.assert_allclose(loaded.get_parameters()[1], model.get_parameters()[1])
