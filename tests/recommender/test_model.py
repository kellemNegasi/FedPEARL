from __future__ import annotations

import numpy as np

from fed_perso_xai.recommender import PairwiseLogisticConfig, SVMRankRecommender


def test_svm_rank_loss_matches_sample_normalized_training_objective() -> None:
    config = PairwiseLogisticConfig(
        epochs=1,
        batch_size=2,
        learning_rate=0.1,
        svm_c=2.5,
        svm_intercept_scaling=2.0,
    )
    model = SVMRankRecommender.from_config(n_features=2, config=config)
    model.set_parameters(
        [
            np.asarray([0.5, -0.25], dtype=np.float64),
            np.asarray([0.4], dtype=np.float64),
        ]
    )
    X = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.5],
        ],
        dtype=np.float64,
    )
    y = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float64)

    loss = model.loss(X, y)

    y_signed = np.where(y > 0.5, 1.0, -1.0)
    margins = y_signed * model.predict_pairwise_logits(X)
    squared_hinge = np.square(np.maximum(0.0, 1.0 - margins))
    regularization = 0.5 * (
        float(np.sum(model.weights**2))
        + float((model.bias[0] / model.intercept_scaling) ** 2)
    )
    expected = (regularization / float(X.shape[0])) + (model.svm_c * float(np.mean(squared_hinge)))

    assert np.isclose(loss, expected)


def test_svm_rank_loss_is_invariant_to_dataset_duplication() -> None:
    config = PairwiseLogisticConfig(
        epochs=1,
        batch_size=2,
        learning_rate=0.1,
        svm_c=1.5,
        svm_intercept_scaling=1.0,
    )
    model = SVMRankRecommender.from_config(n_features=2, config=config)
    model.set_parameters(
        [
            np.asarray([0.2, -0.1], dtype=np.float64),
            np.asarray([0.05], dtype=np.float64),
        ]
    )
    X = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    y = np.asarray([1.0, 0.0], dtype=np.float64)

    duplicated_X = np.vstack([X, X])
    duplicated_y = np.concatenate([y, y])

    assert np.isclose(model.loss(X, y), model.loss(duplicated_X, duplicated_y))
