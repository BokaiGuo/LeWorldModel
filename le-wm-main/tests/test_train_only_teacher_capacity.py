import numpy as np
import torch

from experiments.gdd_wm.audit_train_only_teacher_capacity import (
    _fit_listwise_mlp_predictions,
)


def test_listwise_capacity_accepts_variable_candidate_cells():
    features = np.asarray([
        [0.0, 0.0], [0.1, 0.0],
        [1.0, 0.0], [1.1, 0.0], [1.2, 0.0],
    ], dtype=np.float64)
    target = np.asarray([0.0, 1.0, 1.0, 0.5, 0.0], dtype=np.float64)
    mask = np.ones(len(features), dtype=bool)
    prediction = _fit_listwise_mlp_predictions(
        features,
        target,
        mask,
        ["cell_a", "cell_a", "cell_b", "cell_b", "cell_b"],
        epochs=2,
        learning_rate=1.0e-2,
        weight_decay=1.0e-3,
        seed=7,
        device=torch.device("cpu"),
    )
    assert prediction.shape == (5,)
    assert np.isfinite(prediction).all()
