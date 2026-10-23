from __future__ import annotations

import numpy as np

from experiments.gdd_wm.tworoom_support_gated_repair import calibrate, choose_scores


def test_support_gate_prospectively_releases_only_supported_calibration_gain() -> None:
    train = np.asarray([[0.0], [0.1], [0.2]], dtype=np.float64)
    # Each root has two candidate rows. Root 10 is supported and benefits from
    # the full adapter; root 20 is out of support and the fallback is better.
    calibration = np.asarray([[0.05], [0.10], [10.0], [10.1]], dtype=np.float64)
    roots = np.asarray([10, 10, 20, 20], dtype=np.int64)
    gate, receipt = calibrate(
        train_features=train, calibration_features=calibration, calibration_roots=roots,
        adapter_regret={10: 0.0, 20: 2.0}, fallback_regret={10: 1.0, 20: 0.0},
    )
    assert receipt["status"] == "authorized_calibration_support_gate"
    assert receipt["outcome_labels_read"] is False
    chosen, released = choose_scores(gate, np.asarray([[0.0], [0.1]]), np.asarray([0.0, 1.0]), np.asarray([2.0, 3.0]))
    assert released is True and np.allclose(chosen, [0.0, 1.0])
    chosen, released = choose_scores(gate, np.asarray([[20.0], [20.1]]), np.asarray([0.0, 1.0]), np.asarray([2.0, 3.0]))
    assert released is False and np.allclose(chosen, [2.0, 3.0])


def test_support_gate_fails_closed_when_no_calibration_release_has_positive_gain() -> None:
    train = np.asarray([[0.0], [0.1]], dtype=np.float64)
    features = np.asarray([[0.0], [0.1], [2.0], [2.1]], dtype=np.float64)
    roots = np.asarray([1, 1, 2, 2], dtype=np.int64)
    gate, receipt = calibrate(
        train_features=train, calibration_features=features, calibration_roots=roots,
        adapter_regret={1: 2.0, 2: 2.0}, fallback_regret={1: 1.0, 2: 1.0},
    )
    assert receipt["status"] == "rejected_calibration_support_gate"
    _, released = choose_scores(gate, np.asarray([[0.0], [0.1]]), np.asarray([0.0, 1.0]), np.asarray([2.0, 3.0]))
    assert released is False
