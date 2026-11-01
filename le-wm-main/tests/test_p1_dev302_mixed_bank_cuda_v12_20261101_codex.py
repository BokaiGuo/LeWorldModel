import hashlib

import numpy as np

from experiments.spectral_reachability import (
    p1_dev302_mixed_bank_cuda_v12_20261101_codex as subject,
)


def test_v12_all_candidates_are_action_bounded_and_geometry_recomputed():
    value = subject.protocol()
    rows = value["prefix_bank"]["candidates"]
    assert len(rows) == 17
    assert value["prefix_bank"]["contact_guard"][
        "v8_out_of_bounds_candidates_repaired"
    ] == [10, 12]
    for row in rows:
        actions = np.asarray(row["actions"], np.float32)
        assert actions.shape == (5, 2)
        assert np.max(np.abs(actions)) <= 1.0
        assert hashlib.sha256(actions.astype("<f4").tobytes()).hexdigest() == row[
            "actions_sha256"
        ]
        assert np.asarray(row["rgb_dynamic_predicted_pusher_centers_xy"]).shape == (
            5,
            2,
        )
    assert value["P1_admission"] is False
