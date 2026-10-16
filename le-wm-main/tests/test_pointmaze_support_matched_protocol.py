import numpy as np

from experiments.empirical_spectral_audit.evaluate_pointmaze_support_matched_external_evidence import (
    _simulate_sequence_effects,
)
from experiments.empirical_spectral_audit.repair_pointmaze_export_replay_labels import (
    _replay_states,
)


def test_support_matched_replay_uses_five_physics_steps_per_action() -> None:
    row = {
        "source_state_raw": [0.4, 2.0, 0.1, -0.1],
        "action_blocks_raw": (np.full((4, 10), 0.05, dtype=np.float32)).tolist(),
    }
    actions = np.asarray(row["action_blocks_raw"], dtype=np.float32)
    effects = _simulate_sequence_effects(
        [row],
        np.stack([actions[None], np.zeros((1, 4, 10)), np.zeros((1, 4, 10))], axis=1),
    )[0, 0]
    replay = _replay_states(row)[[0, 1, 3], :2]
    expected = replay - np.asarray(row["source_state_raw"], dtype=np.float64)[:2]
    np.testing.assert_allclose(effects, expected, atol=1e-10, rtol=0.0)
