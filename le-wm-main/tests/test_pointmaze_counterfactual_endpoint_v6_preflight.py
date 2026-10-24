import numpy as np

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as panel
from experiments.gdd_wm.preflight_pointmaze_counterfactual_endpoint_v6 import _collision_pairs, _panel_instance_id
from experiments.gdd_wm.run_pointmaze_counterfactual_endpoint_control import _calibration_role


def test_static_preflight_collision_counter_accepts_only_distinct_canonical_summary_aliases() -> None:
    bank = panel.state_interaction_pair_proposals(seed=19, count=8).reshape(16, panel.TOTAL_STEPS, 2)
    assert _collision_pairs(bank) == 8
    no_alias_bank = bank.copy()
    no_alias_bank[1, 0, 0] += 0.01
    assert _collision_pairs(no_alias_bank) < 8


def test_static_preflight_collision_counter_rejects_invalid_shape() -> None:
    with np.testing.assert_raises_regex(ValueError, "candidate bank"):
        _collision_pairs(np.zeros((2, 19, 2)))


def test_static_preflight_hashes_the_exact_panel_instance_identity_for_roles() -> None:
    # A bare episode id and the V6 runner's emitted root identity need not hash
    # to the same calibration role.  The preflight must use the latter.
    root_id = next(
        value for value in range(1000)
        if _calibration_role(str(value)) != _calibration_role(_panel_instance_id(value))
    )
    assert _panel_instance_id(root_id) == f"pointmaze_v6_episode_{root_id}"
    assert _calibration_role(str(root_id)) != _calibration_role(_panel_instance_id(root_id))
