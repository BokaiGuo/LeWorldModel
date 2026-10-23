import numpy as np
import pytest

import experiments.run_pusht_balanced_temporal_torque_confirmation as balanced
from experiments.run_pusht_balanced_temporal_torque_confirmation import (
    BalancedCohortUnavailable,
    RootExample,
    assert_root_separation,
    confirmation_rule,
    offline_preference,
    matched_probe_reset_seed,
    root_feature_temporal_history,
    root_feature_two_frame,
    select_balanced,
    validate_seed_bands,
)


def _row(root_id: int, label: int) -> RootExample:
    frames = np.zeros((5, 12, 14, 3), dtype=np.uint8)
    frames[1, 1:4, 1:4, 0] = 10
    frames[2, 2:5, 2:5, 1] = 30
    frames[3, 3:6, 3:6, 2] = 60
    frames[4, 4:7, 4:7, :] = 90
    actions = np.asarray(((0.3, 0.0), (0.0, 0.3), (-0.3, 0.0), (0.0, -0.3)), dtype=np.float32)
    return RootExample(root_id, f"digest-{root_id}", frames, actions, label, 0.01 if label else -0.01, 0.02, 0.01)


def test_preference_is_duration_agnostic_but_zero_drift_corrected_and_margin_gated():
    label, margin, plus, minus = offline_preference(
        before_angle=0.0, zero_angle=0.1, plus_angle=0.5, minus_angle=-0.1, minimum_margin=0.01
    )
    assert label == 1
    assert margin > 0.0 and plus > minus
    assert offline_preference(
        before_angle=0.0, zero_angle=0.1, plus_angle=0.101, minus_angle=0.099, minimum_margin=0.01
    )[0] is None


def test_two_fixed_feature_maps_are_label_independent_and_temporal_uses_more_history():
    positive = _row(1, 1)
    negative = _row(1, 0)
    raw_first = root_feature_two_frame(positive.rgb_frames, positive.history_actions, 5)
    raw_second = root_feature_two_frame(negative.rgb_frames, negative.history_actions, 5)
    temporal_first = root_feature_temporal_history(positive.rgb_frames, positive.history_actions, 5)
    temporal_second = root_feature_temporal_history(negative.rgb_frames, negative.history_actions, 5)
    assert np.array_equal(raw_first, raw_second)
    assert np.array_equal(temporal_first, temporal_second)
    assert temporal_first.size > raw_first.size


def test_balanced_sampling_happens_before_fitting_and_rejects_missing_class():
    rows = [_row(index, index % 2) for index in range(12)]
    selected = select_balanced(rows, per_class=4, split="test", seed=9)
    assert sum(row.preference == 0 for row in selected) == 4
    assert sum(row.preference == 1 for row in selected) == 4
    with pytest.raises(RuntimeError, match="label 1"):
        select_balanced([_row(99, 0)], per_class=2, split="test", seed=9)


def test_counterfactual_probe_arms_share_a_single_reset_seed_per_root():
    seed = matched_probe_reset_seed(1_701_210_000, 37)
    assert seed == matched_probe_reset_seed(1_701_210_000, 37)
    assert seed != matched_probe_reset_seed(1_701_210_000, 38)
    # The contract is intentionally one seed, reused for zero/plus/minus,
    # rather than an action-indexed seed that could inject reset randomness.
    assert [seed, seed, seed] == [matched_probe_reset_seed(1_701_210_000, 37)] * 3


def test_unavailable_balanced_cohort_carries_aggregate_receipt_only():
    error = BalancedCohortUnavailable("missing label", collection={"split": "train", "eligible_by_label_before_sampling": {"0": 2, "1": 0}})
    assert str(error) == "missing label"
    assert error.collection["split"] == "train"


def test_collect_band_resets_every_counterfactual_arm_with_the_same_root_seed(monkeypatch):
    """Only the pulse action may vary within a zero/plus/minus triplet."""

    class FakeEnv:
        def __init__(self) -> None:
            self.reset_seeds: list[int] = []
            self.state = np.zeros(7, dtype=np.float64)

        def reset(self, *, seed, options):
            self.reset_seeds.append(int(seed))
            self.state = np.asarray(options["state"], dtype=np.float64).copy()
            return {"state": self.state.copy()}, {}

        def render(self):
            return np.zeros((8, 8, 3), dtype=np.uint8)

    env = FakeEnv()
    states = np.zeros((4, 7), dtype=np.float64)
    states[:, 4] = np.asarray((0.0, 1.0, 0.0, 1.0))
    monkeypatch.setattr(balanced, "collect_trajectory_data", lambda *_args: {"states": states})
    monkeypatch.setattr(balanced, "mode_for_state", lambda _state: "contact")
    monkeypatch.setattr(
        balanced,
        "advance",
        lambda active_env, _action, _steps: (active_env.state.copy(), active_env.render()),
    )

    rows, receipt = balanced.collect_band(
        env,
        split="train",
        band_seed=1_701_210_000,
        per_class=2,
        pool_transitions=4,
        history_steps=2,
        history_magnitude=0.3,
        max_contact_gap=65.0,
        minimum_margin=1.0e-4,
        preference_fn=lambda *, before_angle, **_kwargs: (int(before_angle > 0.5), 1.0, 0.1, -0.1),
        probe_action_fn=lambda *_args: (
            {"zero": np.zeros(2), "plus": np.asarray((1.0, 0.0)), "minus": np.asarray((-1.0, 0.0))},
            {"pair_valid": True},
        ),
    )
    assert len(rows) == 4
    assert receipt["counterfactual_reset_contract"] == "zero/plus/minus share one reset seed per source root"
    # Each candidate performs one history reset followed by a three-arm
    # counterfactual triplet.  The three entries must be identical.
    assert len(env.reset_seeds) == 16
    assert all(len(set(env.reset_seeds[index + 1:index + 4])) == 1 for index in range(0, 16, 4))


def test_root_and_seed_separation_fail_closed():
    validate_seed_bands({"train": 1_700_210_000, "calibration": 1_700_220_000, "test": 1_700_230_000})
    with pytest.raises(ValueError, match="distinct"):
        validate_seed_bands({"train": 1_700_210_000, "calibration": 1_700_210_000, "test": 1_700_230_000})
    assert_root_separation([_row(1, 0)], [_row(2, 1)], [_row(3, 0)])
    with pytest.raises(ValueError, match="root-disjoint"):
        assert_root_separation([_row(1, 0)], [_row(1, 1)])


def test_confirmation_requires_threshold_ci_and_paired_gain():
    raw = {"test_balanced_accuracy": {"estimate": 0.60}}
    temporal = {"test_balanced_accuracy": {"estimate": 0.75, "ci95_low": 0.62}}
    delta = {"estimate": 0.15, "ci95_low": 0.01}
    assert confirmation_rule(temporal, raw, delta, minimum_ba=0.70, minimum_ci_lower=0.60)["status"] == "confirmed"
    delta["ci95_low"] = 0.0
    assert confirmation_rule(temporal, raw, delta, minimum_ba=0.70, minimum_ci_lower=0.60)["status"] == "not_confirmed"
