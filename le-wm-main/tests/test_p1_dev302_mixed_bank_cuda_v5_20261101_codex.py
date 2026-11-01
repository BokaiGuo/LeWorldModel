from experiments.spectral_reachability import (
    p1_dev302_mixed_bank_cuda_v4_20261101_codex as v4,
)
from experiments.spectral_reachability import (
    p1_dev302_mixed_bank_cuda_v5_20261101_codex as subject,
)


def _rows(value):
    return {row["label"]: row for row in value["prefix_bank"]["candidates"]}


def test_v5_keeps_v4_actions_and_uses_memory_guard():
    old = _rows(v4.protocol())
    value = subject.protocol()
    new = _rows(value)
    assert set(old) == set(new)
    for label in old:
        assert old[label]["actions_sha256"] == new[label]["actions_sha256"]
    guard = value["prefix_bank"]["contact_guard"]
    assert guard["kind"].startswith("rgb_development_calibrated_one_step_memory")
    assert guard["contact_center_to_object_pixels"] == 6.5
    assert guard["simulator_state_used_at_deployment"] is False
    assert value["algorithm"]["tail_randomization"] == {
        "kind": "common_random_numbers",
        "seed": subject.runner.SEED,
        "same_sample_stream_for_every_prefix": True,
    }


def test_v5_contact_roster_matches_development_physics_boundary():
    rows = _rows(subject.protocol())
    accepted = {
        label for label, row in rows.items() if row["contact_feasible_prior"]
    }
    assert accepted == {
        "rgb_goal_support_direct",
        "rgb_goal_support_top_route",
        "rgb_goal_support_bottom_route",
    }
    assert rows["rgb_goal_support_top_route"][
        "rgb_dynamic_minimum_object_center_to_pixel_gap"
    ] < 6.5
    assert rows["gn"]["rgb_dynamic_minimum_object_center_to_pixel_gap"] > 6.5
    assert subject.protocol()["P1_admission"] is False


def test_common_rng_patch_preserves_fixed_random_control_protocol():
    before = subject.protocol()
    original = subject.runner.np.random.default_rng

    def patched(seed=None):
        if seed == subject.bank_v5.v4.geometry.RANDOM_SEED:
            return original(seed)
        return original(subject.runner.SEED)

    subject.runner.np.random.default_rng = patched
    try:
        after = subject.protocol()
    finally:
        subject.runner.np.random.default_rng = original
    assert after == before
