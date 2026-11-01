from experiments.spectral_reachability import (
    p1_dev302_mixed_bank_cuda_v2_20261101_codex as subject,
)


def test_v2_rejects_spectral_false_positive_and_keeps_rgb_candidates():
    value = subject.protocol()
    rows = {row["label"]: row for row in value["prefix_bank"]["candidates"]}
    assert rows["spectral"]["contact_feasible_prior"] is False
    assert rows["random_control"]["contact_feasible_prior"] is False
    assert rows["gn"]["contact_feasible_prior"] is True
    assert all(
        rows[label]["contact_feasible_prior"]
        for label in (
            "rgb_nearest_contact",
            "rgb_goal_support_direct",
            "rgb_goal_support_top_route",
            "rgb_goal_support_bottom_route",
        )
    )
    assert value["P1_admission"] is False
