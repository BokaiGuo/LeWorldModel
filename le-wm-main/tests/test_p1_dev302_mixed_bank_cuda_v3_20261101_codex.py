from experiments.spectral_reachability import (
    p1_dev302_mixed_bank_cuda_v3_20261101_codex as subject,
)


def test_v3_rejects_gn_and_admits_goal_support_paths():
    value = subject.protocol()
    rows = {row["label"]: row for row in value["prefix_bank"]["candidates"]}
    assert rows["gn"]["contact_feasible_prior"] is False
    assert rows["spectral"]["contact_feasible_prior"] is False
    assert rows["zero"]["contact_feasible_prior"] is False
    assert rows["random_control"]["contact_feasible_prior"] is False
    assert rows["rgb_goal_support_direct"]["contact_feasible_prior"] is True
    assert rows["rgb_goal_support_top_route"]["contact_feasible_prior"] is True
    assert value["prefix_bank"]["contact_guard"]["calibration"].startswith("historical RGB")
    assert value["P1_admission"] is False
