from experiments.spectral_reachability import (
    p1_dev302_mixed_bank_cuda_v6_20261101_codex as subject,
)


def test_v6_adds_delayed_task_support_and_rejects_wrong_face():
    value = subject.protocol()
    rows = {row["label"]: row for row in value["prefix_bank"]["candidates"]}
    assert len(rows) == 9
    assert rows["rgb_goal_support_top_route"]["contact_feasible_prior"] is False
    assert rows["rgb_goal_support_direct"]["contact_feasible_prior"] is True
    assert rows["rgb_delayed_task_support"]["contact_feasible_prior"] is True
    assert rows["rgb_delayed_task_support"]["task_support_alignment"] >= 0.5
    assert value["algorithm"]["prefix_candidates"] == 9
    assert value["P1_admission"] is False
