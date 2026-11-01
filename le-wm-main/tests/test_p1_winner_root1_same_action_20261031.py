import pytest

from experiments.spectral_reachability import p1_winner_root1_same_action_20261031 as subject


def test_actual_root1_nodes_have_matched_initial_context_and_complete_roster():
    if not subject.RUN.exists():
        pytest.skip("local physical result unavailable")
    subject.configure()
    _, _, nodes = subject.saved_inputs()
    assert len(nodes) == 16
    initial = [node["task"] for node in nodes if node["boundary"] == 0]
    assert len(initial) == 4
    for task in initial[1:]:
        for key in ("root_context", "goal", "action_mean", "action_std"):
            assert task[key] == initial[0][key]
