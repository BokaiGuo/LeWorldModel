import numpy as np
import pytest

from experiments.spectral_reachability import p1_residual_cp5_timing_20261102 as timing


def test_prefix_and_terminal_are_distinct_and_unknown_fails_closed():
    first = np.array([1., 2.])
    terminal = np.array([3., 4.])
    assert timing.timed_endpoint(first, terminal, "prefix_latent") is first
    assert timing.timed_endpoint(first, terminal, "terminal_latent") is terminal
    with pytest.raises(ValueError):
        timing.timed_endpoint(first, terminal, "goal_tangent")


def test_protocol_binds_equal_rollout_budget_and_observed_inputs():
    protocol = timing.protocol()
    assert protocol["metrics"] == ["terminal_latent", "prefix_latent"]
    assert protocol["per_plan"]["horizon_packets"] == 2
    assert protocol["per_plan"]["predict_samples"] == 3200
    assert [x["index"] for x in protocol["roles"]] == [4, 18, 19, 20]
    assert protocol["P1_admission"] is False
