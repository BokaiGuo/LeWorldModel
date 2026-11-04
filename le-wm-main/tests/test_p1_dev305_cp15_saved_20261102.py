"""Check multi-feedback development evidence and explicit cost boundary."""
from experiments.spectral_reachability import p1_dev305_cp15_saved_20261102 as saved


def test_four_paths_replay_and_execute_third_packet():
    result = saved.analyze()
    assert len(result["rows"]) == 4
    assert result["frames_verified"] == 24
    assert all(not row["native_success"] for row in result["rows"])
    assert not result["independent_confirmation"]
    assert not result["full_cost_certified"]
