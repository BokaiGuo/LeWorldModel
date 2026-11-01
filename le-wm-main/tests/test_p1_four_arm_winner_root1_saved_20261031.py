import pytest

from experiments.spectral_reachability import p1_four_arm_winner_root1_saved_20261031 as saved


DIRECTORY = saved.ROOT / "results/p1_four_arm_winner_root1_20261031_RUN01"
REPORT = "5e58003df2d3d70eb87843789b2bb3b0994750ce5dcad2ee4825325a64f14838"


def test_saved_actual_result_and_hash_guard():
    if not DIRECTORY.exists():
        pytest.skip("fixed local physical receipt unavailable")
    assert saved.check(DIRECTORY, REPORT)["nodes"] == 16
    with pytest.raises(ValueError, match="report hash"):
        saved.check(DIRECTORY, "0" * 64)
