import pytest

from experiments.spectral_reachability.run_reacher_pc_tangent_rgb_replay import branch_index


def test_branch_roster_must_be_complete_and_unique():
    branches = [{"prefix_index": i, "tail_index": j, "status": "complete", "completed_steps": 15}
                for i in range(7) for j in range(5)]
    assert len(branch_index({"branches": branches})) == 35
    with pytest.raises(ValueError, match="roster"):
        branch_index({"branches": branches[:-1]})
    corrupted = [dict(b) for b in branches]
    corrupted[-1] = dict(corrupted[0])
    with pytest.raises(ValueError, match="roster"):
        branch_index({"branches": corrupted})
    corrupted = [dict(b) for b in branches]
    corrupted[-1]["completed_steps"] = 14
    with pytest.raises(ValueError, match="incomplete"):
        branch_index({"branches": corrupted})
