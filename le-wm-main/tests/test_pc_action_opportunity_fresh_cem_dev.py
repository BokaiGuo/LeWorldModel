import pytest

from experiments.pc_action_opportunity_fresh_cem_dev_20261104_codex import seed_for


def test_fresh_seeds_do_not_overlap_prior_candidate_streams():
    fresh = {seed_for(root, depth) for root in range(16) for depth in (0, 1)}
    old = {2032800000 + root * 1000 + depth * 100 + index
           for root in range(16) for depth in (0, 1) for index in range(100)}
    assert len(fresh) == 32
    assert not fresh.intersection(old)
    with pytest.raises(ValueError):
        seed_for(16, 0)
