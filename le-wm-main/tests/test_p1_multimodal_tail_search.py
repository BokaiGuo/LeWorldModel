import numpy as np
import pytest

from experiments.spectral_reachability import p1_unified_cp5_feedback_cem_20261101_codex as common
from experiments.spectral_reachability.p1_multimodal_tail_search import plan


def test_fixed_budget_modes_bounds_and_determinism():
    calls = []
    def score(variants):
        calls.append(variants.shape)
        return np.square(variants-.3).mean(axis=(2, 3, 4))
    result = plan([1., 0.], score, common=common, seed=42)
    replay = plan([1., 0.], score, common=common, seed=42)
    assert result == replay
    assert calls == [(64, 5, 2, 5, 2)] * 10
    assert len(result["iterations"]) == 5
    assert all(len(row["modes"]) == 4 for row in result["iterations"])
    assert np.max(np.abs(result["selected"]["actions"])) <= 1


def test_invalid_geometry_or_scoring_fails_closed():
    with pytest.raises(ValueError):
        plan([0., 0.], lambda _: None, common=common, seed=0)
    with pytest.raises(ValueError):
        plan([1., 0.], lambda _: np.full((64, 5), np.nan), common=common, seed=0)


def test_no_feasible_candidates_do_not_trigger_unmetered_retry():
    calls = []
    def score(variants):
        calls.append(1)
        return np.zeros((64, 5))
    result = plan([1., 0.], score, common=common, seed=1)
    assert len(calls) == 5
    assert result["scored_candidates"] == 320
