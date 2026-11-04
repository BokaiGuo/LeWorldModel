import numpy as np

from experiments.spectral_reachability import p1_cp10_coherent_replan_20261102 as subject


def test_protocol_is_development_only_and_equal_recorded_calls():
    p = subject.protocol()
    assert len(p["rows"]) == 8
    assert p["per_arm"]["predict_samples"] == 3200
    assert not p["P1_admission"] and not p["independent_confirmation"]


def test_generic_seed_planner_preserves_budget_and_bounds():
    calls = []
    def score(variants):
        calls.append(variants.shape)
        return np.square(variants).sum(axis=(2, 3, 4))
    result = subject.revision.plan([1., 0.], score, structured=True,
                                   common=subject.base.first.engine.base.common,
                                   seed=subject.base.first.planner.parent.SEED)
    assert calls == [(64, 5, 2, 5, 2)] * 5
    assert result["scored_candidates"] == 320
    assert np.abs(result["selected"]["actions"]).max() <= 1
