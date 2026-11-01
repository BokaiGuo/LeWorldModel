import numpy as np
import pytest

from experiments.spectral_reachability import p1_cp5_cem_baseline_20261101_leibniz as cem


def test_cem_bank_causal_stable_and_exact_budget():
    events, seen = [], []
    def score(raw):
        assert raw.shape == (3, 5, 2) and np.isfinite(raw).all() and (np.abs(raw) <= 1).all()
        seen.append(raw.copy())
        return float(np.square(raw - .2).mean())
    result = cem.optimize(score, lambda k, v: events.append((k, v)))
    assert len(seen) == 65
    assert [k for k, _ in events] == ["bank", "update"] * 4 + ["selected_before_replay"]
    assert result["selected_index"] == min(range(64), key=lambda i: (result["scores"][i], i))
    np.testing.assert_array_equal(seen[-1], seen[result["selected_index"]])
    assert cem.costs()["predict_samples"] == 195
    assert cem.costs()["backward_calls"] == 0
    second = cem.optimize(lambda raw: float(np.square(raw - .2).mean()), lambda *a: None)
    assert result == second


def test_nonfinite_and_replay_mismatch_fail_closed():
    with pytest.raises(ValueError, match="nonfinite"):
        cem.optimize(lambda raw: float("nan"), lambda *a: None)
    counter = iter([0.] * 64 + [1.])
    with pytest.raises(ValueError, match="replay mismatch"):
        cem.optimize(lambda raw: next(counter), lambda *a: None)


def test_stable_tie_is_first():
    result = cem.optimize(lambda raw: 1., lambda *a: None)
    assert result["selected_index"] == 0
    assert all(update["elite_indices"] == [0, 1, 2, 3] for update in result["updates"])
