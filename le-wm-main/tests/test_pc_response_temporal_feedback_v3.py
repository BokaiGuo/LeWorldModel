import numpy as np
import pytest

from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_response_temporal_feedback_v3 as method
from experiments.spectral_reachability import pc_response_temporal_search_v1 as core


@pytest.mark.parametrize("step", [0, 1, 2])
@pytest.mark.parametrize("arm", ["response", "matched_random"])
def test_refinement_starts_from_best_previously_scored_chain(monkeypatch, step, arm):
    horizon = 15 - 5 * step
    meter, full = Meter(core.CAP), []
    snapshots = []
    original_cem = core.cem

    def inspect_cem(base, prefix, count, rng, predict, initial_score, knots):
        if len(prefix) == 0:
            snapshots.append((np.asarray(base).copy(), initial_score,
                              [(chain.copy(), score) for chain, score in full]))
        return original_cem(base, prefix, count, rng, predict, initial_score, knots)

    monkeypatch.setattr(core, "cem", inspect_cem)

    def predict(actions):
        def evaluate(batch):
            position = batch.mean(axis=1)
            # Smooth nontrivial objective makes ordering meaningful.
            scores = ((batch[:, :, 0] - .31) ** 2 + (batch[:, :, 1] + .12) ** 2).mean(axis=1)
            if batch.shape[1] == horizon:
                full.extend((chain.copy(), float(score)) for chain, score in zip(batch, scores, strict=True))
            return position, scores
        return meter.predict(np.asarray(actions), evaluate)

    carry = None if step == 0 else np.zeros((horizon, 2))
    result = method.decide(arm, 3, step, [0., 0.], predict, meter, carry)
    assert len(snapshots) == 1
    start, start_score, prior = snapshots[0]
    assert start_score == min(score for _, score in prior)
    assert any(np.array_equal(start, chain) and start_score == score for chain, score in prior)
    np.testing.assert_array_equal(start, result["refinement_start"]["chain"])
    assert result["selected"]["terminal_score"] == min(score for _, score in full)
    assert meter.token_samples <= core.CAP


@pytest.mark.parametrize("step", [1, 2])
def test_pre_refinement_preserves_needle_probe(step):
    horizon = 15 - 5 * step
    needle = np.zeros((horizon, 2))
    needle[-5:] = core.NOMINALS[0]
    needle[-5, 0] += .05
    meter = Meter(core.CAP)

    def predict(actions):
        def evaluate(batch):
            scores = np.zeros(len(batch))
            if batch.shape[1] == horizon:
                scores[np.all(np.isclose(batch, needle, rtol=0, atol=1e-12), axis=(1, 2))] = -1.
            return batch.mean(axis=1), scores
        return meter.predict(np.asarray(actions), evaluate)

    result = method.decide("response", 0, step, [0., 0.], predict, meter, np.zeros((horizon, 2)))
    assert result["refinement_start"]["terminal_score"] == -1.
    np.testing.assert_array_equal(result["refinement_start"]["chain"], needle)
    assert result["selected"]["terminal_score"] == -1.
    assert meter.token_samples <= core.CAP
