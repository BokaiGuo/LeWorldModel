import numpy as np
import pytest

from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_response_temporal_feedback_v1 as old
from experiments.spectral_reachability import pc_response_temporal_feedback_v2 as fixed
from experiments.spectral_reachability import pc_response_temporal_search_v1 as initial


def score_one_probe(step):
    horizon = 15 - 5 * step
    needle = np.zeros((horizon, 2))
    needle[-5:] = initial.NOMINALS[0]
    needle[-5, 0] += .05
    meter = Meter(initial.CAP)
    scored = []

    def predict(actions):
        def raw(batch):
            pos = batch.mean(axis=1)
            scores = np.zeros(len(batch))
            if batch.shape[1] == horizon:
                hit = np.all(np.isclose(batch, needle, rtol=0, atol=1e-12), axis=(1, 2))
                scores[hit] = -1.
                scored.extend(scores.tolist())
            return pos, scores
        return meter.predict(np.asarray(actions, dtype=float), raw)
    return needle, meter, scored, predict


def test_v1_counterexample_scored_probe_was_ignored():
    _, meter, scored, predict = score_one_probe(1)
    result = old.decide("response", 0, 1, [0., 0.], predict, meter, np.zeros((10, 2)))
    assert min(scored) == -1.
    assert result["selected"]["terminal_score"] == 0.


@pytest.mark.parametrize("step", [1, 2])
@pytest.mark.parametrize("arm", ["response", "matched_random"])
def test_v2_keeps_best_scored_probe_and_rechecks_abstention(step, arm):
    needle, meter, scored, predict = score_one_probe(step)
    result = fixed.decide(arm, 0, step, [0., 0.], predict, meter,
                          np.zeros((15 - 5 * step, 2)))
    assert min(scored) == -1.
    assert result["selected"]["terminal_score"] == min(scored)
    np.testing.assert_array_equal(result["selected"]["chain"], needle)
    # The ten-step probe already has a zero first packet, so its
    # zero-prefix comparator is identical and abstention is appropriate.
    assert result["abstained"] is (step == 1)
    assert result["comparison_score"] == (-1. if step == 1 else 0.)
    assert meter.token_samples <= initial.CAP


def test_v2_zero_prefix_comparator_can_abstain_after_global_check():
    meter = Meter(initial.CAP)

    def predict(actions):
        def raw(batch):
            return batch.mean(axis=1), np.zeros(len(batch))
        return meter.predict(np.asarray(actions, dtype=float), raw)

    result = fixed.decide("response", 0, 1, [0., 0.], predict, meter, np.zeros((10, 2)))
    assert result["abstained"]
    assert result["selected"]["name"] == "zero_prefix"
    assert meter.token_samples <= initial.CAP
