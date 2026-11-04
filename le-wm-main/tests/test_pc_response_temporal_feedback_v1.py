import numpy as np
import pytest

from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_response_temporal_feedback_v1 as feedback
from experiments.spectral_reachability import pc_response_temporal_search_v1 as original


def predictor(meter, trace):
    def predict(actions):
        actions = np.asarray(actions, dtype=float)

        def raw(batch):
            pos = batch.mean(axis=1)
            score = np.linalg.norm(pos - [0.3, -0.2], axis=1)
            trace.append((batch.copy(), score.copy()))
            return pos, score
        return meter.predict(actions, raw)
    return predict


@pytest.mark.parametrize("arm", original.ARMS)
def test_fixed_terminal_feedback_preserves_first_compiler_and_carry(arm):
    first_meter, original_meter = Meter(original.CAP), Meter(original.CAP)
    first_trace, original_trace = [], []
    first = feedback.decide(arm, 0, 0, [0.3, -0.2], predictor(first_meter, first_trace), first_meter)
    old = original.decide(arm, 0, [0.3, -0.2], predictor(original_meter, original_trace), original_meter)
    assert first["selected"] == old["selected"]
    assert len(first_trace) == len(original_trace)
    for (actions, scores), (old_actions, old_scores) in zip(first_trace, original_trace, strict=True):
        np.testing.assert_array_equal(actions, old_actions)
        np.testing.assert_array_equal(scores, old_scores)
    carry = first["selected"]["chain"][5:]
    for step in (1, 2):
        meter, trace = Meter(original.CAP), []
        result = feedback.decide(arm, 0, step, [0.3, -0.2], predictor(meter, trace), meter, carry)
        assert result["carry"] == carry
        np.testing.assert_array_equal(trace[0][0][0], carry)
        assert result["carry_score"] == trace[0][1][0]
        chain = np.asarray(result["selected"]["chain"])
        assert chain.shape == (15 - 5 * step, 2)
        scores = [score for actions, values in trace if actions.shape[1] == len(chain) for score in values]
        assert result["selected"]["terminal_score"] == min(scores)
        assert meter.token_samples <= original.CAP
        carry = chain[5:].tolist()


def test_feedback_rejects_missing_or_invalid_carry():
    meter = Meter(original.CAP)
    with pytest.raises(ValueError, match="invalid carry"):
        feedback.decide("response", 0, 1, [0., 0.], predictor(meter, []), meter, np.zeros((5, 2)))
