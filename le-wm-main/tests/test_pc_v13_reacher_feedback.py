import numpy as np
import pytest

from experiments import pc_v13_reacher_rgb_feedback_dev_20261104_codex as run
from experiments.pc_equal_resource_dev_20261103_codex import Meter


@pytest.mark.parametrize("step", [0, 1, 2])
def test_matched_bank_and_legal_complete_chain(step):
    horizon = 15 - 5 * step
    carry = None if step == 0 else np.full((horizon, 2), .23, np.float32)
    seen = {}
    for arm in run.ARMS:
        meter = Meter(1920)
        batches = []

        def predict(actions):
            raw = np.asarray(actions, np.float32)

            def evaluate(batch):
                features = batch.mean(axis=1)
                return features, ((features - .31) ** 2).sum(axis=1)

            pos, scores = meter.predict(raw, evaluate)
            batches.append(raw.copy())
            return pos, scores

        result = run.decide(arm, step, [.31, .31], predict, meter, carry)
        chain = np.asarray(result["selected"]["chain"], np.float32)
        assert chain.shape == (horizon, 2)
        assert np.max(np.abs(chain)) <= 1
        assert meter.token_samples <= 1920
        assert np.array_equal(batches[-1][1, :5], np.zeros((5, 2)))
        assert np.array_equal(chain, batches[-1][0]) or np.array_equal(chain, batches[-1][1])
        if carry is not None:
            np.testing.assert_array_equal(batches[0][1], carry)
        seen[arm] = batches[0]
    np.testing.assert_array_equal(seen["response"], seen["global_cem"])
