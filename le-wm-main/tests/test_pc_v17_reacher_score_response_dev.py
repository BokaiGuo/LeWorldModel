import numpy as np
import pytest

from experiments import pc_v17_reacher_score_response_dev_20261104_codex as runner
from experiments.pc_equal_resource_dev_20261103_codex import Meter


@pytest.mark.parametrize("step", [0, 1, 2])
def test_paired_arms_share_bank_and_account_for_carry(step):
    horizon = 15 - 5 * step
    carry = None if step == 0 else np.full((horizon, 2), .2, np.float32)
    banks = []
    for arm in runner.ARMS:
        meter, trace = Meter(runner.CAP), []

        def predict(actions):
            raw = np.asarray(actions, np.float32)
            trace.append(raw.copy())
            def evaluate(batch):
                position = batch.mean(axis=1)
                return position, ((position - .3) ** 2).sum(axis=1)
            return meter.predict(raw, evaluate)

        decision = runner.decide(arm, step, np.full(2, .3), predict, meter, carry)
        selected = np.asarray(decision["selected"])
        assert selected.shape == (horizon, 2)
        assert meter.token_samples <= runner.CAP
        assert meter.token_samples == sum(len(batch) * batch.shape[1] // 5 for batch in trace)
        if carry is not None:
            np.testing.assert_array_equal(trace[0][1], carry)
        banks.append(trace[0])
    np.testing.assert_array_equal(banks[0], banks[1])
    np.testing.assert_array_equal(banks[0], banks[2])
