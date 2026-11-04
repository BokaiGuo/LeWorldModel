import numpy as np
import pytest

from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.pc_rgb_feedback_tail_retention_20261103_codex import select_at_feedback


def test_sealed_chain_binding_and_shared_budget():
    chain = np.full((15, 2), .25)
    meter = Meter(96)
    calls = []

    def score(actions):
        def raw(batch):
            calls.append(batch.copy())
            return np.sum(batch ** 2, axis=(1, 2))
        return meter.predict(actions, raw)

    first, decision = select_at_feedback(chain, 0, score)
    np.testing.assert_array_equal(first, chain[:10])
    np.testing.assert_array_equal(calls[0][0], chain[10:])
    assert meter.samples == meter.token_samples == 96
    assert decision["old_tail_samples"] == 1
    assert decision["new_search_samples"] == 95
    assert decision["source"] == "new"
    np.testing.assert_array_equal(decision["tail"], decision["new_tail"])
    assert decision["new_score"] < decision["old_score"]


def test_rejects_bad_root_chain_before_scoring():
    with pytest.raises(ValueError):
        select_at_feedback(np.zeros((10, 2)), 0, lambda _: None)
