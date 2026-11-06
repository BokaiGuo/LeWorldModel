import numpy as np
import pytest

from experiments.pc_multistart_trace_audit import audit_multistart


def example():
    bank = np.array([[[0.0]], [[0.2]], [[0.4]]], dtype=np.float32)
    batches = [(bank, np.array([3., 1., 2.])),
               (np.array([[[0.3]], [[0.5]]]), np.array([.8, 4.])),
               (np.array([[[0.6]]]), np.array([1.5])),
               (np.array([[[0.3]], [[0.]]]), np.array([.8, 3.]))]
    result = {"bank_indices": [1, 2], "bank_scores": [3., 1., 2.],
              "starts": [{"index": 1, "samples": 2, "score": .8,
                          "generations": [{"size": 2, "best_score": .8}]},
                         {"index": 2, "samples": 1, "score": 1.5,
                          "generations": [{"size": 1, "best_score": 1.5}]}],
              "pair_scores": [.8, 3.], "pair_choice": 0,
              "selected": [[.3]], "selected_score": .8}
    return result, batches


def test_trace_audit_and_wrong_start():
    result, batches = example()
    assert audit_multistart(result, batches)["samples"] == 3
    result["bank_indices"] = [0, 2]
    with pytest.raises(ValueError, match="start indices"):
        audit_multistart(result, batches)


def test_trace_audit_detects_lost_incumbent():
    result, batches = example()
    result["starts"][0]["generations"][0]["best_score"] = 4.
    with pytest.raises(AssertionError):
        audit_multistart(result, batches)
