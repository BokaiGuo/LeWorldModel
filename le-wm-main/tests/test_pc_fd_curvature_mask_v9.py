import numpy as np
import pytest

from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_fd_curvature_mask_v9 as v9
from experiments.spectral_reachability import pc_target_reach_paired_abstention_v7 as v7


def test_masks_nonsmooth_column_without_privileged_inputs():
    structure = {"svd": 0, "fd_action_samples": 0, "svd_seconds": 0.}
    def predict(chains):
        a = np.asarray(chains)
        p = np.stack((np.maximum(a[:, 0, 0], 0), a[:, 1, 1]), axis=1)
        return p, np.linalg.norm(p, axis=1)
    _, result = v9.v6.response(np.zeros((5, 2)), [], np.array([1., 1.]), predict, structure)
    assert result["masked_coordinates"] == [0]
    assert result["raw_matrix"][0][0] > 0
    assert result["matrix"][0][0] == 0
    assert structure["masked_coordinates"] == 1


def test_zero_rank_keeps_zero_candidate_and_completes():
    meter = Meter(1920)
    def predict(chains):
        raw = np.asarray(chains, np.float32)
        return meter.predict(raw, lambda a: (np.zeros((len(a), 2)), np.ones(len(a))))
    result = v9.decide("response", "response", 0, 0, [1., 1.], predict, meter, np.zeros((2, 5, 2)))
    assert [item["name"] for item in result["first_bank"]] == ["zero"]
    assert result["abstained"]
    assert meter.token_samples <= 1920


@pytest.mark.parametrize("step", [0, 1, 2])
def test_linear_predictor_replays_v7(step):
    horizon, nominal = 15-5*step, np.zeros((2, 5, 2))
    carry = None if step == 0 else np.full((horizon, 2), .1)
    def run(method):
        meter, trace = Meter(1920), []
        def predict(chains):
            raw = np.asarray(chains, np.float32)
            def evaluate(a):
                pos = np.asarray(a, np.float64).sum(axis=1)
                return pos, np.linalg.norm(pos-[1, .5], axis=1)
            position, score = meter.predict(raw, evaluate)
            trace.append(raw)
            return position, score
        result = method.decide("response", "response", 0, step, [1, .5], predict, meter, nominal, carry)
        return result, trace, meter.token_samples
    expected, a, tokens_a = run(v7)
    actual, b, tokens_b = run(v9)
    assert tokens_a == tokens_b <= 1920
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(actual["selected"]["chain"], expected["selected"]["chain"])
    assert actual["selected"]["terminal_score"] == expected["selected"]["terminal_score"]
    assert actual["structure"].get("masked_coordinates", 0) == 0
