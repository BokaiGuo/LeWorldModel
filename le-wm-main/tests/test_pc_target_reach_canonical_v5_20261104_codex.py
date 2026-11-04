import numpy as np
import pytest

from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_target_reach_canonical_v5 as v5


def make_predict(meter):
    def predict(actions):
        raw = np.asarray(actions, dtype=np.float32)

        def evaluate(batch):
            position = batch.sum(axis=1) / 10
            score = np.linalg.norm(position - np.array([0.2, -0.1]), axis=1)
            return position, score

        return meter.predict(raw, evaluate)

    return predict


@pytest.mark.parametrize("step", [0, 1, 2])
def test_valid_terminal_horizons_and_bounds(step):
    meter = Meter(1920)
    carry = None if step == 0 else np.zeros((15 - 5 * step, 2))
    result = v5.decide("response", "response", 2, step, [0.2, -0.1], make_predict(meter), meter, carry)
    assert np.asarray(result["selected"]["chain"]).shape == (15 - 5 * step, 2)
    assert meter.token_samples <= 1920
    assert result["retained"] == sorted(result["retained"])
    for program in result["programs"]:
        assert program["rng_seed"] == 2027110400 + 2 * 100 + step * 10000 + 100000 + program["parent_index"]
        assert sum(g["size"] for g in program["generations"]) == program["samples"]


def test_invalid_inputs_fail_closed():
    meter = Meter(1920)
    with pytest.raises(ValueError):
        v5.decide("invalid", "response", 0, 0, [0, 0], make_predict(meter), meter)
    with pytest.raises(ValueError):
        v5.decide("response", "response", 0, 1, [0, 0], make_predict(meter), meter, None)


def test_shared_candidates_keep_rng_identity():
    outcomes = []
    for strategy in v5.STRATEGIES:
        meter = Meter(1920)
        outcomes.append(v5.decide(strategy, "response", 3, 0, [0.2, -0.1], make_predict(meter), meter))
    left, right = ({p["parent_index"]: p for p in outcome["programs"]} for outcome in outcomes)
    common = left.keys() & right.keys()
    assert common
    for index in common:
        assert left[index]["rng_seed"] == right[index]["rng_seed"]
        if left[index]["samples"] == right[index]["samples"]:
            assert left[index]["chain"] == right[index]["chain"]
            assert left[index]["terminal_score"] == right[index]["terminal_score"]
