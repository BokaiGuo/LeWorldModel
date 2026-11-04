import numpy as np
import pytest

from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_response_temporal_search_v1 as core
from experiments.spectral_reachability import pc_target_reach_action_dim_v6 as v6
from experiments.spectral_reachability import pc_target_reach_canonical_v5 as v5


def evaluator(trace):
    def predict(actions):
        raw = np.asarray(actions, dtype=float)
        trace.append(raw.copy())
        weighted = raw * np.linspace(.2, 1., raw.shape[1])[None, :, None]
        position = np.stack((weighted.sum(axis=(1, 2)), (weighted**2).sum(axis=(1, 2))), axis=1)
        return position, np.linalg.norm(position - np.array([.7, .4]), axis=1)
    return predict


def metered(meter, trace):
    evaluate = evaluator(trace)
    return lambda actions: meter.predict(np.asarray(actions), evaluate)


@pytest.mark.parametrize("step", [0, 1, 2])
@pytest.mark.parametrize("strategy", ["response", "position"])
def test_two_dimension_exact_v5_equivalence(step, strategy):
    old_meter, new_meter, old_trace, new_trace = Meter(1920), Meter(1920), [], []
    carry = None if step == 0 else np.tile([.1, -.2], (15-5*step, 1))
    old = v5.decide(strategy, "response", 2, step, [.7, .4], metered(old_meter, old_trace), old_meter, carry)
    new = v6.decide(strategy, "response", 2, step, [.7, .4], metered(new_meter, new_trace), new_meter,
                    np.asarray(core.NOMINALS), carry)
    for key in old:
        if key == "structure":
            assert old[key]["svd"] == new[key]["svd"]
            assert old[key]["fd_action_samples"] == new[key]["fd_action_samples"]
        else:
            assert old[key] == new[key], key
    assert old_meter.token_samples == new_meter.token_samples
    assert len(old_trace) == len(new_trace)
    for left, right in zip(old_trace, new_trace, strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("step", [0, 1, 2])
def test_five_dimensional_packet_and_all_probe_coordinates(step):
    meter, trace = Meter(1920), []
    nominals = np.stack([np.zeros((5, 5)), np.full((5, 5), .2)])
    carry = None if step == 0 else np.zeros((15-5*step, 5))
    result = v6.decide("response", "response", 4, step, [.7, .4], metered(meter, trace), meter, nominals, carry)
    assert np.asarray(result["selected"]["chain"]).shape == (15-5*step, 5)
    assert meter.token_samples <= 1920
    first_probes = trace[0 if step == 0 else 1]
    assert first_probes.shape == (51, 5, 5)
    delta = (first_probes[1:26] - first_probes[26:]).reshape(25, 25)
    np.testing.assert_allclose(delta, np.eye(25)*.1, atol=1e-15, rtol=0)
    for batch in trace:
        assert np.max(np.abs(batch)) <= 1


def test_rejects_unsafe_nominal_probe():
    meter = Meter(1920)
    with pytest.raises(ValueError):
        v6.decide("response", "response", 0, 0, [0., 0.], metered(meter, []), meter,
                  np.full((2, 5, 5), .99))
