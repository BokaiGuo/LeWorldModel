import numpy as np
import pytest

from experiments.spectral_reachability import pc_recursive_response_v13 as v13
from experiments.spectral_reachability.pc_timed_program_execution import execute


def config(segments=3):
    return v13.Config(action_dim=2, segment_length=5, segments=segments,
                      nominals=(tuple((0.0, 0.0) for _ in range(5)),),
                      token_cap=1920, seed=31)


def predictor(actions):
    feature = actions.sum(axis=1)
    return feature, np.linalg.norm(feature - [1., -1.], axis=1)


@pytest.mark.parametrize("arm", ["v17", "multistart_cem"])
@pytest.mark.parametrize("segments", [1, 2, 3])
def test_real_compilers_share_meter_and_preserve_scored_incumbent(arm, segments):
    c = config(segments)
    horizon = 5 * segments
    carry = np.full((horizon, 2), .03, dtype=np.float32)
    syncs = []
    record = execute(predictor, np.array([1., -1.]), c, arm=arm, carry=carry,
                     synchronize=lambda: syncs.append(1))
    assert record["resource"]["prediction_tokens"] <= c.token_cap
    assert record["resource"]["prediction_tokens"] == sum(x["prediction_tokens"] for x in record["calls"])
    assert record["resource"]["device_synchronized"]
    assert len(syncs) >= 2 + 2 * len(record["calls"])
    assert not record["resource"]["serialization_included"]
    selected = np.asarray(record["result"]["selected"], dtype=np.float32)
    assert any(np.any(np.all(batch["actions"] == selected, axis=(1, 2)))
               for batch in record["traces"] if batch["actions"].shape[1] == horizon)


def test_rejects_unknown_arm():
    with pytest.raises(ValueError, match="unsupported planner"):
        execute(predictor, [1., -1.], config(), arm="other")
