import copy

import pytest

from experiments.spectral_reachability import p1_response_prefix_pilot_20261031 as pilot


@pytest.fixture
def saved():
    read = pilot.replay.source.panel.source.saved.read_verified
    joined = read(pilot.uphill.JOIN, "2f8c75e502794248c399385e01db5cc991fb69b1eab7f34e4676becd1fe04e81")
    return joined, read(pilot.replay.OUTCOME, pilot.replay.OUTCOME_SHA), read(pilot.REPLAY, pilot.REPLAY_SHA)


def test_fixed_roster_and_zero_tail(saved):
    joined, model, confirmation = saved
    result = pilot.assemble(copy.deepcopy(joined), model, confirmation)
    assert len(result["rows"]) == 16
    for logical, arm in enumerate(model["rows"]):
        for offset in range(2):
            row = result["rows"][2 * logical + offset]
            index = arm["candidate_sets"]["diverse"][offset]
            assert row["candidate_index"] == index
            assert row["raw15_float32_hex"] == arm["candidates"][str(index)]["prefix_float32_hex"] + "00" * 80
    assert result["P1_admission"] is False


def test_rejects_missing_cuda_replay_candidate(saved):
    joined, model, confirmation = saved
    confirmation = copy.deepcopy(confirmation)
    confirmation["rows"][0]["candidate"] = -1
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        pilot.assemble(copy.deepcopy(joined), model, confirmation)


def test_rejects_bad_action(saved):
    joined, model, confirmation = saved
    model = copy.deepcopy(model)
    index = model["rows"][0]["candidate_sets"]["diverse"][0]
    model["rows"][0]["candidates"][str(index)]["prefix_float32_hex"] = "00"
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        pilot.assemble(copy.deepcopy(joined), model, confirmation)
