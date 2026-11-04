import copy
import json

import pytest

from experiments.spectral_reachability import pc_ap_reference_replay_consumer_20261102_codexexclusive as replay


@pytest.fixture
def slot():
    if not replay.base.INPUT.exists():
        pytest.skip("saved AP run unavailable")
    report = json.loads(replay.base.REFERENCE.read_text())
    row = next(r for r in report["rows"] if r["seed"] == 20263160 and r["arm"] == "ap_observed_target")
    events, _ = replay.base.read_chain(replay.base.INPUT / "20263160_ap_observed_target.jsonl")
    return row, events


def test_saved_slot_matches(slot):
    row, events = slot
    assert replay.compare_slot(row, events)["physical_steps"] == 10


@pytest.mark.parametrize("field", ["seed", "action", "distance", "rgb", "stopped"])
def test_reference_discrepancies_rejected(slot, field):
    original, events = slot
    row = copy.deepcopy(original)
    if field == "seed":
        row["decisions"][0]["search_seed"] += 1
    elif field == "action":
        row["decisions"][0]["raw_actions"][0][0] += 1
    elif field == "distance":
        row["steps"][0]["distance"] += 1
    elif field == "rgb":
        row["decisions"][0]["successor_rgb_sha256"] = "0" * 64
    else:
        row["steps"][0]["terminated"] = True
    with pytest.raises(ValueError):
        replay.compare_slot(row, events)
