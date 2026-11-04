import copy
import json

import pytest

from experiments.spectral_reachability import reacher_cem32_same_chain_rgb_20261102 as rgb
from experiments.spectral_reachability import reacher_cem32_same_chain_rgb_saved_20261102 as saved


@pytest.fixture(scope="module")
def source():
    plan_path = rgb.ROOT / "results/reacher_cem32_same_chain_rgb_PLAN_20261102_v1.json"
    journal = rgb.ROOT / "results/reacher_cem32_same_chain_rgb_RUN_20261102_v1/events.jsonl"
    if not plan_path.exists() or not journal.exists():
        pytest.skip("development run not present")
    return (json.loads(plan_path.read_text()), rgb.exp.run.saved.verify_journal(journal),
            json.loads(rgb.PLAN.read_text()), rgb.exp.run.saved.verify_journal(rgb.JOURNAL),
            rgb.setup._sha256_file(plan_path))


def test_saved_data_validate(source):
    assert len(saved.validate(*source)["rows"]) == 6


@pytest.mark.parametrize("field", ["frame", "history", "coordinate", "order"])
def test_tampering_rejected(source, field):
    plan, events, physical_plan, physical_events, digest = source
    events = copy.deepcopy(events)
    if field == "frame":
        events[1]["frame_hashes"][0] = "tampered"
    elif field == "history":
        events[1]["raw_history"][0][0] += 1
    elif field == "coordinate":
        events[1]["actual_coordinate"][0] += 10
    else:
        events[1], events[2] = events[2], events[1]
    with pytest.raises(Exception):
        saved.validate(plan, events, physical_plan, physical_events, digest)
