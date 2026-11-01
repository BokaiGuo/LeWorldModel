from copy import deepcopy
from pathlib import Path

import pytest

from experiments.spectral_reachability import p1_backtrack_physical_source_join_20261030 as join


@pytest.fixture(scope="module")
def sources():
    if not Path(join.REPLAY_PATH).is_file():
        pytest.skip("sealed P1 replay not present")
    read = join.replay.one.saved.read_verified
    action = read(join.replay.INPUT, join.replay.INPUT_SHA)
    report = read(join.REPLAY_PATH, join.REPLAY_SHA)
    source = read(join.replay.one.source.REPORT, join.replay.one.source.REPORT_SHA)
    prep = read(source["prepared"], source["prepare_sha256"])
    return action, report, prep, join.physical.source_material()


def test_fixed_source_roster_and_distinct_identity_schemes(sources):
    result = join.assemble(*sources)
    assert len(result["rows"]) == 16
    assert len(result["task_rows"]) == 4
    assert all(x["warmup_model_history_exact_numpy"] for x in result["task_rows"])
    assert not result["physical_authorized"] and not result["P1_admission"]
    assert result["task_rows"][0]["model_root_sha256"] != result["task_rows"][0]["native_root_sha256"]


def test_rejects_action_and_history_tampering(sources):
    action, report, prep, physical = sources
    bad = deepcopy(action)
    bad["rows"][0]["raw15"][0][0] += .1
    with pytest.raises(ValueError, match="raw action"):
        join.assemble(bad, report, prep, physical)
    bad_prep = deepcopy(prep)
    bad_prep["requests"][0]["task"]["root_context"]["action_history"][0][0] += .1
    with pytest.raises(ValueError, match="history mismatch"):
        join.assemble(action, report, bad_prep, physical)
