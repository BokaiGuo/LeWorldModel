import copy
import hashlib

import pytest

from experiments.spectral_reachability import e3_cell03_end_to_end_replay_join_20261031_main as e


def fixture():
    entries, physical, planner = [], [], []
    for b in (10, 15):
        action = bytes([b]) * 40
        entries.append({"boundary": b, "original_action_bits": action.hex(), "body_sha256": "body",
                        "original_row_sha256": "original"})
        physical.append({"boundary": b, "pixel_sha256": "pixel",
                         "next_action_sha256": hashlib.sha256(action).hexdigest(),
                         "physical_frames_match": True, "encoder_recompute_bitwise": True})
        planner.append({"boundary": b, "action_exact": True, "result_exact": True,
                        "row_sha256": "node", "axes": {}})
    return ({"rows": physical, "input_sha256": {}}, {"rows": planner},
            {"original_report_sha256": e.replay.source.REPORT_SHA,
             "original_completion_sha256": e.replay.source.COMPLETION_SHA,
             "authority_sha256": e.replay.source.AUTHORITY_SHA, "rows": entries}, copy.deepcopy(entries))


def test_join_and_reject_wrong_action():
    data = fixture()
    result = e.join(*data)
    assert len(result["rows"]) == 2
    assert not result["scientific_admission"]
    bad = copy.deepcopy(data)
    bad[0]["rows"][0]["next_action_sha256"] = "wrong"
    with pytest.raises(ValueError, match="replay action"):
        e.join(*bad)


def test_reject_reconstructed_body_drift():
    data = fixture()
    data[3][0]["body_sha256"] = "wrong"
    with pytest.raises(ValueError, match="source drift"):
        e.join(*data)
