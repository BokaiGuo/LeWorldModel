import copy
import json

import pytest

from experiments import pc_tworoom_incumbent_elite_replay_20261107_codex as audit


@pytest.fixture
def arm():
    result = json.loads(audit.run.OUTPUT.read_text())
    return result["rows"][0]["arms"]["corrected"]


def test_recorded_arm_is_valid(arm):
    audit.validate_arm(arm)


@pytest.mark.parametrize("mutation", ("action", "winner", "budget", "clock", "success", "old_tail"))
def test_tampered_trace_fails_closed(arm, mutation):
    altered = copy.deepcopy(arm)
    if mutation == "action":
        altered["trace"][0]["action"][0] += 0.001
    elif mutation == "winner":
        altered["decisions"][0]["winner"] = (altered["decisions"][0]["winner"] + 1) % 680
    elif mutation == "budget":
        altered["decisions"][0]["queries"] = 679
    elif mutation == "clock":
        altered["decisions"][0]["remaining"] -= 5
    elif mutation == "success":
        altered["ever_success"] = not altered["ever_success"]
    else:
        altered["decisions"][1]["old_tail_rank"] += 1
    with pytest.raises(ValueError):
        audit.validate_arm(altered)
