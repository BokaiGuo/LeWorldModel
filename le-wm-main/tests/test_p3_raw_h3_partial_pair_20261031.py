import copy

import pytest

from experiments.spectral_reachability import p3_raw_h3_partial_pair_20261031 as partial
from test_p3_raw_h3_physical_saved_20261031 import completed


def test_complete_row_rejects_action_and_same_time_frame_tampering(completed):
    p, report, events, refs, image = completed
    row, plan = report["rows"][0], p["plans"][0]
    partial.validate_complete(row, plan, refs[plan["root_slot"]], events, lambda *_: image)
    changed = copy.deepcopy(events)
    next(e for e in changed if e["kind"] == "returned_transition")["actual_bits"] = "00"
    with pytest.raises(ValueError, match="transition binding"):
        partial.validate_complete(row, plan, refs[plan["root_slot"]], changed, lambda *_: image)
    changed = copy.deepcopy(events)
    plan_frame = next(e for e in changed if e["kind"] == "fresh_observation"
                      and e["branch"] == 0 and e["receipt"]["phase"] == "plan"
                      and e["receipt"]["step"] == 0)
    plan_frame["receipt"]["pixel_sha256"] = "00"
    with pytest.raises(ValueError, match="frame identity"):
        partial.validate_complete(row, plan, refs[plan["root_slot"]], changed, lambda *_: image)
