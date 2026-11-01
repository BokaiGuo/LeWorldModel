import copy
import json

import numpy as np
import pytest

from experiments.spectral_reachability import p1_real_vs_imagined_midpoint_20261031 as subject
from experiments.spectral_reachability import p1_uphill_packet_audit_20261031_main as old


def test_independent_pairing_matches_existing_saved_result():
    new = json.loads((subject.ROOT / "results/p1_real_vs_imagined_midpoint_20261031_RUN01.json").read_text())
    previous = subject.prior.open_run.paired.read(old.SOURCE_COMPARE, old.SOURCE_COMPARE_SHA)
    assert len(new["rows"]) == len(previous["rows"]) == 24
    for a, b in zip(new["rows"], previous["rows"], strict=True):
        assert a["index"] == b["index"]
        assert a["open_endpoint"] == b["imagined_endpoint"]
        assert a["feedback_endpoint"] == b["real_endpoint"]
        assert a["open_status"] == b["imagined_status"]
        assert a["feedback_status"] == b["real_status"]
        assert a["feedback_minus_open"] is None if b["imagined_minus_real"] is None else all(
            abs(a["feedback_minus_open"][k] + value) < 1e-10
            for k, value in b["imagined_minus_real"].items()
        )


def test_changed_executed_action_is_rejected():
    raw = np.zeros((10, 2), dtype="<f4")
    plan = {"index": 0, "role": "zero", "logical_pair": 0,
            "first_packet_hex": raw[:5].tobytes().hex(), "second_packet_hex": raw[5:].tobytes().hex(),
            "midpoint_pixel_sha256": "a"}
    row = {"index": 0, "role": "zero", "logical_pair": 0,
           "open_returned_suffix": 10, "feedback_returned_suffix": 10}
    events = [{"step": i+1, "actual_bits": raw[i].tobytes().hex()} for i in range(10)]
    subject.validate_packets(0, row, plan, plan, events, events)
    changed = copy.deepcopy(events)
    changed[-1]["actual_bits"] = np.ones(2, dtype="<f4").tobytes().hex()
    with pytest.raises(Exception):
        subject.validate_packets(0, row, plan, plan, events, changed)
