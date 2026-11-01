import copy
import json

import pytest

from experiments.spectral_reachability import p3_raw_h1_prefix_pair_audit_20261031 as audit


def test_pair_accepts_sealed_real_checkpoints_and_rejects_drift():
    path = audit.previous.continuation.PREVIOUS / "batch_00/report.json"
    rows = json.loads(path.read_bytes())["rows"]
    old, new = rows[:2]
    result = audit.compare_pair(old, new)
    assert [check["step"] for check in result["checks"]] == [0, 5]
    bad = copy.deepcopy(new)
    audit.checkpoint(bad, 5)["snapshot"]["state"][0] += 1
    with pytest.raises(Exception, match="physical checkpoint mismatch"):
        audit.compare_pair(old, bad)
    bad = copy.deepcopy(new)
    bad["selected_plan"]["commands"][0][0] += 0.1
    with pytest.raises(Exception, match="pair identity"):
        audit.compare_pair(old, bad)
