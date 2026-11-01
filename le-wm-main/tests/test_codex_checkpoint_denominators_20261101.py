import copy
import hashlib
import json

import pytest

from experiments.spectral_reachability import codex_checkpoint_denominators_20261101 as m


def source():
    raw = m.SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == m.SHA
    return json.loads(raw)


def test_real_fixed_roster_and_missing_stop():
    result = m.census(source())
    assert (result["planned_slots"], result["observed_slots"]) == (18, 17)
    group = next(g for g in result["groups"] if g["checkpoint"] == 15 and g["arm"] == "joint")
    assert (group["planned"], group["observed"]) == (2, 1)
    assert group["slots"][1]["native_success_at_checkpoint"] is None
    assert group["slots"][1]["missing_reason"] == "native_stop_before_checkpoint"


def test_duplicate_and_false_missing_are_rejected():
    d = source()
    d["rows"][1] = copy.deepcopy(d["rows"][0])
    with pytest.raises(ValueError, match="roster"):
        m.census(d)
    d = source()
    d["rows"][17]["real_Q"] = 0
    with pytest.raises(ValueError, match="missing slot"):
        m.census(d)


def test_missing_success_is_not_false():
    d = source()
    d["rows"][0]["native_metrics"]["native_success"] = None
    with pytest.raises(ValueError, match="observed slot"):
        m.census(d)
