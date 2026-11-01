import copy
import hashlib
import json
import subprocess
import sys

import pytest
from experiments.spectral_reachability import codex_e1_end_to_end_ledger_interface_20261030_v1 as m


def fixture():
    phases = {}
    for p in m.PHASES:
        measures = {}
        for axis in m.AXES:
            h = hashlib.sha256((p + "/" + axis).encode()).hexdigest()
            measures[axis] = {"attempted": 1, "completed": 1, "event_set_sha256": h}
        phases[p] = {"scope": "per_arm", "receipt_sha256": "b" * 64, "measurements": measures}
    return {"schema": m.SCHEMA, "scope_sha256": "a" * 64, "arm": "existing_learned_prior",
            "task_sha256": "c" * 64, "target_sha256": "d" * 64, "phases": phases}


def test_complete_declarations_never_certify_budget():
    r = m.inspect(fixture())
    assert r["structural_status"] == "complete_declarations_only"
    assert all(r[k] is False for k in ("coverage_verified", "independent_evidence_verified",
                                       "equal_budget_certified", "ranking_authorized"))


@pytest.mark.parametrize("phase", m.PHASES)
def test_missing_phase_is_error(phase):
    d = fixture()
    del d["phases"][phase]
    with pytest.raises(ValueError):
        m.inspect(d)


@pytest.mark.parametrize("phase", ("proposal", "inherited"))
def test_unknown_proposal_or_inherited_cannot_be_zeroed(phase):
    d = fixture()
    d["phases"][phase]["measurements"]["predict_api_calls"] = None
    report = m.inspect(d)
    assert report["structural_status"] == "incomplete"
    assert phase + ":predict_api_calls:unknown" in report["blockers"]


def test_shared_total_is_not_per_arm_attribution():
    d = fixture()
    d["phases"]["inherited"]["scope"] = "shared_unallocated"
    assert "inherited:not_attributed_per_arm" in m.inspect(d)["blockers"]


def test_attempts_not_refunded_and_invalid_counts_rejected():
    d = fixture()
    count = d["phases"]["proposal"]["measurements"]["high_predictor_calls"]
    count.update(attempted=3, completed=1)
    assert m.inspect(d)["structural_status"] == "complete_declarations_only"
    count["attempted"] = 0
    with pytest.raises(ValueError):
        m.inspect(d)
    count["attempted"] = True
    with pytest.raises(ValueError):
        m.inspect(d)


def test_receipt_missing_and_zero_is_not_proof():
    d = fixture()
    d["phases"]["proposal"]["receipt_sha256"] = None
    d["phases"]["proposal"]["measurements"]["cpu_proposal_draws"]["attempted"] = 0
    d["phases"]["proposal"]["measurements"]["cpu_proposal_draws"]["completed"] = 0
    r = m.inspect(d)
    assert "proposal:receipt_missing" in r["blockers"]
    assert not r["independent_evidence_verified"]


def test_duplicate_event_sets_rejected():
    d = fixture()
    d["phases"]["inherited"]["measurements"]["predict_api_calls"]["event_set_sha256"] = (
        d["phases"]["proposal"]["measurements"]["predict_api_calls"]["event_set_sha256"])
    with pytest.raises(ValueError):
        m.inspect(d)


def test_missing_axis_and_extra_axis_rejected():
    d = fixture()
    del d["phases"]["inherited"]["measurements"]["high_input_rows"]
    with pytest.raises(ValueError):
        m.inspect(d)
    d = fixture()
    d["phases"]["proposal"]["measurements"]["invented"] = None
    with pytest.raises(ValueError):
        m.inspect(d)


def test_wrong_types_and_unknown_arm():
    d = fixture()
    d["arm"] = "unknown"
    with pytest.raises(ValueError):
        m.inspect(d)
    d = fixture()
    d["phases"]["replay"]["measurements"]["flops"] = {"attempted": -1, "completed": 0,
                                                        "event_set_sha256": "e" * 64}
    with pytest.raises(ValueError):
        m.inspect(d)


def test_cli_incomplete_and_hash_mismatch():
    d = fixture()
    d["phases"]["inherited"]["measurements"]["high_predictor_calls"] = None
    raw = json.dumps(d).encode()
    cmd = [sys.executable, "-B", "-m", m.__name__]
    result = subprocess.run(cmd + [hashlib.sha256(raw).hexdigest()], input=raw, capture_output=True)
    assert result.returncode == 3
    assert json.loads(result.stdout)["equal_budget_certified"] is False
    result = subprocess.run(cmd + ["0" * 64], input=raw, capture_output=True)
    assert result.returncode == 2


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b"x" * (m.LIMIT + 1)])
def test_parser_rejects(raw):
    with pytest.raises(ValueError):
        m.parse(raw)
