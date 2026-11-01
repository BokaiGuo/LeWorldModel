import copy
import json
import subprocess
import sys

import pytest

from experiments.spectral_reachability import p1_reset_ledger_intake_20261101_exclusive as intake


def fixture():
    return {"schema": intake.SCHEMA, "calls": [
        {"call_id": c, "ledger_sha256": None, "coverage": "unknown", "attempts": []}
        for c in ("A:1", "B:1", "B:2", "B:3", "C:1", "D:1", "E:1", "F:1", "G:1", "H:1")
    ]}


def test_unknown_remains_blocked():
    result = intake.inspect(fixture())
    assert len(result["missing_or_unknown"]) == 10
    assert result["training_disjoint"] == "unknown"
    assert result["P1_admission"] is False


def test_collisions_keep_all_records_and_no_certification():
    data = fixture()
    for index in (0, 1):
        data["calls"][index].update(ledger_sha256="a" * 64, coverage="declared_complete",
            attempts=[{"attempt_id": "x", "record_pointer": "/attempts/0", "state_sha256": "b" * 64}])
    result = intake.inspect(data)
    assert len(result["exact_state_digest_collisions"][0]["records"]) == 2
    assert not result["historical_coverage_certified"]


def test_complete_declaration_is_not_authority():
    data = fixture()
    for call in data["calls"]:
        call.update(coverage="declared_complete", ledger_sha256="a" * 64)
    result = intake.inspect(data)
    assert result["missing_or_unknown"] == []
    assert not result["freshness_certified"] and not result["source_run_authorized"]


def test_missing_state_and_duplicate_call():
    data = fixture()
    data["calls"][0].update(ledger_sha256="a" * 64, attempts=[
        {"attempt_id": "a", "record_pointer": "/0", "state_sha256": None}])
    assert "A:1/a:state" in intake.inspect(data)["missing_or_unknown"]
    data["calls"][1] = copy.deepcopy(data["calls"][0])
    with pytest.raises(ValueError):
        intake.inspect(data)


def test_unbound_attempt_and_unknown_fields_rejected():
    data = fixture()
    data["calls"][0]["attempts"] = [{"attempt_id": "a", "record_pointer": "/0", "state_sha256": None}]
    with pytest.raises(ValueError):
        intake.inspect(data)
    data = fixture()
    data["calls"][0]["outcome"] = 1
    with pytest.raises(ValueError):
        intake.inspect(data)


def test_cli_fail_closed_and_duplicate_keys():
    script = intake.__file__
    run = subprocess.run([sys.executable, "-B", script], input=json.dumps(fixture()),
                         text=True, capture_output=True, check=False)
    assert run.returncode == 2
    assert json.loads(run.stdout)["P1_admission"] is False
    run = subprocess.run([sys.executable, "-B", script], input='{"schema":1,"schema":2}',
                         text=True, capture_output=True, check=False)
    assert run.returncode == 2 and "duplicate" in run.stderr
