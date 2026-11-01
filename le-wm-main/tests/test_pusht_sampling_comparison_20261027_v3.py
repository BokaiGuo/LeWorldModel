"""Finalization regression: real str binding keys and real V2 branch payloads."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import run_pusht_sampling_comparison_20261027_v3 as v3

OLD = Path("results/pusht_sampling_comparison_20261027_v2_physical")


def strict_digest(path):
    # Same API requirement as the actual base.digest; strings fail here.
    assert isinstance(path, Path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def real_rows():
    if not OLD.exists():
        pytest.skip("actual V2 physical branches absent")
    return [json.loads((OLD/f"root_{i}.{m}.physical.json").read_text())
        for i in range(4) for m in v3.v2.METHODS]


def test_finalize_actual_str_path_bindings_and_eight_real_payloads(tmp_path, real_rows):
    protocol = json.loads((OLD/"protocol.json").read_text())
    bindings = protocol["plans_bindings"]
    assert bindings and all(type(path) is str for path in bindings)
    report = v3.finalize(output=tmp_path,rows=real_rows,protocol=protocol,bindings=bindings,
        digest=strict_digest,verify_sources=lambda: [{"status":"verified"}])
    saved = json.loads((tmp_path/"report.json").read_text())
    assert saved == report
    assert saved["status"] == "guarded_calibration_recorded"
    assert saved["gates"]["bound_bytes_unchanged"]
    assert saved["completed_full_horizon_branches"] == 8
    assert saved["new_predict_calls"] == saved["new_encoder_calls"] == 0


def test_sourceverify_exception_still_writes_full_failure_report(tmp_path, real_rows):
    protocol = json.loads((OLD/"protocol.json").read_text())
    def fail():
        raise ValueError("injected source SHA mismatch")
    report = v3.finalize(output=tmp_path,rows=real_rows,protocol=protocol,
        bindings=protocol["plans_bindings"],digest=strict_digest,verify_sources=fail)
    assert report["status"] == "integrity_fail"
    assert not report["gates"]["sources_inputs_unchanged"]
    assert len(report["rows"]) == 8
    assert report["failures"][0]["phase"] == "source_verification"
    assert "injected source SHA" in report["failures"][0]["message"]
    assert json.loads((tmp_path/"report.json").read_text()) == report


def test_hash_exception_does_not_short_circuit_sourceverify_or_report(tmp_path):
    observed = []
    def fail(path):
        assert isinstance(path, Path)
        observed.append(str(path))
        raise FileNotFoundError(str(path))
    def verify():
        observed.append("sourceverify")
        raise RuntimeError("source failure too")
    report = v3.finalize(output=tmp_path, rows=[], protocol={},
        bindings={"missing_a":"a", "missing_b":"b"},digest=fail,verify_sources=verify)
    assert observed == ["missing_a","missing_b","sourceverify"]
    assert len(report["failures"]) == 3
    assert report["status"] == "integrity_fail"
    assert (tmp_path/"report.json").exists()


def test_wrong_hash_fails_without_exception_and_keeps_report(tmp_path):
    report = v3.finalize(output=tmp_path,rows=[],protocol={},bindings={"x":"expected"},
        digest=lambda path:"different",verify_sources=lambda:[])
    assert not report["gates"]["bound_bytes_unchanged"]
    assert report["status"] == "integrity_fail"
    assert report["binding_verification"][0]["actual"] == "different"


def test_exact_replay_checks_entire_trace_and_branch(real_rows):
    row = real_rows[0]
    result = v3.comparison(row,copy.deepcopy(row))
    assert result["full_trace_exact"] and result["whole_branch_exact"]
    old = copy.deepcopy(row)
    old["trace"][0]["state"][0] += 1e-12
    result = v3.comparison(row,old)
    assert not result["full_trace_exact"] and not result["whole_branch_exact"]
    old = copy.deepcopy(row)
    old["official"]["official_success"] = not old["official"]["official_success"]
    result = v3.comparison(row,old)
    assert result["full_trace_exact"] and not result["whole_branch_exact"]


def test_duplicate_branch_matrix_not_accepted(tmp_path,real_rows):
    rows = real_rows[:7]+[real_rows[0]]
    report = v3.finalize(output=tmp_path,rows=rows,protocol={},bindings={"x":"ok"},
        digest=lambda path:"ok",verify_sources=lambda:[])
    assert not report["gates"]["all_eight_retained"]
    assert report["status"] == "integrity_fail"


def test_comparison_mismatch_cannot_pass_finalize(tmp_path,real_rows):
    comparisons = [v3.comparison(r,r) for r in real_rows]
    comparisons[-1]["full_trace_exact"] = False
    report = v3.finalize(output=tmp_path,rows=real_rows,protocol={},bindings={"x":"ok"},
        digest=lambda path:"ok",verify_sources=lambda:[],comparisons=comparisons)
    assert not report["gates"]["old_v2_all_eight_exact"]
    assert report["status"] == "integrity_fail"
