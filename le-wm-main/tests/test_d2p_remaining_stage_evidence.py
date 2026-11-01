"""Synthetic report-contract tests only; no neural/physical experiment evidence."""
import json

import pytest

from experiments.spectral_reachability import d2p_remaining_stage_evidence as evidence


def write_report(tmp_path, payload, relative="results/example.json"):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


SPEC = ("example.json", "fixture.v1", "completed", ("cuda_only", "frozen"))


def report():
    return {"schema": "fixture.v1", "status": "completed", "gates": {"cuda_only": True, "frozen": True}}


def sra():
    return {"inputs": {"calibration_roots": [1, 2], "outcome_roots": [3, 4]},
            "proxy_calibration": {"roots": [1, 2]},
            "rows": [{"root": root, "method": method} for method in ("a", "b") for root in (3, 4)]}


def test_missing_report_not_run(tmp_path):
    entry, payload = evidence.inspect_report(tmp_path, SPEC)
    assert entry["availability"] == "missing"
    assert entry["evidence_status"] == "not_run"
    assert payload == {}


def test_report_checks_not_admission_or_seal(tmp_path):
    path = write_report(tmp_path, report() | {"P1_admission": True})
    before = path.read_bytes()
    entry, _ = evidence.inspect_report(tmp_path, SPEC)
    assert entry["evidence_status"] == "report_checks_pass"
    assert entry["admission"] is False
    assert entry["seal_verified"] is False
    assert path.read_bytes() == before


@pytest.mark.parametrize("damage", ["schema", "status", "missing", "empty", "false", "integer", "extra", "mapping"])
def test_strict_required_and_extra_gates(tmp_path, damage):
    payload = report()
    if damage == "schema":
        payload["schema"] = "wrong"
    elif damage == "status":
        payload["status"] = "running"
    elif damage == "missing":
        del payload["gates"]["frozen"]
    elif damage == "empty":
        payload["gates"] = {}
    elif damage in ("false", "integer"):
        payload["gates"]["frozen"] = False if damage == "false" else 1
    elif damage == "extra":
        payload["gates"]["new_gate"] = False
    else:
        payload["gates"] = []
    write_report(tmp_path, payload)
    entry, _ = evidence.inspect_report(tmp_path, SPEC)
    assert entry["evidence_status"] == "blocked"


@pytest.mark.parametrize("raw", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}', '[]', '{}', 'broken'])
def test_strict_json_rejects_invalid(raw):
    with pytest.raises(ValueError):
        evidence.strict_json(raw.encode())


def test_oversize_not_read(tmp_path, monkeypatch):
    write_report(tmp_path, report())
    monkeypatch.setattr(evidence, "MAX_BYTES", 10)
    entry, payload = evidence.inspect_report(tmp_path, SPEC)
    assert entry["availability"] == "present"
    assert entry["evidence_status"] == "blocked"
    assert "limit" in entry["issues"][0]
    assert payload == {}


def test_escape_and_directory_rejected(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results/example.json").symlink_to("/etc/passwd")
    assert "escapes" in evidence.inspect_report(tmp_path, SPEC)[0]["issues"][0]
    with pytest.raises(ValueError, match="regular"):
        evidence.bounded_read(tmp_path, "results")


def test_sra_root_pairing_recomputed_not_flag():
    payload = sra()
    result = evidence.check_sra_roots(payload)
    assert result["status"] == "report_local_disjoint_and_paired"
    assert result["outcome_roots"] == 2
    assert result["source_episode_independence"] == "unknown"


@pytest.mark.parametrize("damage", ["overlap", "duplicate", "unpaired", "proxy", "empty", "badroot", "boolroot"])
def test_sra_bad_roots_block_even_with_true_flag(damage):
    payload = sra() | {"gates": {"root_disjoint_calibration_outcome": True}}
    if damage == "overlap":
        payload["inputs"]["calibration_roots"] = [1, 3]
        payload["proxy_calibration"]["roots"] = [1, 3]
    elif damage == "duplicate":
        payload["rows"].append(payload["rows"][0])
    elif damage == "unpaired":
        payload["rows"].pop()
    elif damage == "proxy":
        payload["proxy_calibration"]["roots"] = [1, 5]
    elif damage == "empty":
        payload["inputs"]["outcome_roots"] = []
    elif damage == "badroot":
        payload["rows"][0]["root"] = {}
    else:
        payload["inputs"]["outcome_roots"] = [True, 4]
    assert evidence.check_sra_roots(payload)["status"] == "blocked"


def test_known_root_reuse_not_outcome_and_no_source_claim():
    calibration = {"rows": [{"root_sha256": "a" * 64}, {"root_sha256": "b" * 64}]}
    candidate = {"rows": [{"root_sha256": "b" * 64}]}
    result = evidence.compare_saved_roots(calibration, candidate)
    assert result["status"] == "known_calibration_reuse"
    assert result["overlap"] == 1
    candidate["rows"][0]["root_sha256"] = "c" * 64
    result = evidence.compare_saved_roots(calibration, candidate)
    assert result["status"] == "rgb_hash_disjoint_only"
    assert result["independent_outcome_established"] is False
    candidate["rows"].append(candidate["rows"][0])
    assert evidence.compare_saved_roots(calibration, candidate)["status"] == "unknown"


def test_no_discovery_or_promotion_and_canonical_anchors(tmp_path, monkeypatch):
    for relative in evidence.DOCUMENTS.values():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Doc\n| P4：RGB 闭环多环境 | exact source |\n| P5 | exact replication |\n")
    for spec in evidence.REPORTS.values():
        relative, schema, status, required = spec
        write_report(tmp_path / "le-wm-main", {"schema": schema, "status": status,
            "gates": {key: True for key in required}, "P1_admission": True,
            "P4_admission": True, "P5_admission": True}, "results/" + relative)
    def no_walk(*args, **kwargs):
        raise AssertionError("directory traversal forbidden")
    monkeypatch.setattr(evidence.Path, "rglob", no_walk)
    monkeypatch.setattr(evidence.Path, "glob", no_walk)
    result = evidence.audit(tmp_path)
    assert result["P1_admitted"] is False
    assert result["neural_work"] == 0
    assert all(stage["status"] == "blocked" and not stage["admitted"] for stage in result["stages"].values())
    assert all(slot["status"] == "not_run" for slot in result["required_report_slots"].values())
    assert result["documents"]["closure"]["anchors"][0]["line"] == 2


def test_missing_documents_do_not_hide_blocker(tmp_path, capsys):
    assert evidence.main(["--project-root", str(tmp_path)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert all(doc["status"] == "blocked" for doc in result["documents"].values())
    assert not result["efficacy_claim"]


def search_report():
    return {"rows": [{"root_sha256": str(index) * 64, "full_replay_exact": True,
        "planning_predict_samples": 80, "budget": {
            "binding": {"root_id": str(index) * 64}, "predict_sample_limit": 82,
            "attempted_predict_samples": 80 + replay,
            "phases": [{"phase": phase, "predict_samples": count, "completed": True,
                        "failed_predict_calls": 0} for phase, count in (("planning", 80), ("evaluation", replay))]}}
        for index, replay in enumerate((2, 2, 1, 2))]}


def test_search_v2_saved_counts_not_new_cuda_or_physics():
    result = evidence.search_accounting(search_report())
    assert result["status"] == "saved_receipts_consistent"
    assert result["actual_predict_samples"] == 327
    assert result["per_root_actual_samples"] == [82, 82, 81, 82]
    assert result["physical_execution_evidence"] is False
    assert result["compute_matched_claim"] is False


@pytest.mark.parametrize("damage", ["count", "binding", "duplicate_phase", "incomplete", "cap", "duplicate_root"])
def test_search_bad_saved_receipts_block(damage):
    payload = search_report()
    row = payload["rows"][0]
    if damage == "count":
        row["budget"]["attempted_predict_samples"] = 81
    elif damage == "binding":
        row["budget"]["binding"]["root_id"] = "bad"
    elif damage == "duplicate_phase":
        row["budget"]["phases"][1]["phase"] = "planning"
    elif damage == "incomplete":
        row["budget"]["phases"][1]["completed"] = False
    elif damage == "cap":
        row["budget"]["predict_sample_limit"] = 100
    else:
        payload["rows"][1]["root_sha256"] = row["root_sha256"]
    assert evidence.search_accounting(payload)["status"] == "blocked"


def test_missing_v1_is_failed_attempt_not_unrun_experiment(tmp_path):
    result = evidence.audit(tmp_path)
    entry = result["reports"]["program_search_v1"]
    assert entry["availability"] == "missing"
    assert entry["evidence_status"] == "blocked"
    assert "failed attempt" in entry["attempt_status"]
    assert result["reports"]["program_search_v2"]["evidence_status"] == "not_run"


def test_physical_execution_preserves_guard_stop_and_all_failures(tmp_path):
    spec = evidence.REPORTS["search_physical"]
    rows = [{"root_index": i, "root_sha256": str(i) * 64,
             "status": "observation_unavailable" if i == 3 else "completed",
             "error": "observation lost" if i == 3 else None,
             "planned_commands": 5 if i == 2 else 10,
             "completed_commands": 10 if i < 2 else 5,
             "remaining_commands_not_executed": 5 if i == 3 else 0,
             "official": {"official_success": False, "object_only_success": False}} for i in range(4)]
    write_report(tmp_path / "le-wm-main", {"schema": spec[1], "status": spec[2],
        "gates": {gate: True for gate in spec[3]}, "rows": rows, "P1_admission": False}, "results/" + spec[0])
    result = evidence.audit(tmp_path)
    saved = result["reports"]["search_physical"]
    assert saved["evidence_status"] == "report_checks_pass"
    assert saved["saved_execution_rows"] == rows
    assert len(saved["saved_execution_rows"]) == 4
    assert result["stages"]["P4"]["status"] == "blocked"
