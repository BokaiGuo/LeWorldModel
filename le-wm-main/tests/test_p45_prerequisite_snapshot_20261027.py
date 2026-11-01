"""Offline saved-content checks; fixtures are NOT neural/efficacy evidence."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import p45_prerequisite_snapshot_20261027 as audit


def composed():
    rows = []
    for root in "abcd":
        context = {"latent_history": [[0.0]], "action_history": []}
        actions = [[0.0, 0.0]] * 10
        transitions = [{"start_step": i*5, "end_step": (i+1)*5,
                        "input_context_sha256": audit.digest(context),
                        "output_context_sha256": audit.digest(context),
                        "actions_sha256": audit.digest(actions[i*5:(i+1)*5])} for i in range(2)]
        body = {"root": {"root_id": root*64, "context": context}, "actions": actions,
                "actions_sha256": audit.digest(actions), "transitions": transitions,
                "terminal_context": context}
        receipt = body | {"receipt_sha256": audit.digest(body)}
        rows.append({"root_sha256": root*64, "programs": [
            {"candidate_id": f"prefix_{i}", "receipt": deepcopy(receipt),
             "split_full_exact": True, "saved_forecast_matches": True} for i in range(16)]})
    return {"rows": rows, "predict": [{"predict_samples": 384, "predict_calls": 384,
            "completed": True, "failed_predict_calls": 0}]}


def test_composition_rehash_not_model_replay():
    result = audit.composition_arithmetic(composed())
    assert result["status"] == "saved_content_recomputed"
    assert result["rehash_chain_edges"] == 128
    assert result["independent_transition_replay"] is False
    assert result["mechanism_proven"] is False


@pytest.mark.parametrize("damage", ["hash", "chain", "actions", "duplicate", "missing", "root", "count", "bool", "flag"])
def test_composition_damage_fails_closed(damage):
    report = composed()
    item = report["rows"][0]["programs"][0]
    receipt = item["receipt"]
    if damage == "hash":
        receipt["receipt_sha256"] = "0"*64
    elif damage == "chain":
        receipt["transitions"][1]["input_context_sha256"] = "0"*64
        receipt["receipt_sha256"] = audit.digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
    elif damage == "actions":
        receipt["actions"][0] = [1.0, 0.0]
    elif damage == "duplicate":
        report["rows"][0]["programs"][1]["candidate_id"] = "prefix_0"
    elif damage == "missing":
        report["rows"][0]["programs"].pop()
    elif damage == "root":
        report["rows"][0]["root_sha256"] = "z"*64
    elif damage == "count":
        report["predict"][0]["predict_samples"] = 383
    elif damage == "bool":
        report["predict"][0]["predict_calls"] = True
    else:
        item["split_full_exact"] = 1
    assert audit.composition_arithmetic(report)["status"] == "blocked"


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'[]', b'{}'])
def test_reuses_existing_strict_parser(raw):
    with pytest.raises(ValueError):
        audit.prior.strict_json(raw)


def test_named_larger_read_does_not_change_old_limit(tmp_path):
    path = tmp_path / "larger.json"
    path.write_bytes(b" " * 2_000_001)
    assert len(audit.read(tmp_path, "larger.json")) == 2_000_001
    assert audit.prior.MAX_BYTES == 2_000_000
    with pytest.raises(ValueError, match="2 MB"):
        audit.prior.bounded_read(tmp_path, "larger.json")


def test_escape_directory_and_size_rejected(tmp_path, monkeypatch):
    (tmp_path / "escape").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="escapes"):
        audit.read(tmp_path, "escape")
    with pytest.raises(ValueError, match="regular"):
        audit.read(tmp_path, ".")
    (tmp_path / "too_big").write_bytes(b"12345")
    monkeypatch.setattr(audit, "MAX_BYTES", 4)
    with pytest.raises(ValueError):
        audit.read(tmp_path, "too_big")


@pytest.mark.parametrize("damage", ["integer_gate", "extra_failed", "schema", "protocol", "rows"])
def test_report_checks_fail_closed(tmp_path, damage):
    spec = ("fixture/report.json", "fixture.v1", "completed", ("cuda_only",))
    payload = {"schema": "fixture.v1", "status": "completed", "gates": {"cuda_only": True}}
    if damage == "integer_gate":
        payload["gates"]["cuda_only"] = 1
    elif damage == "extra_failed":
        payload["gates"]["unexpected"] = False
    elif damage == "schema":
        payload["schema"] = "other"
    elif damage == "protocol":
        payload["protocol"] = None
    else:
        payload["rows"] = [1]
    path = tmp_path / "le-wm-main/results/fixture/report.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload))
    entry, _ = audit.inspect(tmp_path, spec)
    assert entry["status"] == "blocked"
    assert entry["admission"] is False


def test_unknown_sources_not_zero_performance_or_comparable(tmp_path):
    docs = {k: audit.document(tmp_path, v) for k, v in audit.DOCS.items()}
    results = audit.baseline_findings(docs)
    assert all(r["classification"] == "source_unavailable" for r in results)
    assert all(r["same_protocol_misuse_detected_in_inspected_reports"] is None for r in results)
    assert all(r["direct_comparable"] is False for r in results)


def test_no_discovery_no_inference_no_promotion(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("discovery prohibited")
    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    result = audit.snapshot(tmp_path)
    assert result["P1_admitted"] is False
    assert all(not stage["admitted"] for stage in result["stages"].values())
    assert result["new_neural_calls"] == result["new_physical_execution"] == 0
    assert all(not p["formal_stage_satisfied"] for p in result["prerequisites"])
    assert result["all_read_bytes_stable"] is False


def test_cli_stdout_and_write_once(tmp_path, capsys):
    (tmp_path / "le-wm-main/results").mkdir(parents=True)
    assert audit.main(["--project-root", str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().out)["P1_admitted"] is False
    assert audit.main(["--project-root", str(tmp_path), "--write"]) == 1
    output = tmp_path / "le-wm-main/results" / audit.NAME / "manifest.json"
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        audit.main(["--project-root", str(tmp_path), "--write"])
    assert output.read_bytes() == before


def test_real_archive_saved_content_no_neural_execution():
    root = Path(__file__).resolve().parents[2]
    result = audit.snapshot(root)
    artifacts = result["artifacts"]
    assert artifacts["composition_replay"]["recomputed"]["rehash_receipts"] == 64
    assert artifacts["composition_replay"]["root_relation_to_calibration"]["overlap"] == 4
    assert artifacts["program_search_v2"]["recomputed"]["actual_predict_samples"] == 327
    counts = artifacts["reacher_sample_matched"]["recomputed"]["rows"]
    assert [r["seed"] for r in counts] == [20261040, 20261041]
    assert all(r["d2p_cem_samples_equal"] and not r["d2p_cem_calls_equal"] for r in counts)
    runtime = artifacts["representation_runtime"]["recomputed"]
    assert runtime["images"] == 12 and runtime["programs"] == 64
    assert runtime["max_direct_official_error"] == 0
    assert runtime["Q_task_efficacy_repaired"] is False
    assert artifacts["pointmaze_sra_p4"]["root_audit"]["outcome_roots"] == 144
    assert [r["classification"] for r in result["reported_baseline_audit"]] == [
        "insufficient_metadata", "insufficient_metadata", "insufficient_metadata", "context_only"]
    assert result["all_read_bytes_stable"] is True
