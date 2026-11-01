import json

import pytest

from experiments.spectral_reachability import d2p_downstream_frontier as frontier


def test_missing_evidence_fails_closed(tmp_path):
    result = frontier.audit(tmp_path)
    assert not result["P1_admitted"]
    assert all(not value["admitted"] and not value["physically_validated"] for value in result["stages"].values())
    assert all(not value["available"] for value in result["evidence"].values())
    assert result["neural_work"] == result["new_physical_execution"] == 0


def test_oversized_input_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(frontier, "MAX_BYTES", 4)
    path = tmp_path / "report.json"
    path.write_text("12345")
    assert "bounded-read" in frontier.inspect(tmp_path, "report.json")["error"]


@pytest.mark.parametrize("text", ["[]", "{", "null"])
def test_invalid_json_fails_closed(tmp_path, text):
    (tmp_path / "report.json").write_text(text)
    assert not frontier.inspect(tmp_path, "report.json")["available"]


def test_sources_never_executed(tmp_path):
    (tmp_path / "source.py").write_text("raise RuntimeError('must not execute')\ndef sample(): return 1\n")
    result = frontier.inspect(tmp_path, "source.py", source=True)
    assert result["available"] and result["symbols"] == ["sample"]


def bank():
    return {"rows": [{"root_sha256": str(i), "selected_candidate_id": "0", "branches": [
        {"candidate_id": str(j), "stages": [{"official_success": False, "object_only_success": False} for _ in range(2)]}
        for j in range(16)]} for i in range(4)]}


def test_saved_bank_counts_not_efficacy():
    assert frontier.task_contract_counts(bank()) == {"roots": 4, "branches": 64, "saved_stages": 128,
        "official_terminal_successes": 0, "object_only_terminal_successes": 0}


@pytest.mark.parametrize("damage", ["duplicate", "missing_label", "truncated", "missing_root"])
def test_invalid_saved_bank_rejected(damage):
    report = bank()
    if damage == "duplicate":
        report["rows"][1]["root_sha256"] = "0"
    elif damage == "missing_label":
        del report["rows"][0]["branches"][0]["stages"][0]["official_success"]
    elif damage == "truncated":
        report["rows"][0]["branches"].pop()
    else:
        report["rows"].pop()
    with pytest.raises(ValueError):
        frontier.task_contract_counts(report)


def test_positive_flags_do_not_promote_or_write(tmp_path):
    path = tmp_path / frontier.REPORTS["official_contract"]
    path.parent.mkdir(parents=True)
    report = bank() | {"schema": "pusht_sealed_task_contract.v1", "status": "completed", "P1_admission": True}
    path.write_text(json.dumps(report))
    before = path.read_bytes()
    result = frontier.audit(tmp_path)
    assert path.read_bytes() == before
    assert not result["P1_admitted"]
    assert all(not stage["admitted"] for stage in result["stages"].values())


def test_symlink_escape_rejected(tmp_path):
    (tmp_path / "outside").symlink_to("/etc/passwd")
    assert "outside evidence root" in frontier.inspect(tmp_path, "outside")["error"]
