import copy
from pathlib import Path

import pytest

from experiments.spectral_reachability import p3_preoutcome_winner_physical_20261102_codexexclusive as physical
from experiments.spectral_reachability import p3_preoutcome_winner_physical_saved_20261102_codexexclusive as saved
from experiments.spectral_reachability import p3_venv_winner_cpu_physical_20261102_codexexclusive as old


@pytest.fixture(scope="module")
def previous_saved():
    directory = physical.ROOT / "results/p3_venv_winner_cpu_physical_20261102_codexexclusive_RUN_01"
    prepared = physical.ROOT / "results/p3_venv_winner_cpu_physical_20261102_codexexclusive_PREP_01.json"
    if not directory.exists() or not prepared.exists():
        pytest.skip("existing saved fixture unavailable")
    plan = physical.moving.read(prepared)
    report = physical.moving.read(directory / "report.json")
    events = physical.moving.journal(directory / "journal.jsonl", report["journal_chain_tail"])
    _, data = old.material()
    plan["model_accounting"] = {"fixture": "historical saved record"}
    events[0]["prepare"] = plan
    return plan, data["references"], report, events, directory


def test_actual_material_reconstructs_fixed_winners():
    plan, data = physical.material()
    assert [(x["arm"], x["root_slot"], x["candidate_index"]) for x in plan["plans"]] == [
        ("signal", 0, 0), ("signal", 1, 4), ("reverse", 0, 14), ("reverse", 1, 4)]
    assert plan["plans"][1]["raw_float32_sha256"] == plan["plans"][3]["raw_float32_sha256"]
    assert len(data["references"]) == 2 and not plan["independent_outcome"]


def test_read_existing_saved_fixture(previous_saved):
    plan, refs, report, events, directory = previous_saved
    result = saved.validate(plan, refs, report, events, directory)
    assert result["status"] == "saved_physical_records_verified"
    assert len(result["branches"]) == 4 and not result["independent_physics_attestation"]


@pytest.mark.parametrize("field", ["action", "count", "root", "early_stop", "closure"])
def test_reject_counterexamples(previous_saved, field):
    plan, refs, report, events, directory = copy.deepcopy(previous_saved)
    if field == "action":
        next(e for e in events if e["kind"] == "returned_transition" and e["phase"] == "plan")["command"][0] = 9
    elif field == "count":
        report["counts"]["public_step"]["attempted"] -= 1
        events[-1]["counts"] = report["counts"]
    elif field == "root":
        next(e for e in events if e["kind"] == "fresh_observation" and e["receipt"]["step"] == 0)["receipt"]["pixel_sha256"] = "0" * 64
    elif field == "early_stop":
        next(e for e in events if e["kind"] == "returned_transition" and e["phase"] == "plan")["truncated"] = True
    else:
        next(e for e in events if e["kind"] == "branch_closed")["receipt"]["arm"] = "bad"
    with pytest.raises(ValueError):
        saved.validate(plan, refs, report, events, directory)


def test_native_stop_is_censored(previous_saved):
    plan, refs, report, events, directory = copy.deepcopy(previous_saved)
    row = report["rows"][0]
    trans = [e for e in events if e.get("branch") == 0 and e["kind"] == "returned_transition"
             and e.get("phase") == "plan"]
    trans[-1]["truncated"] = True
    row.update(status="native_stopped", truncated=True)
    next(e for e in events if e["kind"] == "branch_closed")["receipt"] = row
    result = saved.validate(plan, refs, report, events, directory)
    assert result["branches"][0]["endpoint"] == "censored_native_stop"


def test_execution_requires_authority_and_hidden_cuda(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="authority"):
        physical.run("missing", "bad", "missing", "bad")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(ValueError, match="hidden"):
        physical.run("missing", "bad", "missing", physical.AUTH)
