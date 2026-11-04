import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.spectral_reachability import p2_multi_joint_development_20261102_codexexclusive as m
from experiments.spectral_reachability import p2_multi_joint_saved_20261102_codexexclusive as saved
from experiments.spectral_reachability import p2_multi_joint_physical_20261102_codexexclusive as physical


def test_fixed_two_roots_and_single_shared_bank():
    p = m.plan_value()
    assert [(c["base"], c["root"]) for c in p["cases"]] == [(0, 0), (2, 1)]
    assert len({c["task_id"] for c in p["cases"]}) == 2
    assert len(p["bank_sha256"]) == 16
    assert p["jacobian_shapes"] == [[192, 10], [192, 20]]
    assert p["equal_total_compute"] is p["independent_confirmation"] is False


def test_cannot_use_cpu_neural_entry_or_run_without_authority(tmp_path):
    with pytest.raises(ValueError, match="CUDA only"):
        m.run_root(None, None, None, None, "cpu")
    with pytest.raises(ValueError, match="authorization"):
        m.execute("unused", "unused", tmp_path / "out", "bad")


def test_ledger_hard_reserve(tmp_path):
    with pytest.raises(ValueError, match="output reserve"):
        m.BoundedLedger(tmp_path / "journal", 1024)


def test_two_root_analytic_schedule_and_saved_tamper(tmp_path, monkeypatch):
    cases, bank = m.input_cases()
    folder = tmp_path / (m.NAME + "_RUN_fixture")
    folder.mkdir()
    rows = []
    for case in cases:
        path = folder / f"root_{case['base']}_journal.jsonl"
        ledger = m.BoundedLedger(path, 128 * 1024 * 1024)
        meter = m.host.TimedMeter(ledger, lambda: None)
        def analytic(state, history, action):
            for axis in m.matched.HOOKS:
                meter.call(axis, lambda: None)
            token = action.reshape(1, 10)
            projected = torch.cat((token, token.new_zeros((1, 182))), dim=1)
            return torch.cat((state[:, 1:], (state[:, -1] + projected)[:, None]), dim=1), history
        transition = m.host.measured_transition(analytic, meter)
        try:
            result = m._run_root(case, bank, transition, meter, "cpu")
        finally:
            ledger.close()
        audit = m.consumer.audit_complete(path, result, bank)
        handoff = m.physical_handoff(case, result)
        assert handoff["physical_ready"] is False and len(handoff["arms"]) == 2
        assert all(x["full25_float32_hex"].startswith(case["prefix_float32_hex"]) for x in handoff["arms"])
        recipe = physical.build_recipe(case, handoff, "fixture_model_report")
        for entry, arm in zip(recipe["plans"], handoff["arms"], strict=True):
            assert entry["raw_float32_hex"] == arm["full25_float32_hex"]
            assert entry["native_root_sha256"] == case["native_root_sha256"]
        forged = json.loads(json.dumps(handoff))
        forged["arms"][0]["full25_float32_hex"] = "00" * 100
        with pytest.raises(ValueError, match="fixed actions"):
            physical.build_recipe(case, forged, "fixture_model_report")
        result_sha = m.host.base.write_new(folder / f"root_{case['base']}_result.json", result)
        handoff_sha = m.host.base.write_new(folder / f"root_{case['base']}_handoff.json", handoff)
        rows.append({"base": case["base"], "root": case["root"], "status": "complete",
                     "journal_sha256": m.host.base.file_sha(path), "result_sha256": result_sha,
                     "handoff_sha256": handoff_sha, "audit": audit})
    m.host.base.write_new(folder / "binding.json", {"plan": "fixture", "plan_sha256": "fixture"})
    report = {"schema": m.NAME + ".report_v1", "plan_sha256": "fixture",
              "status": "complete_model_only_physical_blocked", "error": None,
              "rows": rows, "frozen_parameters_unchanged": True, "device": {"uuid": m.host.UUID},
              "wall_seconds": 1.0, "physical_execution": False, "historical_sra_authenticated": False,
              "equal_total_compute": False, "independent_confirmation": False, "P2_admission": False}
    digest = m.host.base.write_new(folder / "report.json", report)
    monkeypatch.setattr(m, "RESULTS", tmp_path)
    monkeypatch.setattr(m, "verify_plan", lambda *args: {"source_limits": ["fixture only"]})
    monkeypatch.setattr(m, "input_cases", lambda: (cases, bank))
    summary = saved.consume(folder, digest)
    assert [x["base"] for x in summary["roots"]] == [0, 2]
    assert all(x["audit"]["transition_calls"] == 640 for x in summary["roots"])
    assert summary["physical_execution"] is False
    handoff_path = folder / "root_2_handoff.json"
    content = json.loads(handoff_path.read_bytes())
    content["arms"][0]["candidate_index"] = 99
    handoff_path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="source SHA"):
        saved.consume(folder, digest)


def test_physical_authorization_checked_before_any_execution():
    with pytest.raises(ValueError, match="authorization"):
        physical.run_host("unused", "unused", "wrong")


def test_physical_saved_consumer_two_archived_parser_fixtures(tmp_path, monkeypatch):
    import shutil
    from test_p2_joint_physical_20261102_codexexclusive import historical_fixture
    from experiments.spectral_reachability import p2_multi_joint_physical_saved_20261102_codexexclusive as pc
    directory, old_report, events, prepared = historical_fixture()
    session = tmp_path / "session"
    session.mkdir()
    rows, recipes = [], []
    for base, root in ((0, 0), (2, 1)):
        folder = session / (physical.physical.NAME + f"_MULTI_B{base}_01")
        shutil.copytree(directory, folder)
        for name in ("binding.json", "journal.jsonl", "report.json"):
            (folder / name).unlink()
        journal = b"".join((json.dumps(e, sort_keys=True) + "\n").encode() for e in events)
        (folder / "journal.jsonl").write_bytes(journal)
        old_sha = f"old_{base}"
        m.host.base.write_new(folder / "binding.json", {"plan_sha256": "fixture", "base": base,
                                                         "old_plan_sha256": old_sha})
        report = dict(old_report, schema=physical.NAME + ".root_report_v1", plan_sha256="fixture",
                      journal_sha256=m.host.base.sha(journal))
        report_sha = m.host.base.write_new(folder / "report.json", report)
        rows.append({"base": base, "status": "recorded_development_unconsumed",
                     "output": str(folder), "report_sha256": report_sha})
        recipes.append({"case": {"base": base, "root": root, "native_root_sha256": "fixture",
                                  "old_physical_prep_sha256": old_sha}, "prepared": prepared})
    report = {"schema": physical.NAME + ".session_v1", "plan_sha256": "fixture",
              "status": "recorded_development_unconsumed", "error": None, "rows": rows,
              "physical_execution": True, "equal_total_compute": False, "independent_confirmation": False,
              "P2_admission": False, "wall_seconds": 1.0}
    session_sha = m.host.base.write_new(session / "session_report.json", report)
    monkeypatch.setattr(physical, "SESSION", session)
    monkeypatch.setattr(physical, "verify", lambda *args: {"recipes": recipes})
    value = pc.consume("fixture", "fixture", session_sha)
    assert [x["summary"]["png_verified"] for x in value["roots"]] == [222, 222]
    (session / (physical.physical.NAME + "_MULTI_B2_01") / "journal.jsonl").write_bytes(b"broken")
    with pytest.raises(ValueError, match="root report/journal binding"):
        pc.consume("fixture", "fixture", session_sha)
