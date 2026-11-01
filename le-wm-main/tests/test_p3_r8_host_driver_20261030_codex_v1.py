"""CPU-only durable lifecycle tests; no model, CUDA or physical environment."""
import json

import pytest

from experiments.spectral_reachability import p3_r8_host_driver_20261030_codex_v1 as d


def setup(monkeypatch, tmp_path):
    base = d.v2.read_json(d.BASE)
    plan = dict(output=str(tmp_path / "run"), decision_cap=10_000_000,
                decision_reserve=1_000_000, capsule_cap=10_000_000,
                capsule_reserve=1_000_000, min_free_mib=1, min_total_mib=1,
                sources=d.source_hashes())
    monkeypatch.setattr(d, "validate", lambda p, h: (base, {"tasks": [1, 2]}))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", base["gpu_uuid"])
    snapshot = lambda uuid: dict(uuid=uuid, free_mib=10, total_mib=10)
    return plan, base, snapshot


def test_lifecycle_with_real_durable_and_capsule_stores(monkeypatch, tmp_path):
    plan, base, snapshot = setup(monkeypatch, tmp_path)
    def execute(p, b, prepared, ds, ns, cs, gpu):
        ds.event({"kind": "R8_test", "value": 3})
        ns.event("physical", "record", "fixture", 0, {})
        first = cs.capsule(0, 0, [{"kind": "actual_fixture"}])
        second = cs.capsule(1, 0, [{"kind": "different_stage"}])
        assert first["path"] != second["path"]
        rows = [dict(slot=s, root=r, policy=policy, status="callback_complete_not_certified",
                     cleanup="returned", error=None) for s, r, policy in d.r8.ROSTER]
        return {"rows": rows}, {"fixture": True}, None
    d.run(plan, "fixture", d.AUTH, execute=execute, snapshotter=snapshot)
    out = tmp_path / "run"
    end = json.loads((out / "end.json").read_text())
    assert end["candidate_complete"] and not end["full_saved_strict"]
    assert end["decision_sha256"] == d.old.file_sha(out / "decision/journal.jsonl")
    assert len(end["capsule_files"]) == 3
    assert len(end["rows"]) == 6
    with pytest.raises(FileExistsError):
        d.run(plan, "fixture", d.AUTH, execute=execute, snapshotter=snapshot)


def test_failed_execution_preserves_partial_and_no_retry(monkeypatch, tmp_path):
    plan, _, snapshot = setup(monkeypatch, tmp_path)
    def execute(p, b, prepared, ds, ns, cs, gpu):
        ds.event({"kind": "partial_fixture"})
        cs.capsule(0, 0, [1])
        raise RuntimeError("intentional")
    with pytest.raises(RuntimeError, match="intentional"):
        d.run(plan, "fixture", d.AUTH, execute=execute, snapshotter=snapshot)
    out = tmp_path / "run"
    assert (out / "decision/journal.jsonl").exists()
    assert (out / "capsules/signal_s0_0000.json").exists()
    failure = json.loads((out / "failure.json").read_text())
    assert failure["stage"] == "runtime" and failure["retry_authorized"] is False
    assert not (out / "end.json").exists()


def test_insufficient_gpu_or_wrong_authorization_before_output(monkeypatch, tmp_path):
    plan, _, snapshot = setup(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="authorization"):
        d.run(plan, "fixture", "no", snapshotter=snapshot)
    with pytest.raises(ValueError, match="GPU admission"):
        d.run(plan, "fixture", d.AUTH,
              snapshotter=lambda uuid: dict(uuid=uuid, free_mib=0, total_mib=10))
    assert not (tmp_path / "run").exists()


def test_plan_rejects_source_tamper(tmp_path):
    output = d.ROOT / ("results/" + d.NAME + "_RUN_TEST_NEVER_START")
    plan = d.build_plan(output)
    plan["sources"][d.SOURCE_NAMES[0]] = "0" * 64
    with pytest.raises(ValueError, match="fixed plan"):
        d.validate(plan, d.v2.canonical_sha(plan))
