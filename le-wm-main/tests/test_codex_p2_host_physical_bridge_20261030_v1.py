from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import codex_p2_host_physical_bridge_20261030_v1 as b


def fake_report():
    return dict(status="exploratory_complete", fixture=False, P1_admission=False,
                P2_admission=False, physical_authorized=False, equal_total_compute=False,
                results=[{"arm": arm} for arm in b.ARMS])


def test_plan_extraction_calls_strict_validator_for_both_arms(monkeypatch):
    calls = []
    def validate(result, **kwargs):
        calls.append(kwargs["arm"])
        raw = [[0.0, 0.0] for _ in range(10)]
        return dict(actions=raw, candidate_id=kwargs["arm"], witness={"receipt": {"actions": raw}})
    monkeypatch.setattr(b.frontier, "validate_result", validate)
    ready = dict(root_sha256="a"*64, root_index=0, root={}, action_domain={})
    provenance = {arm: {"candidate_bank_sha256": b.frontier.digest(
        {"bank": {}, "arm": arm, "contract": b.frontier.contract()})} for arm in b.ARMS}
    plans = b.validated_plans(fake_report(), {}, ready, {}, provenance)
    assert calls == list(b.ARMS) and len(plans) == 2
    assert all(len(p["commands"]) == 10 and not p["selection_uses_physical_outcome"] for p in plans)


def test_fixture_or_unadmitted_status_never_extracts(monkeypatch):
    monkeypatch.setattr(b.frontier, "validate_result", lambda *args, **kw: pytest.fail("must not enter"))
    report = fake_report()
    report["fixture"] = True
    with pytest.raises(ValueError):
        b.validated_plans(report, {}, {}, {}, {})


def test_tampered_inputs_rejected(tmp_path):
    path = tmp_path / "source"
    path.write_bytes(b"original")
    h = b.sha(path)
    b.check_sources({str(path): h})
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        b.check_sources({str(path): h})


def test_physics_blocked_without_any_callback():
    calls = []
    with pytest.raises(RuntimeError, match="BLOCKED"):
        b.physical_dispatch(lambda: calls.append(1))
    assert not calls


def test_wrong_bank_digest_rejected_before_candidate_validator(monkeypatch):
    monkeypatch.setattr(b.frontier, "validate_result", lambda *args, **kw: pytest.fail("must not enter"))
    with pytest.raises(ValueError, match="bank"):
        b.validated_plans(fake_report(), {}, {}, {}, {
            arm: {"candidate_bank_sha256": "0"*64} for arm in b.ARMS})


def test_host_rejects_cpu_before_output(tmp_path):
    class CPU:
        type = "cpu"
    output = tmp_path / "attempt"
    with pytest.raises(ValueError, match="CUDA"):
        b.host_plan(model=None, device=CPU(), ready={}, compiled={}, provenance_by_arm={},
                    caps={}, output=output, source_bindings={}, expected_model_state={},
                    authorization="EXPLORATORY_P2_HOST_CUDA_COORDINATED")
    assert not output.exists()


def test_host_failure_saved_with_synthetic_cuda_metadata(tmp_path, monkeypatch):
    fake_cuda = SimpleNamespace(is_available=lambda: True, synchronize=lambda device: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    monkeypatch.setattr(checked, "tensor_state_hash", lambda model: {"state": "pinned"})
    def fail(**kwargs):
        raise RuntimeError("synthetic planner failure")
    monkeypatch.setattr(b.gated, "run_native", fail)
    device = SimpleNamespace(type="cuda")
    model = SimpleNamespace(modules=lambda: [SimpleNamespace(training=False)],
                            parameters=lambda: [SimpleNamespace(requires_grad=False, device=device)],
                            buffers=lambda: [])
    sources = {str(Path(p).resolve()): b.sha(p) for p in
               (b.__file__, b.gated.__file__, b.frontier.__file__)}
    output = tmp_path / "attempt"
    with pytest.raises(RuntimeError, match="synthetic planner failure"):
        b.host_plan(model=model, device=device, ready={}, compiled={}, provenance_by_arm={},
                    caps={}, output=output, source_bindings=sources,
                    expected_model_state={"state": "pinned"},
                    authorization="EXPLORATORY_P2_HOST_CUDA_COORDINATED")
    record = json.loads((output / "bridge_report.json").read_text())
    assert record["status"] == "partial_or_failed"
    assert record["physical_authorized"] is False and record["P2_admission"] is False
    assert (output / "binding.json").is_file()
