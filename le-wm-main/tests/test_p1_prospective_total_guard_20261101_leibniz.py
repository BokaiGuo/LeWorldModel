import hashlib
from pathlib import Path

import pytest

from experiments.spectral_reachability import p1_prospective_total_guard_20261101_leibniz as g


class Handle:
    def __init__(self, module, hook):
        self.module, self.hook = module, hook

    def remove(self):
        self.module.hooks.remove(self.hook)


class Module:
    def __init__(self):
        self.hooks = []

    def register_forward_pre_hook(self, hook):
        self.hooks.append(hook)
        return Handle(self, hook)

    def __call__(self):
        for hook in list(self.hooks):
            hook(self, ())


class Model:
    def __init__(self):
        for name in (n for names in g.HOOKS.values() for n in names):
            setattr(self, name, Module())


def contract():
    path = Path(g.__file__).resolve()
    return {"schema": g.NAME + ".contract.v1", "run_id": "test", "arm": "cem",
            "caps": dict.fromkeys(g.AXES, 10),
            "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()},
            "identity": dict.fromkeys(g.IDENTITY, "a" * 64),
            "shared_initialization": "full_standalone_charge_each_arm",
            "hook_modules": g.HOOKS, "scope": "per_root_per_arm_start_before_load"}


def guard(tmp_path, value=None):
    value = value or contract()
    return g.Guard(tmp_path / "events.jsonl", value, g.digest(value))


def test_nested_full_vector_and_hash_replay(tmp_path):
    ledger = guard(tmp_path)
    model = Model()
    ledger.install_model_sentinels(model)
    ledger.dispatch("shared_setup", "model_load", 1, lambda: None)
    ledger.dispatch("feedback", "feedback", 1, lambda: ledger.dispatch(
        "feedback", "source_encode", 2, lambda: (model.encoder(), model.projector())))
    ledger.dispatch("planning", "action_encode", 1, model.action_encoder)
    ledger.dispatch("planning", "predict", 3, lambda: (model.predictor(), model.pred_proj()))
    ledger.dispatch("planning", "backward", 4, lambda: None)
    ledger.dispatch("planning", "candidate", 2, lambda: None)
    ledger.dispatch("physical", "physical_dispatch", 1,
                    lambda: ledger.dispatch("physical", "physics_step", 10, lambda: None))
    result = ledger.close()
    audit = g.verify_journal(result["journal"], result["sha256"], g.digest(contract()))
    assert audit["charged"]["source_encode_samples"] == 2
    assert audit["charged"]["predict_samples"] == 3
    assert audit["charged"]["vjp_rows"] == 4
    assert audit["charged"]["physics_steps"] == 10
    assert not audit["coverage_verified"] and audit["status"] == "closed_unverified"


def test_direct_model_bypass_fails_and_is_recorded(tmp_path):
    ledger, model = guard(tmp_path), Model()
    ledger.install_model_sentinels(model)
    with pytest.raises(ValueError, match="unguarded"):
        model.predictor()
    receipt = ledger.close()
    assert g.verify_journal(receipt["journal"], receipt["sha256"], g.digest(contract()))["status"] == "failed_or_blocked"


def test_missing_hook_fails(tmp_path):
    ledger = guard(tmp_path)
    ledger.install_model_sentinels(Model())
    with pytest.raises(ValueError, match="missing model hook"):
        ledger.dispatch("planning", "predict", 1, lambda: None)
    receipt = ledger.close()
    assert receipt["status"] == "failed_or_blocked"
    g.verify_journal(receipt["journal"], receipt["sha256"], g.digest(contract()))


def test_exhaustion_blocks_before_dispatch_and_retains_attempts(tmp_path):
    value = contract()
    value["caps"]["candidate_evaluations"] = 1
    ledger = guard(tmp_path, value)
    ledger.dispatch("planning", "candidate", 1, lambda: None)
    calls = []
    with pytest.raises(ValueError, match="budget exceeded"):
        ledger.dispatch("planning", "candidate", 1, lambda: calls.append(True))
    assert calls == []
    with pytest.raises(ValueError, match="fail-stopped"):
        ledger.dispatch("planning", "candidate", 1, lambda: None)
    receipt = ledger.close()
    audit = g.verify_journal(receipt["journal"], receipt["sha256"], g.digest(value))
    assert audit["charged"]["candidate_evaluations"] == 1


def test_failed_call_is_charged(tmp_path):
    ledger = guard(tmp_path)
    def bad():
        raise RuntimeError("failure")
    with pytest.raises(RuntimeError):
        ledger.dispatch("physical", "physics_step", 3, bad)
    receipt = ledger.close()
    audit = g.verify_journal(receipt["journal"], receipt["sha256"], g.digest(contract()))
    assert audit["charged"]["physics_steps"] == 3 and audit["status"] == "failed_or_blocked"


def test_hash_and_scope_tamper_rejected(tmp_path):
    value = contract()
    ledger = guard(tmp_path, value)
    receipt = ledger.close()
    with pytest.raises(ValueError, match="journal hash"):
        g.verify_journal(receipt["journal"], "0" * 64, g.digest(value))
    value["shared_initialization"] = "free"
    with pytest.raises(ValueError, match="scope"):
        g.validate(value, g.digest(value))


def test_facade_refuses_cpu():
    class Cpu:
        device = type("Device", (), {"type": "cpu"})()
        shape = (1, 1)
    with pytest.raises(ValueError, match="CUDA"):
        g.LeWMFacade._cuda(Cpu())


def test_pair_rejects_mismatched_caps_or_root():
    s, c = contract(), contract()
    s.update(arm="structured", run_id="s")
    c["run_id"] = "c"
    assert g.validate_pair(s, c)["pair_compatible"]
    c["caps"]["predict_calls"] += 1
    with pytest.raises(ValueError, match="paired arms/caps"):
        g.validate_pair(s, c)
    c["caps"]["predict_calls"] -= 1
    c["identity"]["root"] = "b" * 64
    with pytest.raises(ValueError, match="paired identity"):
        g.validate_pair(s, c)


def test_source_drift_fails_closed(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("original")
    value = contract()
    value["source_sha256"][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    ledger = guard(tmp_path, value)
    source.write_text("changed")
    receipt = ledger.close()
    assert receipt["status"] == "failed_or_blocked"
    with pytest.raises(ValueError, match="source drift"):
        g.verify_journal(receipt["journal"], receipt["sha256"], g.digest(value))


def test_wrong_model_module_inside_dispatch_is_bypass(tmp_path):
    ledger, model = guard(tmp_path), Model()
    ledger.install_model_sentinels(model)
    with pytest.raises(ValueError, match="unguarded"):
        ledger.dispatch("planning", "predict", 1, model.action_encoder)
    receipt = ledger.close()
    assert receipt["status"] == "failed_or_blocked"
    g.verify_journal(receipt["journal"], receipt["sha256"], g.digest(contract()))
