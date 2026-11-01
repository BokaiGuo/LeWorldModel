import hashlib
from pathlib import Path

import pytest

from experiments.spectral_reachability import p1_total_guard_host_bridge_20261101_leibniz as bridge

g, spec = bridge.total, bridge.legacy.spec


def contract(arm):
    files = (Path(g.__file__).resolve(), Path(bridge.__file__).resolve())
    return {"schema": g.NAME + ".contract.v1", "run_id": arm, "arm": arm,
            "caps": dict.fromkeys(g.AXES, 10), "identity": dict.fromkeys(g.IDENTITY, "a" * 64),
            "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "shared_initialization": "full_standalone_charge_each_arm", "hook_modules": g.HOOKS,
            "scope": "per_root_per_arm_start_before_load"}


def old_contract():
    ident = dict.fromkeys(spec.IDENTITY, "a" * 64)
    arms = {a: {"initializer_sha256": "a" * 64, "optimizer_sha256": "a" * 64,
                "optimizer_config_sha256": "a" * 64} for a in spec.ARMS}
    caps = dict.fromkeys(spec.AXES, 10)
    return {"schema": spec.NAME + ".contract.v1", "identity": ident, "arms": arms,
            "levels": [{"id": "node", "caps": caps},
                       {"id": "larger", "caps": {k: v+1 for k, v in caps.items()}}],
            "scope": "standalone_per_root_per_arm_including_attributable_setup",
            "phase_policy_sha256": "a" * 64}


def test_both_arms_share_draft_and_fail_closed():
    pair = bridge.pair_protocol(contract("structured"), contract("cem"))
    assert pair["pair_check"]["pair_compatible"]
    assert pair["coverage"]["host_execution_authorized"] is False
    with pytest.raises(ValueError, match="direct_begin_end"):
        bridge.require_host_ready()


@pytest.mark.parametrize("arm", ["structured", "cem"])
def test_actual_legacy_driver_selection_is_precharged(tmp_path, arm):
    c = contract(arm)
    guard = g.Guard(tmp_path / "total.jsonl", c, g.digest(c))
    old = old_contract()
    driver = bridge.legacy.Driver(tmp_path / "node", old, spec.digest(old), "structured", "node", "x")
    reservation = dict.fromkeys(spec.AXES, 0)
    reservation["selection_evaluations"] = 1
    with bridge.node_patch(guard):
        assert driver.dispatch("screening", "selection", reservation, lambda: 7) == 7
    result = driver.finish()
    receipt = guard.close()
    audited = g.verify_journal(receipt["journal"], receipt["sha256"], g.digest(c))
    assert result["status"] == "closed_unverified"
    assert audited["charged"]["candidate_evaluations"] == 1


def test_unknown_physical_resource_aborts_before_callback(tmp_path):
    c = contract("cem")
    guard = g.Guard(tmp_path / "total.jsonl", c, g.digest(c))
    calls = []
    with pytest.raises(ValueError, match="unmapped physical"):
        bridge.physical_call(guard, lambda *args: calls.append("ran"), None, "set_state", lambda: None)
    assert not calls
    receipt = guard.close()
    assert receipt["status"] == "failed_or_blocked"


def test_mixed_node_reservation_rejected_before_original(tmp_path):
    c = contract("cem")
    guard = g.Guard(tmp_path / "total.jsonl", c, g.digest(c))
    r = dict.fromkeys(spec.AXES, 0)
    r.update(selection_evaluations=1, predict_calls=1, predict_samples=1)
    calls = []
    with pytest.raises(ValueError, match="mixed"):
        bridge.node_dispatch(guard, lambda *args: calls.append(True), None, "trial", "selection", r, lambda: None)
    assert calls == []
    assert guard.close()["status"] == "failed_or_blocked"


def test_physical_budget_blocks_before_actual_callback(tmp_path):
    c = contract("cem")
    c["caps"]["physics_steps"] = 0
    guard = g.Guard(tmp_path / "total.jsonl", c, g.digest(c))
    calls = []
    with pytest.raises(ValueError, match="budget exceeded"):
        bridge.physical_call(guard, lambda *args: calls.append(True), None, "physics_step", lambda: None)
    assert calls == []
    assert guard.close()["status"] == "failed_or_blocked"
