"""Synthetic metadata only; no model inference, CUDA, or physics."""
import pytest

from experiments.spectral_reachability import codex_p2_frontier_shared_cap_gate_20261030_v1 as g
from test_p2_composition_predict_frontier_cuda_20261027_v1 import fake_model, tensor, dispatch
from test_p2_composition_predict_frontier_20261027_v1 import base, legacy_inputs


def caps(**changes):
    result = dict(predict=168, action_encoder=168, jacobian=41, vjp_directions=7872, svd=42, solve=41)
    result.update(changes)
    return result


def test_real_frontier_hook_shared_allocation_and_predispatch_cap():
    model, replay, calls = fake_model()
    log = []
    gate = g.SharedCapGate(caps(), log.append)
    with g.guarded_entries(model, replay, gate) as entries:
        for spec, scope in zip(g.frontier.expected_entries(80), g.frontier.expected_scopes()):
            replay._p2_scope = scope
            dispatch(model, spec["kind"], tensor(spec["kind"]))
    assert len(calls) == len(entries) == 664
    assert all(gate.attempted[a]["predict"] == 168 and
               gate.attempted[a]["action_encoder"] == 168 for a in g.ARMS)
    assert g.frontier.validate_actual(entries)["standalone_predict"] == [168, 168]
    assert gate.snapshot()["equal_total_compute"] is False


def test_cap_failure_before_native_dispatch_no_refund():
    model, replay, calls = fake_model()
    gate = g.SharedCapGate(caps(action_encoder=0), lambda event: None)
    with g.guarded_entries(model, replay, gate):
        replay._p2_scope = g.frontier.expected_scopes()[0]
        with pytest.raises(ValueError, match="cap exhausted"):
            dispatch(model, "action_encoder", tensor("action_encoder"))
    assert not calls and gate.failed
    with pytest.raises(ValueError, match="poisoned"):
        gate.event({})


def test_ad_and_solve_charge_shared_and_conditional():
    gate = g.SharedCapGate(caps(), lambda event: None)
    gate.event(dict(kind="jacobian", completed=False, shape=[192, 20], vectorize=True))
    gate.event(dict(kind="bounded_ridge_svd", completed=False))
    gate.event(dict(kind="mode_svd", completed=False))
    gate.event(dict(kind="suffix_jacobian", completed=False, shape=[192, 10], vectorize=True))
    gate.event(dict(kind="suffix_solve", completed=False, svd_calls_requested=1))
    a, b = (gate.attempted[x] for x in g.ARMS)
    assert (a["jacobian"], a["vjp_directions"], a["svd"], a["solve"]) == (1, 192, 2, 1)
    assert (b["jacobian"], b["vjp_directions"], b["svd"], b["solve"]) == (2, 384, 3, 2)


def test_failed_journal_poisoned_and_no_dispatch():
    model, replay, calls = fake_model()
    def broken(event):
        raise OSError("journal failure")
    gate = g.SharedCapGate(caps(), broken)
    with pytest.raises(OSError):
        with g.guarded_entries(model, replay, gate):
            replay._p2_scope = g.frontier.expected_scopes()[0]
            dispatch(model, "action_encoder", tensor("action_encoder"))
    assert gate.failed and not calls


def test_invalid_caps_fail_before_work():
    with pytest.raises(ValueError):
        g.SharedCapGate({"predict": 168}, lambda event: None)


def test_existing_actual_arm_loops_with_synthetic_replay():
    _, args, calls, suffix_calls = legacy_inputs()
    args["bank"] = g.frontier.candidate_bank(base(), g.frontier.ROOTS[0])
    gate = g.SharedCapGate(caps(), lambda event: None)
    results = [g.frontier.run_arm(arm=arm, journal=gate.event, **args) for arm in g.ARMS]
    assert all(r["complete"] for r in results)
    assert [len(r["attempts"]) for r in results] == [80, 40]
    assert len(calls) == 328 and len(suffix_calls) == 80
    assert all(len(r["winner"]["witness"]["receipt"]["actions"]) == 10 for r in results)


def test_native_entry_rejects_cpu_before_creating_output(tmp_path):
    class CPU:
        type = "cpu"
    output = tmp_path / "attempt"
    with pytest.raises(ValueError, match="CUDA only"):
        g.run_native(model=None, device=CPU(), ready={}, compiled={}, provenance_by_arm={},
                     output=output, caps=caps())
    assert not output.exists()
