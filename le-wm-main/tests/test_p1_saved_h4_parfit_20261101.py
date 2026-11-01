"""Synthetic saved artifacts; no GPU, model or physical evidence."""
import hashlib
import json

import numpy as np
import pytest

from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from experiments.spectral_reachability import p1_carry_binding_20261101 as binding
from experiments.spectral_reachability import p1_h4_parfit_20261101 as h4
from experiments.spectral_reachability import p1_saved_h4_parfit_20261101 as saved


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    return saved.sha(path)


def write_chain(path, values, *, physical=False):
    previous = None
    rows = []
    for index, (kind, payload) in enumerate(values):
        if physical:
            body = {"index": index, "kind": kind, "previous_sha256": previous, **payload}
            digest = hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()
            event = {**body, "event_sha256": digest}
        else:
            body = {"index": index, "kind": kind, "previous": previous, "payload": payload}
            digest = spec.digest(body)
            event = {**body, "sha256": digest}
        rows.append(event)
        previous = digest
    path.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in rows))
    return rows


def fixture(folder, monkeypatch, tamper=None):
    arm, episode = "random", "s0_random"
    source = folder / "source"
    source.write_text("fixture only")
    material = {"schema": "p1_reverse_h4_parfit_20261101.prepare.v1",
                "protocol": {"fixture": True}, "sources": {str(source): saved.sha(source)}}
    prepared = folder / "prepared.json"
    prepared_sha = write_json(prepared, material)
    nodes, previous, accounts = [], None, {}
    for boundary in (0, 5, 10, 15):
        horizon, carry = h4.dimensions(boundary)
        rgb = str(boundary // 5 + 1) * 64
        cost = h4.node_cost(arm, boundary)
        config = {"decision_boundary": boundary, "allocated_horizon_packets": 4,
                  "planning_horizon_packets": horizon}
        policy = {"horizon": horizon}
        contract = {"identity": {"optimizer_config": spec.digest(config)},
                    "phase_policy_sha256": spec.digest(policy)}
        contract_sha = spec.digest(contract)
        ledger_path = folder / f"ledger_{boundary}.jsonl"
        ledger_path.write_text(json.dumps({"payload": {"contract": contract}}) + "\n")
        ledger = {"journal": str(ledger_path), "sha256": saved.sha(ledger_path)}
        account = {"status": "closed_unverified", "charged_upper_bounds": cost}
        accounts[str(ledger_path)] = account
        zero_plan = np.zeros((horizon, 5, 2), np.float32)
        plan = binding.BoundPlan.seal(episode, boundary, zero_plan)
        feedback = None
        if carry:
            linked = previous.bind_feedback(episode=episode, boundary=boundary,
                                            issued_packet=np.zeros((5, 2), np.float32), rgb_sha256=rgb)
            feedback = {k: v for k, v in linked.items() if k != "tail"}
            if tamper == "tail_binding" and boundary == 5:
                feedback["previous_plan_sha256"] = "f" * 64
        bank = np.zeros((8, 4, 5, 2), np.float32)
        idx = 8 if carry else 0
        node_events = [
            ("h4_contract_binding", {"boundary": boundary, "config": config, "policy": policy,
                                     "config_sha256": spec.digest(config), "policy_sha256": spec.digest(policy),
                                     "contract_sha256": contract_sha, "projection": "identity_192_exact"}),
            ("optimized_candidates_before_exchange", {"raw": bank.tolist()}),
        ]
        if horizon > 1:
            node_events.append(("suffix_exchange", {"scores": [0.] * 64, "tail_indices": list(range(8))}))
        node_events.extend([
            ("selected_full_plan", {"raw": zero_plan.tolist(), "index": idx,
                                    "scores": [0.] * (9 if carry else 8), "charges": cost, "horizon": horizon}),
            ("h4_feedback_binding", {"boundary": boundary, "horizon": horizon,
                                     "current_rgb_sha256": rgb, "feedback": feedback}),
        ])
        trace = folder / f"trace_{boundary}.jsonl"
        write_chain(trace, node_events)
        hook_cost = dict(cost)
        if tamper == "hook_cost" and boundary == 10:
            hook_cost["predict_calls"] += 1
        receipt = {"status": "complete", "trace": str(trace), "trace_sha256": saved.sha(trace),
                   "contract_sha256": contract_sha, "ledger": ledger, "account": account,
                   "module_hook_audit": {"ledger_charged_upper_bounds": hook_cost,
                       "module_forward_pre_hooks": {"predict_calls": cost["predict_calls"],
                                                    "action_encode_calls": cost["action_encode_calls"]},
                       "autograd_api_counts": {"grad_attempted": cost["backward_calls"],
                                               "grad_completed": cost["backward_calls"]}},
                   "selected_full_plan": zero_plan.tolist(), "selected_full_plan_sha256": plan.plan_sha256,
                   "selected_index": idx, "commands": zero_plan[0].tolist(),
                   "scores": [0.] * (9 if carry else 8)}
        nodes.append({"boundary": boundary, "status": "sealed", "receipt": receipt})
        previous = plan
    records = []
    for i in range(20):
        if i % 5 == 0:
            receipt = nodes[i // 5]["receipt"]
            digest = spec.digest(receipt)
            records.append(("boundary_sealed", {"episode_id": episode, "boundary": i,
                                                 "plan_sha256": digest}))
        command = [0., 0.]
        if tamper == "returned_command" and i == 7:
            command = [.1, 0.]
        records.extend([("authorized_dispatch", {"episode_id": episode, "step": i + 1, "command": command}),
                        ("control_return", {"episode_id": episode, "step": i + 1, "command": command})])
    physical = write_chain(folder / "journal.jsonl", records, physical=True)
    row = {"arm": arm, "episode_id": episode, "control_returned": 20, "full_horizon": True,
           "executed_raw": np.zeros((20, 2)).tolist(),
           "checkpoints": [{"step": b, "pixel_sha256": str(b // 5 + 1) * 64} for b in (0, 5, 10, 15)]}
    report = {"status": "recorded_known_root_not_admitted", "P1_admission": False, "rows": [row],
              "protocol": material["protocol"], "prepared": str(prepared), "prepare_sha256": prepared_sha,
              "journal_sha256": saved.sha(folder / "journal.jsonl"),
              "journal_chain_tail": physical[-1]["event_sha256"], "nodes": {episode: nodes}}
    report_sha = write_json(folder / "report.json", report)
    monkeypatch.setattr(saved.consumer, "consume", lambda path, *_: accounts[path])
    if tamper == "source":
        source.write_text("changed")
    return report_sha


def test_saved_fixture_and_historical_physical_chain(tmp_path, monkeypatch):
    report_sha = fixture(tmp_path, monkeypatch)
    result = saved.check(tmp_path, report_sha)
    assert [r["horizon"] for r in result["nodes"]] == [4, 3, 2, 1]
    assert [r["source"] for r in result["nodes"]] == ["fresh", "carry", "carry", "carry"]
    assert result["P1_admission"] is False


@pytest.mark.parametrize("tamper", ["returned_command", "tail_binding", "hook_cost", "source"])
def test_rejects_tampered_fixture(tmp_path, monkeypatch, tamper):
    report_sha = fixture(tmp_path, monkeypatch, tamper)
    with pytest.raises(ValueError):
        saved.check(tmp_path, report_sha)


@pytest.mark.parametrize("mode,expected", [
    ("full", "full20_eligible_for_strict_check"),
    ("terminated", "native_early_termination_or_truncation"),
    ("truncated", "native_early_termination_or_truncation"),
    ("error", "partial_or_error_requires_inspection")])
def test_report_only_branch_classification(tmp_path, mode, expected):
    row = {"arm": "random", "episode_id": "s0_random",
           "control_returned": 20 if mode == "full" else 7,
           "full_horizon": mode == "full", "terminated": mode == "terminated",
           "truncated": mode == "truncated", "native_success": mode == "terminated"}
    report = {"rows": [row], "nodes": {"s0_random": [{"boundary": 0, "status": "sealed"}]},
              "status": "fixture", "errors": []}
    sha = write_json(tmp_path / "report.json", report)
    result = saved.status_only(tmp_path, sha)
    assert result["classification"] == expected
    assert result["status"] == "report_only_not_integrity_verified"
