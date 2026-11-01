import numpy as np
import pytest
import torch
import hashlib
import json
from pathlib import Path

from experiments.spectral_reachability import p1_carry_binding_20261101 as binding
from experiments.spectral_reachability import p1_rgb_carry_replan_codex_20261101 as selector
from experiments.spectral_reachability import codex_p1_cp10_greedy_carry_20261101_exclusive as greedy
from experiments.spectral_reachability import codex_four_arm_consumer_20261030 as consumer


class Leaf:
    def __init__(self):
        self.cost = {}

    def _dispatch(self, phase, operation, cost, fn):
        for key, value in cost.items():
            self.cost[key] = self.cost.get(key, 0) + value
        return fn()


def test_candidate_tie_cost_and_old_plan_binding(monkeypatch):
    def prediction(leaf, phase, states, history, plan, mean, std):
        leaf._dispatch(phase, "model", {"predict_calls": 1, "predict_samples": 1,
                                      "action_encode_calls": 1}, lambda: None)
        return plan[:, 0, 0, :1]

    monkeypatch.setattr(selector.raw_model, "_normal", prediction)
    bank = torch.zeros((8, 3, 5, 2))
    bank[:, 0, 0, 0] = torch.tensor([0.8, 0.6, 0.3, 0.1, 0.1, 0.5, 0.4, 0.7])
    bank[:, 1:, :, :] = torch.arange(8)[:, None, None, None] / 10
    leaf = Leaf()
    selected = selector.select(leaf, states=torch.zeros((1, 3, 1)), history=torch.zeros((1, 2, 10)),
                               goal=torch.zeros((1, 1)), mean=torch.zeros(2), std=torch.ones(2),
                               new_candidates=bank[:, :1])
    assert selected["selected_index"] == 3
    assert selected["incremental_cost"] == binding.extra_cost(8, 1, has_carry=False)
    assert leaf.cost == {k: v for k, v in selected["incremental_cost"].items() if v}
    plan = bank[selected["selected_index"]].numpy().astype(np.float32)
    sealed = binding.BoundPlan.seal("episode", 0, plan)
    feedback = sealed.bind_feedback(episode="episode", boundary=5, issued_packet=plan[0],
                                    rgb_sha256="a" * 64)
    np.testing.assert_array_equal(feedback["tail"], plan[1:])
    with pytest.raises(ValueError):
        sealed.bind_feedback(episode="episode", boundary=5, issued_packet=np.zeros((5, 2), np.float32),
                             rgb_sha256="a" * 64)


def test_protocol_pins_rule_and_193_cap():
    host = greedy.configure()
    protocol = host.protocol()
    assert protocol["prefix_rule"].startswith("argmin_model_mse")
    assert protocol["prefix_extra_cost"]["predict_calls"] == 8
    assert protocol["common_node_caps"]["replay_calls"] == 193
    assert protocol["independent_confirmation"] is False


def test_recorded_bank_actions_feedback_and_ledger():
    root = Path(__file__).resolve().parents[1] / "results"
    run = root / "codex_p1_cp10_greedy_carry_20261101_exclusive_RUN_CODEXGREEDY01"
    if not run.exists():
        pytest.skip("optional host run absent")
    source = root / "codex_p1_common_structured_cp10_host_20261101_exclusive_RUN_CODEX01"
    report = json.loads((run / "report.json").read_bytes())
    original = json.loads((source / "report.json").read_bytes())
    assert report["cp10_goal_binding"] == original["cp10_goal_binding"]
    assert report["model_before"] == report["model_after"] == original["model_before"]
    row = report["rows"][0]
    assert report["host"]["device"] == "cuda:0" and row["initial_exact"]
    assert row["status"] == "native_stopped_control" and row["control_returned"] == 16
    trace = lambda p: [json.loads(line) for line in p.read_text().splitlines()]
    events = trace(run / "common_structured_b00/candidates.jsonl")
    old = trace(source / "common_structured_b00/candidates.jsonl")
    event = lambda rows, kind: next(r["payload"] for r in rows if r["kind"] == kind)
    bank = event(events, "optimized_candidates")["raw"]
    assert bank == event(old, "optimized_candidates")["raw"]
    selection = event(events, "same_bank_one_packet_greedy")
    assert selection["index"] == min(range(8), key=lambda i: (selection["scores"][i], i)) == 4
    assert selection["selected_full_h3"] == bank[4]
    assert row["executed_raw"][:5] == bank[4][0]
    assert selection["original_h3_index"] == 0
    checkpoints = {p["step"]: p for p in row["checkpoints"]}
    previous = None
    for node in report["nodes"]["s0_structured"]:
        boundary = node["boundary"]
        records = trace(run / f"common_structured_b{boundary:02}/candidates.jsonl")
        selected = event(records, "selected_full_plan")
        sealed = binding.BoundPlan.seal("s0_structured", boundary,
                                        np.asarray(selected["raw"], np.float32))
        assert sealed.plan_sha256 == selected["plan_sha256"]
        seal = json.loads((run / f"s0_structured_b{boundary:02}_seal.json").read_bytes())
        assert seal["identity"]["observation"]["pixel_sha256"] == checkpoints[boundary]["pixel_sha256"]
        if boundary:
            feedback = event(records, "first_feedback_selection")["feedback"]
            actual = previous.bind_feedback(episode="s0_structured", boundary=boundary,
                                            issued_packet=np.asarray(row["executed_raw"][boundary-5:boundary], np.float32),
                                            rgb_sha256=checkpoints[boundary]["pixel_sha256"])
            assert all(feedback[k] == actual[k] for k in feedback)
        previous = sealed
        receipt = node["receipt"]
        account, ledger = receipt["account"], receipt["ledger"]
        assert hashlib.sha256(Path(receipt["trace"]).read_bytes()).hexdigest() == receipt["trace_sha256"]
        assert consumer.consume(ledger["journal"], ledger["sha256"], account["contract_sha256"],
                                "structured", "node", account["run_id"]) == account
    assert report["nodes"]["s0_structured"][0]["receipt"]["account"]["charged_upper_bounds"]["replay_calls"] == 16
