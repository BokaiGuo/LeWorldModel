import json

import pytest

from experiments.spectral_reachability import p2_joint_host_20261102_codexexclusive as host
from experiments.spectral_reachability import p2_joint_saved_20261102_codexexclusive as saved
from experiments.spectral_reachability import p2_common_objective_ledger_20261102_exclusive as core


def test_timing_failure_is_durable_and_fail_closed(tmp_path):
    ledger = core.Ledger(tmp_path / "events")
    meter = host.TimedMeter(ledger, lambda: None)
    with pytest.raises(RuntimeError):
        with meter.phase("composition_inherited"):
            meter.ledger.call("init_transition", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    ledger.close()
    events = [json.loads(x)["record"] for x in (tmp_path / "events").read_text().splitlines()]
    timing = [x["payload"] for x in events if x["kind"] == "timed_call"]
    assert len(timing) == 1 and timing[0]["status"] == "failed"
    assert timing[0]["elapsed_ns"] >= 0 and meter.failed


def test_transition_token_measurement_and_shape_rejection(tmp_path):
    import numpy as np
    ledger = core.Ledger(tmp_path / "events")
    meter = host.TimedMeter(ledger, lambda: None)
    transition = host.measured_transition(lambda a, b, c: (a, b), meter)
    state, history, action = np.zeros((1, 3, 192)), np.zeros((1, 2, 10)), np.zeros((5, 2))
    with meter.phase("composition_inherited"):
        transition(state, history, action)
        with pytest.raises(ValueError):
            transition(state[:, :2], history, action)
    ledger.close()
    events = [json.loads(x)["record"] for x in (tmp_path / "events").read_text().splitlines()]
    rows = [x["payload"] for x in events if x["kind"] == "transition_shape"]
    assert rows == [{"state_shape": [1, 3, 192], "history_shape": [1, 2, 10],
                     "action_shape": [5, 2], "context_tokens": 3, "successor_tokens": 1}]


def test_consumer_rejects_incomplete_or_forged(tmp_path):
    ledger = core.Ledger(tmp_path / "events")
    ledger.close()
    with pytest.raises(ValueError):
        saved.audit_complete(tmp_path / "events", {"rows": [], "model_selected_winners": {}})


def test_gpu_authorization_fails_before_any_cuda():
    with pytest.raises(ValueError):
        host.require_authorization("wrong", "GPU-wrong")


def test_deadline_rejects_before_dispatch(tmp_path):
    journal = core.Ledger(tmp_path / "deadline")
    meter = host.TimedMeter(journal, lambda: None, deadline=0)
    called = []
    with meter.phase("composition_inherited"), pytest.raises(TimeoutError):
        meter.ledger.call("init_transition", lambda: called.append(True))
    journal.close()
    assert not called and meter.failed


def test_full_saved_schedule_and_forged_winner(tmp_path):
    import copy
    import numpy as np
    import torch
    from experiments.spectral_reachability import p2_joint_vs_conditional_matched_20261102_codexexclusive as m

    fixed = {"context_json": json.dumps({"latent_history": np.zeros((3, 192)).tolist(),
                                         "action_history": np.zeros((2, 10)).tolist()}),
             "goal_json": json.dumps([0.1] * 192)}
    bank = [np.full((10, 2), i / 100, dtype=np.float32).tobytes() for i in range(16)]
    journal = core.Ledger(tmp_path / "full")
    meter = host.TimedMeter(journal, lambda: None)
    def analytic(state, history, action):
        # These are synthetic hooks, not a neural model execution.
        for axis in m.HOOKS:
            meter.call(axis, lambda: None)
        projected = torch.cat((action.reshape(1, 10), action.new_zeros((1, 182))), dim=1)
        return torch.cat((state[:, 1:], (state[:, -1] + projected)[:, None]), dim=1), history
    transition = host.measured_transition(analytic, meter)
    context, goal = json.loads(fixed["context_json"]), json.loads(fixed["goal_json"])
    evaluate = core.objective(transition, context, goal, meter.ledger, "cpu")
    try:
        rows = m.optimize_schedule(
            bank, lambda arm, x, i: m._initialize_with_transition(fixed, x, transition, meter, "cpu", arm, i),
            evaluate, lambda e: meter.ledger.record("optimizer_" + e["kind"], e))
        meter.finish()
    finally:
        journal.close()
    winners = {}
    for arm in m.ARMS:
        score, index, iteration, entry = min(
            ((e["model_score"], r["candidate_index"], e["winner_iteration"], e)
             for r in rows for e in r["arms"] if e["arm"] == arm), key=lambda x: x[:3])
        winners[arm] = {"candidate_index": index, "iteration": iteration, "model_score": score,
                        "raw_float32_hex": entry["raw_float32_hex"]}
    result = {"rows": rows, "model_selected_winners": winners}
    summary = saved.audit_complete(tmp_path / "full", result, bank)
    assert summary["transition_calls"] == 640 and summary["observed_context_tokens"] == 1920
    assert summary["gradient_calls"] == 256 and summary["flops"] is None
    forged = copy.deepcopy(result)
    forged["model_selected_winners"][m.ARMS[0]]["candidate_index"] = 99
    with pytest.raises(ValueError, match="global winner"):
        saved.audit_complete(tmp_path / "full", forged, bank)
