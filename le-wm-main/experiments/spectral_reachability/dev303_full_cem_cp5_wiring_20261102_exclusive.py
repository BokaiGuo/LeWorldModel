"""Development-only H3 -> actual cp5 feedback -> H2 CEM integration seam.

No environment, model, GPU, experiment outcome, or output file is opened here
on import. A separate trusted host must authenticate all observation/action
receipts, load the pinned model and persist the events before physical use.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

NAME = "dev303_full_cem_cp5_wiring_20261102_exclusive"
ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "results/task_p1_dev303_goal_blind_20261101_codex.json.gz"
TASK_SHA = "3787aba5dfe49c97b9c738e347e228c8089449af0df84a5f91788ceb769b2b12"
BLIND_SHA = "092af72253c9b0463e5c7a5814052599843dc4b6a979bd040d7f318c93dcddd9"
ITERATIONS, SAMPLES, ELITES = 4, 16, 4


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_task():
    raw = TASK.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == TASK_SHA, "task SHA")
    export = json.loads(gzip.decompress(raw))
    require(export["development_seed"] == 301000003
            and type(export["development_seed"]) is int
            and export["blind_source_sha256"] == BLIND_SHA
            and export["P1_admission"] is False
            and export["new_independent_confirmation"] is False
            and export["global_freshness"] == "unknown", "task authority")
    return export["task"]


def budget(horizon):
    require(type(horizon) is int and horizon in (2, 3), "horizon")
    calls = (ITERATIONS * SAMPLES + 1) * horizon
    return {"candidate_scores": ITERATIONS * SAMPLES, "winner_replays": 1,
            "action_encoder_calls": calls, "predict_calls": calls,
            "predict_samples": calls}


def contract():
    return {"schema": NAME + ".v1", "task_sha256": TASK_SHA,
            "development_seed": 301000003, "steps": [0, 5],
            "feedback_steps": [5], "horizons_packets": [3, 2],
            "packet_steps": 5, "control_steps_max": 15,
            "node_budgets": [budget(3), budget(2)],
            "budget_scope": "candidate prediction and replay only; encoder, host, timing and shared work excluded",
            "physical_execution_authorized": False, "cuda_execution_authorized": False,
            "fairness_proven": False, "P1_admission": False,
            "independent_confirmation": False, "official_reproduction": False}


def optimize(horizon, score, *, seed, emit=lambda event: None):
    require(type(seed) is int and 0 <= seed < 2**63, "seed")
    budget(horizon)
    rng = np.random.default_rng(seed)
    center = np.zeros((horizon, 5, 2), np.float32)
    scale = np.full_like(center, 0.5)
    candidates, values = [], []
    for iteration in range(ITERATIONS):
        bank = np.clip(rng.normal(center, scale, (SAMPLES, horizon, 5, 2)),
                       -1, 1).astype(np.float32)
        bank[0] = center
        emit({"type": "bank", "iteration": iteration, "raw": bank.tolist()})
        current = []
        for candidate in bank:
            emit({"type": "score_attempt", "index": len(values)})
            value = float(score(candidate.copy()))
            require(np.isfinite(value), "nonfinite score")
            current.append(value)
            candidates.append(candidate.copy())
            values.append(value)
            emit({"type": "score_return", "index": len(values)-1, "value": value})
        elites = sorted(range(SAMPLES), key=lambda i: (current[i], i))[:ELITES]
        center = bank[elites].mean(axis=0, dtype=np.float32)
        scale = np.clip(bank[elites].std(axis=0, dtype=np.float32), .05, 1.)
        emit({"type": "update", "iteration": iteration, "elite_indices": elites,
              "mean": center.tolist(), "std": scale.tolist()})
    winner = min(range(len(values)), key=lambda i: (values[i], i))
    chosen = candidates[winner]
    emit({"type": "selection", "index": winner, "score": values[winner],
          "raw": chosen.tolist()})
    emit({"type": "replay_attempt", "index": winner})
    repeat = float(score(chosen.copy()))
    require(np.isfinite(repeat) and np.isclose(repeat, values[winner], rtol=1e-6, atol=1e-8),
            "winner replay mismatch")
    emit({"type": "replay_return", "index": winner, "score": repeat})
    return {"selected_index": winner, "score": values[winner], "actions": chosen.copy()}


class TorchScorer:
    """Use an injected real LeafAdapter/model, with attempted/returned counters."""
    def __init__(self, leaf, context):
        import torch
        self.torch, self.leaf = torch, leaf
        self.context = context
        self.attempted = {"action_encode": 0, "predict": 0}
        self.returned = dict(self.attempted)
        states, history, goal, mean, std, matrix = (context[k] for k in
            ("states", "history", "goal", "mean", "std", "matrix"))
        tensors = (states, history, goal, mean, std, matrix)
        require(all(isinstance(x, torch.Tensor) and x.is_floating_point()
                    and x.device == states.device and bool(torch.isfinite(x).all()) for x in tensors),
                "finite device-aligned tensors")
        d = states.shape[-1] if states.ndim == 3 else 0
        require(d > 0 and states.shape == (1, 3, d) and history.shape == (1, 2, 10)
                and goal.shape == (1, d) and mean.shape == std.shape == (2,)
                and matrix.shape == (d, d) and bool((std > 0).all()), "context shape")

    def __call__(self, actions):
        t, c = self.torch, self.context
        a = np.asarray(actions)
        require(a.ndim == 3 and a.shape[0] in (2, 3) and a.shape[1:] == (5, 2)
                and np.isfinite(a).all() and np.all(np.abs(a) <= 1), "raw actions")
        states, history = c["states"], c["history"]
        with t.no_grad():
            for packet in t.as_tensor(a, dtype=states.dtype, device=states.device):
                token = ((packet - c["mean"]) / c["std"]).reshape(1, 1, 10)
                action_context = t.cat((history, token), dim=1)
                self.attempted["action_encode"] += 1
                encoded = self.leaf.action_encode(action_context, phase="verification")
                self.returned["action_encode"] += 1
                self.attempted["predict"] += 1
                pred = self.leaf.predict(states, encoded, phase="verification")
                self.returned["predict"] += 1
                require(isinstance(pred, t.Tensor) and pred.shape == states.shape
                        and bool(t.isfinite(pred).all()), "prediction")
                states = t.cat((states, pred[:, -1:]), dim=1)[:, -3:]
                history = action_context[:, -2:]
            residual = (states[:, -1] - c["goal"]).double()
            value = (residual @ c["matrix"].double().T).square().mean()
            require(bool(t.isfinite(value)), "score")
            return float(value.item())


def run_callbacks(observe, score, issue, *, seed, emit=lambda event: None):
    """Execute a causal schedule through host-supplied callbacks.

    The host, not this seam, is responsible for real RGB/action provenance,
    durable logging, resource caps, frozen source binding and runtime authority.
    """
    require(type(seed) is int and 0 <= seed < 2**63 - 1, "seed")
    actual = np.empty((0, 2), dtype=np.float32)
    nodes = []
    for index, (step, horizon, execute_steps) in enumerate(((0, 3, 5), (5, 2, 10))):
        emit({"type": "observation_attempt", "step": step})
        obs = observe(step, actual.copy())
        require(type(obs) is dict and obs.get("step") == step
                and isinstance(obs.get("rgb_sha256"), str) and len(obs["rgb_sha256"]) == 64
                and all(c in "0123456789abcdef" for c in obs["rgb_sha256"])
                and np.array_equal(np.asarray(obs.get("actual_actions")), actual),
                "actual feedback receipt")
        emit({"type": "observation_return", "step": step, "rgb_sha256": obs["rgb_sha256"]})
        result = optimize(horizon, lambda plan: score(obs, plan), seed=seed + index, emit=emit)
        command = result["actions"].reshape(-1, 2)[:execute_steps].copy()
        emit({"type": "dispatch_attempt", "step": step, "actions": command.tolist()})
        receipt = issue(step, command.copy())
        require(type(receipt) is dict and type(receipt.get("stopped")) is bool, "issue receipt")
        returned = np.asarray(receipt.get("executed_actions"))
        require(returned.ndim == 2 and returned.shape[1:] == (2,) and len(returned) <= execute_steps
                and np.isfinite(returned).all()
                and np.array_equal(returned, command[:len(returned)])
                and (len(returned) == execute_steps or receipt["stopped"]), "actual dispatched actions")
        actual = np.concatenate((actual, returned.astype(np.float32)), axis=0)
        emit({"type": "dispatch_return", "step": step, "count": len(returned),
              "stopped": receipt["stopped"]})
        nodes.append({"step": step, "horizon": horizon, "selected_index": result["selected_index"],
                      "planned_budget": budget(horizon), "executed_steps": len(returned)})
        if receipt["stopped"]:
            break
    return {"nodes": nodes, "actual_actions": actual, "fairness_proven": False,
            "physical_execution_authorized": False, "P1_admission": False,
            "receipt_authentication": "external_host_required"}


def run_model_callbacks(leaf, observe, issue, *, seed, emit):
    """Connect the CEM schedule to the real leaf interface; host must persist events."""
    require(callable(emit), "event sink required")
    active = None
    usage = []

    def account():
        nonlocal active
        if active is None:
            return
        step, scorer = active
        row = {"type": "node_model_usage", "step": step,
               "attempted": dict(scorer.attempted), "returned": dict(scorer.returned)}
        emit(row)
        usage.append(row)
        active = None

    def score(obs, plan):
        nonlocal active
        if active is None:
            active = (obs["step"], TorchScorer(leaf, obs["context"]))
        require(active[0] == obs["step"], "node changed before accounting")
        return active[1](plan)

    def checked_issue(step, commands):
        require(active is not None and active[0] == step, "missing node scorer")
        expected = budget(3 if step == 0 else 2)["predict_calls"]
        scorer = active[1]
        require(scorer.attempted == scorer.returned ==
                {"action_encode": expected, "predict": expected}, "model call accounting")
        account()
        return issue(step, commands)

    try:
        result = run_callbacks(observe, score, checked_issue, seed=seed, emit=emit)
    finally:
        account()
    result["model_usage"] = usage
    return result
