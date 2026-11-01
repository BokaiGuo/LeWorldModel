"""CPU-only online carry selector with per-node replay budgets and JSONL ledger.

Adapters are responsible for producing current-RGB contexts and model-native
candidate banks. This module does not assert that their provenance is correct.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import p1_carry_binding_20261101 as binding
from . import p1_rgb_carry_replan_codex_20261101 as selector


def dls_feedback_candidates(bank, boundary):
    """Apply the existing fixed DLS/Gaussian mix at its supported boundaries."""
    from .e3_cp10_dls_feedback_mix_20261031 import mix_bank
    if boundary == 0:
        return np.asarray(bank).copy()
    return mix_bank(bank, boundary)


def digest(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.tobytes()).hexdigest()


class Meter:
    """Charge attempted calls before dispatch, including calls that fail."""
    def __init__(self, leaf):
        self.leaf = leaf
        self.cost = {k: 0 for k in binding.extra_cost(1, 1, has_carry=False)}

    def _charge(self, cost):
        for k, v in cost.items():
            self.cost[k] += v

    def _dispatch(self, phase, operation, cost, fn):
        self._charge(cost)
        return self.leaf._dispatch(phase, operation, cost, fn)

    def action_encode(self, context, *, phase):
        self._charge({"action_encode_calls": 1})
        return self.leaf.action_encode(context, phase=phase)

    def predict(self, states, actions, *, phase):
        self._charge({"predict_calls": 1, "predict_samples": len(states)})
        return self.leaf.predict(states, actions, phase=phase)


def allocate(remaining, node_limit, *, candidates, horizon, has_carry):
    """Largest fresh prefix fitting both remaining episode and node budgets."""
    if type(candidates) is not int or candidates < 1:
        raise ValueError("at least one fresh candidate required")
    keys = binding.extra_cost(1, horizon, has_carry=has_carry)
    for budget in (remaining, node_limit):
        if set(budget) != set(keys) or any(type(v) is not int or v < 0 for v in budget.values()):
            raise ValueError("budgets must contain nonnegative integer cost dimensions")
    for n in range(candidates, 0, -1):
        cost = binding.extra_cost(n, horizon, has_carry=has_carry)
        if all(cost[k] <= min(remaining[k], node_limit[k]) for k in cost):
            return n, cost
    raise ValueError("insufficient selector budget")


class OnlineExecutor:
    """One episode; `observe`, `candidates`, and `issue` are adapter callbacks.

    observe(boundary) returns rgb, states, history, goal, mean, std and a
    nonempty context_source. candidates(boundary, observation) returns a
    float32 tensor [N,H,5,2] and a nonempty source string. issue(packet)
    executes one packet and returns a JSON-serializable receipt.
    """
    def __init__(self, *, episode, leaf, observe, candidates, issue,
                 budget, node_limit, ledger):
        if not isinstance(episode, str) or not episode:
            raise ValueError("episode required")
        self.episode, self.leaf = episode, leaf
        self.observe, self.candidates, self.issue = observe, candidates, issue
        self.remaining, self.node_limit = dict(budget), dict(node_limit)
        self.ledger = Path(ledger)
        self.boundary, self.previous, self.issued = 0, None, None
        self.stopped = False

    def record(self, event, **payload):
        with self.ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": event, "episode": self.episode,
                                "boundary": self.boundary, **payload},
                               allow_nan=False, sort_keys=True) + "\n")
            f.flush()

    def step(self):
        if self.stopped:
            raise RuntimeError("executor stopped; reconcile before any retry")
        meter = Meter(self.leaf)
        reserved = None
        try:
            obs = self.observe(self.boundary)
            rgb = np.asarray(obs["rgb"])
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or not rgb.size:
                raise ValueError("RGB must be nonempty HxWx3 uint8")
            if not isinstance(obs["context_source"], str) or not obs["context_source"]:
                raise ValueError("context source required")
            rgb = np.ascontiguousarray(rgb).copy()
            rgb_hash = digest(rgb)
            self.record("observation", rgb=rgb.tolist(), rgb_shape=list(rgb.shape),
                        rgb_sha256=rgb_hash, context_source=obs["context_source"])
            feedback = None
            if self.previous is not None:
                feedback = self.previous.bind_feedback(
                    episode=self.episode, boundary=self.boundary,
                    issued_packet=self.issued, rgb_sha256=rgb_hash)
            bank, source = self.candidates(self.boundary, obs)
            if not isinstance(source, str) or not source:
                raise ValueError("candidate source required")
            if not isinstance(bank, torch.Tensor) or bank.device.type != "cpu":
                raise ValueError("CPU tensor candidates required")
            for key in ("states", "history", "goal", "mean", "std"):
                if not isinstance(obs[key], torch.Tensor) or obs[key].device.type != "cpu":
                    raise ValueError("CPU context tensors required")
            self.record("context", context_source=obs["context_source"],
                        tensors={key: obs[key].detach().tolist() for key in
                                 ("states", "history", "goal", "mean", "std")},
                        provenance_verified=False)
            carry = feedback is not None and len(feedback["tail"]) > 0
            horizon = len(feedback["tail"]) if carry else bank.shape[1]
            n, reserved = allocate(self.remaining, self.node_limit,
                                   candidates=bank.shape[0], horizon=horizon, has_carry=carry)
            self.record("reservation", reserved=reserved, fresh_count=n,
                        candidate_source=source, candidate_bank=bank.detach().tolist(),
                        candidate_bank_sha256=digest(bank.detach().numpy()),
                        previous_plan_sha256=feedback["previous_plan_sha256"] if feedback else None,
                        issued_packet_sha256=feedback["issued_packet_sha256"] if feedback else None)
            prev = torch.from_numpy(self.previous.plan.copy()) if carry else None
            result = selector.select(meter, **{k: obs[k] for k in
                ("states", "history", "goal", "mean", "std")},
                new_candidates=bank[:n], previous_selected=prev)
            if meter.cost != reserved or result["incremental_cost"] != reserved:
                raise RuntimeError("selector settlement mismatch")
            plan = result["selected_raw"].detach().numpy().copy()
            sealed = binding.BoundPlan.seal(self.episode, self.boundary, plan)
            self.record("selection", selected_source=result["selected_source"],
                        selected_index=result["selected_index"], scores=result["scores"],
                        plan=plan.tolist(), plan_sha256=sealed.plan_sha256,
                        rgb_sha256=rgb_hash, horizon_packets=horizon)
            packet = plan[0].copy()
            self.record("issue_intent", packet=packet.tolist(), packet_sha256=digest(packet),
                        plan_sha256=sealed.plan_sha256)
            receipt = self.issue(packet.copy())
            self.record("issue_receipt", receipt=receipt, packet_sha256=digest(packet))
            self.previous, self.issued = sealed, packet
            return result
        except Exception as exc:
            self.stopped = True
            self.record("failure", error_type=type(exc).__name__, error=str(exc),
                        retry_safe=False)
            raise
        finally:
            for k in self.remaining:
                self.remaining[k] -= meter.cost[k]
            self.record("settlement", reserved=reserved, actual=meter.cost,
                        remaining=self.remaining, scope="selector_replay_only")
            if not self.stopped:
                self.boundary += 5


class DemoLeaf:
    def _dispatch(self, phase, operation, cost, fn):
        return fn()

    def action_encode(self, context, *, phase):
        return context

    def predict(self, states, actions, *, phase):
        out = states.clone()
        out[:, -1, :] += actions[:, -1, :].mean(dim=-1, keepdim=True)
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, help="new JSONL output path")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if args.steps < 1 or Path(args.ledger).exists():
        parser.error("steps must be positive and ledger must not already exist")
    def observe(boundary):
        return dict(rgb=np.full((2, 2, 3), boundary, dtype=np.uint8),
                    states=torch.zeros(1, 2, 2), history=torch.zeros(1, 1, 10),
                    goal=torch.ones(1, 2), mean=torch.zeros(2), std=torch.ones(2),
                    context_source="synthetic_demo_not_rgb_encoder")
    def candidates(boundary, obs):
        return torch.full((2, 3, 5, 2), .1, dtype=torch.float32), "synthetic_demo"
    cap = binding.extra_cost(3, 3, has_carry=True)
    executor = OnlineExecutor(episode="demo", leaf=DemoLeaf(), observe=observe,
        candidates=candidates, issue=lambda packet: {"simulated": True},
        budget={k: v * args.steps for k, v in cap.items()}, node_limit=cap,
        ledger=args.ledger)
    for _ in range(args.steps):
        executor.step()
    print(json.dumps({"status": "simulated", "ledger": args.ledger}))


if __name__ == "__main__":
    main()
