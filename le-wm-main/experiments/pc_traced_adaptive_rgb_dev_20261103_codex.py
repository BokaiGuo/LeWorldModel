"""Adaptive same-bank RGB pilot with every real model prediction traced."""
import hashlib
from pathlib import Path

import numpy as np

from experiments import pc_fixed_terminal_fair_feedback_dev_20261103_codex as runner
from experiments import pc_multi_nominal_rgb_dev_20261103_codex as policy
from experiments.spectral_reachability.pc_spectral_bank_planner_20261103_codex import Meter

PROTOCOL = policy.ROOT.parent / "docs/PC_TRACED_ADAPTIVE_RGB_DEV_PROTOCOL_20261103_CODEX.md"
PREFIX = policy.ROOT / "results/pc_traced_adaptive_rgb_dev_20261103_codex"
ARMS = ("uniform", "adaptive", "greedy", "cem")


class SuffixState:
    def __init__(self, prefix, horizon, seed):
        self.prefix = np.asarray(prefix, dtype=float)
        self.rng = np.random.default_rng(seed)
        self.loc = np.zeros((horizon - len(prefix), 2))
        self.scale = np.full(self.loc.shape, 1 / 3)
        self.best, self.score, self.count = None, float("inf"), 0

    def step(self, size, predict):
        if not 1 <= size <= 12:
            raise ValueError("invalid suffix batch")
        samples = np.clip(self.loc + self.rng.normal(size=(size, *self.loc.shape)) * self.scale, -1, 1)
        samples[0] = self.loc
        chains = np.concatenate((np.broadcast_to(self.prefix, (size, *self.prefix.shape)), samples), axis=1)
        scores = np.asarray(predict(chains)[1])
        if scores.shape != (size,) or not np.isfinite(scores).all():
            raise ValueError("invalid suffix scores")
        idx = int(np.argmin(scores))
        if scores[idx] < self.score:
            self.best, self.score = chains[idx].copy(), float(scores[idx])
        if size > 1:
            elite = samples[np.argsort(scores, kind="stable")[:min(4, size)]]
            self.loc = elite.mean(0)
            self.scale = np.maximum(elite.std(0), 1e-5)
        self.count += size


def adaptive(root, step, target, predict, token_meter, carry):
    horizon = 15 - step * 5
    if horizon == 5:
        return policy.decide("svd", root, step, target, predict, token_meter, carry)
    if (carry is None) != (step == 0):
        raise ValueError("missing or unexpected carry")
    carry_score = None
    if carry is not None:
        carry = np.asarray(carry, dtype=float)
        if carry.shape != (horizon, 2):
            raise ValueError("carry horizon mismatch")
        carry_score = float(predict(carry[None])[1][0])
    seed = 2026113000 + 10 * root + step
    structure = Meter({"prediction_samples": 42, "svd": 2, "linear_solve": 2})
    bank, evidence = policy.generator.generate(lambda x: predict(np.asarray(x)[None])[0][0],
                                                target, policy.NOMINALS, structure, mode="svd")
    prefix_scores = [{"name": x["name"], "score": float(predict(np.asarray(x["prefix"])[None])[1][0])}
                     for x in bank]
    remaining = (policy.CAP - token_meter.token_samples) // (horizon // 5)
    suffix_total = 3 * remaining // 4
    if suffix_total < 12 * len(bank):
        raise ValueError("insufficient adaptive budget")
    states = [SuffixState(x["prefix"], horizon, seed) for x in bank]
    allocation = []
    for index in range(len(bank)):
        states[index].step(12, predict)
        allocation.append({"candidate": index, "samples": 12})
    remaining = suffix_total - 12 * len(bank)
    while remaining:
        idx = min(range(len(states)), key=lambda i: (states[i].score, i))
        count = min(12, remaining)
        states[idx].step(count, predict)
        allocation.append({"candidate": idx, "samples": count})
        remaining -= count
    programs = [{"name": item["name"], "chain": state.best.tolist(),
                 "terminal_score": state.score, "samples": state.count}
                for item, state in zip(bank, states, strict=True)]
    selected = min(programs, key=lambda x: x["terminal_score"])
    count = (policy.CAP - token_meter.token_samples) // (horizon // 5)
    chain, score = policy.refine(selected["chain"], selected["terminal_score"], count,
                                 np.random.default_rng(seed + 1), lambda x: predict(x)[1])
    if score < selected["terminal_score"]:
        selected = {"name": "joint_refined", "chain": chain.tolist(),
                    "terminal_score": score, "refined_from": selected["name"], "samples": count}
    if carry is not None and carry_score < selected["terminal_score"]:
        selected = {"name": "carry", "chain": carry.tolist(), "terminal_score": carry_score}
    return {"horizon": horizon, "programs": programs, "prefix_scores": prefix_scores,
            "generator": evidence, "allocation": allocation, "selected": selected,
            "selection_source": selected["name"], "abstained_reconfiguration": selected["name"] == "zero",
            "carry": None if carry is None else carry.tolist(), "carry_score": carry_score,
            "spectral_counts": structure.counts, "spectral_seconds": structure.seconds}


def decide(arm, root, step, target, predict, token_meter, carry=None):
    if arm not in ARMS:
        raise ValueError("invalid arm")
    trace = []

    def traced(actions):
        positions, scores = predict(actions)
        actions = np.asarray(actions)
        trace.append({"requested_actions": actions.tolist(), "model_actions": actions.astype(np.float32).tolist(),
                      "positions": np.asarray(positions).tolist(),
                      "scores": np.asarray(scores).tolist(),
                      "tokens": len(actions) * actions.shape[1] // 5})
        return positions, scores
    if arm == "adaptive":
        result = adaptive(root, step, target, traced, token_meter, carry)
    else:
        result = policy.decide("svd" if arm == "uniform" else arm,
                               root, step, target, traced, token_meter, carry)
    result["prediction_trace"] = trace
    result["adaptive_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["base_policy_sha256"] = hashlib.sha256(Path(policy.__file__).read_bytes()).hexdigest()
    return result


def main():
    runner.PROTOCOL, runner.PREFIX, runner.ARMS, runner.decide = PROTOCOL, PREFIX, ARMS, decide
    runner.main()


if __name__ == "__main__":
    main()
