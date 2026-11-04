"""Predeclared, token-metered comparison arms for sealed successor wall runs."""
import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments import pc_rgb_successor_exact_same_policy_dev_20261103_codex as prior
from experiments.pc_budget_allocation_20261103_codex import allocate_samples, optimize
from experiments.spectral_reachability.pc_spectral_bank_planner_20261103_codex import Meter

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_SUCCESSOR_EXACT_CONTROLS_DEV_PROTOCOL_20261103_CODEX.md"
PREFIX = ROOT / "results/pc_rgb_successor_exact_controls_dev_20261103_codex"
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
ARMS = ("same_bank_greedy", "random_bridge", "lookahead", "full_cem")


def dependencies():
    paths = (PROTOCOL, Path(__file__), Path(prior.__file__), Path(prior.policy.__file__),
             Path(prior.policy.policy.__file__), Path(prior.policy.policy.generator.__file__),
             Path(prior.exact.__file__), Path(prior.exact.wall.__file__),
             Path(prior.exact.wall.physics.__file__), Path(optimize.__code__.co_filename),
             prior.PLANS, prior.AUDIT)
    return {str(p): prior.sha(p) for p in paths}


def decide(task, root, arm, expected_bank):
    start, meter, trace = time.monotonic(), prior.Budget(), []

    def predict(actions):
        requested = np.asarray(actions, dtype=float)
        pos, scores = meter.predict(task, requested)
        trace.append({"requested": requested.tolist(), "model": requested.astype(np.float32).tolist(),
                      "positions": pos.tolist(), "scores": scores.tolist(),
                      "tokens": len(requested) * requested.shape[1] // 5})
        return pos, scores

    seed = 2026121300 + root
    bank, first, scores, programs = [], [], [], []
    structure = Meter({"prediction_samples": 21, "svd": 1, "linear_solve": 0})
    if arm == "full_cem":
        count = (480 - 3 - meter.token_samples) // 3
        chain, score, _ = optimize([], 15, count, np.random.default_rng(seed), lambda a: predict(a)[1])
        selected = {"name": "full_cem", "chain": chain.tolist(), "terminal_score": score}
    else:
        first, _ = prior.policy.policy.generator.generate(
            lambda a: predict(np.asarray(a)[None])[0][0], np.asarray(task["goal"]),
            prior.policy.NOMINALS, structure, mode="svd")
        if first != expected_bank:
            raise ValueError("regenerated first bank differs from sealed successor run")
        if arm == "same_bank_greedy":
            for item in first:
                scores.append(float(predict(np.asarray(item["prefix"])[None])[1][0]))
            selected_first = first[int(np.argmin(scores))]
            bank = [{"name": selected_first["name"], "prefix": selected_first["prefix"]}]
        elif arm in ("random_bridge", "lookahead"):
            rng = np.random.default_rng(2026121200 + root)
            for item in first:
                for name, suffix in (("zero", np.zeros((5, 2))), ("random", rng.uniform(-1, 1, (5, 2)))):
                    bank.append({"name": f'{item["name"]}/{name}',
                                 "prefix": np.concatenate((item["prefix"], suffix)).tolist()})
            if arm == "lookahead":
                for item in bank:
                    scores.append(float(predict(np.asarray(item["prefix"])[None])[1][0]))
                bank = [bank[int(np.argmin(scores))]]
        else:
            raise ValueError("unknown control")
        affordable = (480 - 3 - meter.token_samples) // 3
        counts = allocate_samples(3 * affordable // 4, 0, len(bank), 1)
        for item, count in zip(bank, counts, strict=True):
            chain, score, _ = optimize(item["prefix"], 15, count,
                                       np.random.default_rng(seed), lambda a: predict(a)[1])
            programs.append({"name": item["name"], "chain": chain.tolist(),
                             "terminal_score": score, "samples": count})
        selected = min(programs, key=lambda item: item["terminal_score"])
        count = (480 - 3 - meter.token_samples) // 3
        chain, score = prior.policy.policy.refine(selected["chain"], selected["terminal_score"], count,
                                                   np.random.default_rng(seed + 1), lambda a: predict(a)[1])
        if score < selected["terminal_score"]:
            selected = {"name": "joint_refined", "chain": chain.tolist(),
                        "terminal_score": score, "refined_from": selected["name"], "samples": count}
    original = selected.copy()
    comparison = np.asarray(original["chain"]).copy()
    if arm != "full_cem":
        comparison[:5] = 0.
    comparison_score = float(predict(comparison[None])[1][0])
    abstained = arm != "full_cem" and comparison_score <= original["terminal_score"]
    if abstained:
        selected = {"name": "zero_prefix", "chain": comparison.tolist(), "terminal_score": comparison_score}
    return {"root": root, "task": task, "arm": arm, "first_bank": first, "bank": bank,
            "selection_scores": scores, "programs": programs, "precomparison": original,
            "comparison_chain": comparison.tolist(), "comparison_score": comparison_score,
            "abstained": bool(abstained) if arm != "full_cem" else None,
            "selected": selected, "tokens": meter.token_samples, "svd": structure.counts.get("svd", 0),
            "wall_seconds": time.monotonic() - start, "trace": trace}


def plan():
    if PLANS.exists() or AUDIT.exists():
        raise FileExistsError("control output exists")
    source, sealed = dependencies(), json.loads(prior.PLANS.read_text())
    if sealed["dependencies"] != prior.dependencies() or json.loads(prior.AUDIT.read_text())["plans_sha256"] != prior.sha(prior.PLANS):
        raise ValueError("prior source seal mismatch")
    rows = [decide(row["task"], row["root"], arm, row["result"]["first_bank"])
            for row in sealed["rows"] for arm in ARMS]
    if source != dependencies():
        raise ValueError("source drift")
    with PLANS.open("x") as stream:
        json.dump({"dependencies": source, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print(f"Sealed {len(rows)} exact control decisions.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    data = json.loads(PLANS.read_text())
    if data["dependencies"] != dependencies():
        raise ValueError("source drift")
    rows = []
    for row in data["rows"]:
        selected, total, found = np.asarray(row["selected"]["chain"]), 0, False
        for batch in row["trace"]:
            requested, actions = np.asarray(batch["requested"]), np.asarray(batch["model"], dtype=np.float32)
            np.testing.assert_array_equal(requested.astype(np.float32), actions)
            pos = prior.exact.endpoints(row["task"], actions)
            scores = np.linalg.norm(pos - row["task"]["goal"], axis=1)
            np.testing.assert_allclose(pos, batch["positions"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(scores, batch["scores"], atol=1e-12, rtol=0)
            if requested.shape[1:] == selected.shape:
                idx = np.flatnonzero(np.all(requested == selected, axis=(1, 2)))
                found |= any(abs(scores[i] - row["selected"]["terminal_score"]) < 1e-12 for i in idx)
            cost = len(actions) * actions.shape[1] // 5
            if cost != batch["tokens"]:
                raise ValueError("incorrect batch charge")
            total += cost
        if not found or total != row["tokens"] or total > 480:
            raise ValueError("selection or token mismatch")
        rows.append({"root": row["root"], "arm": row["arm"], "tokens": total, "batches": len(row["trace"])})
    with AUDIT.open("x") as stream:
        json.dump({"plans_sha256": prior.sha(PLANS), "rows": rows}, stream, indent=2)
    print(f"Recomputed all prediction batches for {len(rows)} controls.", flush=True)


if __name__ == "__main__":
    commands = {"plan": plan, "audit": audit}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise SystemExit("choose plan or audit")
    commands[sys.argv[1]]()
