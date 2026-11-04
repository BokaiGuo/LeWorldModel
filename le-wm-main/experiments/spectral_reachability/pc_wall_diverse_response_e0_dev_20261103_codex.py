"""Diverse fixed prefixes, suffix-response starts, joint full-action refinement."""
import json
import time
from pathlib import Path

import numpy as np

from . import pc_wall_response_joint_e0_dev_20261103_codex as previous
from .d2p_goal_mode_rank_20261102 import rank_goal_modes

base, joint, factor = previous.base, previous.joint, previous.joint.factor
PROTOCOL = previous.ROOT.parent / "docs/PC_WALL_DIVERSE_RESPONSE_E0_DEV_PROTOCOL_20261103_CODEX.md"
OUTPUT = previous.ROOT / "results/pc_wall_diverse_response_e0_dev_20261103_codex.json"
METHODS = ("goal", "singular", "ridge", "random", "nominal")


def prefix_parameters(prefix):
    a = np.asarray(prefix)
    point = np.array([a[0, 1] * 2 / np.pi, a[2, 0] / .3])
    reconstructed = joint.decode(np.array([[*point, 0., 0.]]))[0, :6]
    if not np.allclose(a, reconstructed, atol=1e-12, rtol=0):
        raise ValueError("prefix template mismatch")
    return point


def plan(task, index, retained, cap, method):
    if method not in METHODS or cap not in previous.BUDGETS or len(retained) != 8:
        raise ValueError("invalid development setting")
    started = time.monotonic()
    bank = factor.prior.prefix_bank()
    calls = 33  # Reused fixed screen is charged to this hypothetical arm.

    def endpoints(parameters):
        nonlocal calls
        params = np.asarray(parameters)
        calls += len(params)
        if calls > cap:
            raise ValueError("budget exceeded")
        trajectory, _ = base.rollout(task["start"], joint.decode(params))
        return trajectory[:, -1, :2]

    starts, evidence = [], []
    for position, selected in enumerate(retained):
        pre = prefix_parameters(bank[selected])
        center = np.array([*pre, 0., .5])
        if method in ("goal", "singular", "ridge"):
            eps = .01
            center_out = endpoints(center[None])[0]
            directions = np.eye(4)[2:]
            positive, negative = endpoints(center + eps * directions), endpoints(center - eps * directions)
            response = ((positive - negative) / (2 * eps)).T
            residual = np.asarray(task["goal"]) - center_out
            if method == "ridge":
                update = response.T @ np.linalg.solve(response @ response.T + .001 * np.eye(2), residual)
                detail = {}
            else:
                ranked = rank_goal_modes(response, residual, .001)
                mode = int(ranked["order"][0]) if method == "goal" else 0
                update = ranked["right_t"][mode] * ranked["coefficients"][mode]
                detail = {"mode": mode, "singular": ranked["singular"].tolist(),
                          "contributions": ranked["contributions"].tolist()}
            update *= min(1., .5 / max(float(np.max(np.abs(update))), 1e-30))
            center[2:] = np.clip(center[2:] + update, -1, 1)
            evidence.append({"prefix_index": selected, "response": response.tolist(),
                             "residual": residual.tolist(), **detail})
        elif method == "random":
            direction = np.random.default_rng(2026112300 + 10 * index + position).normal(size=2)
            center[2:] = np.clip(center[2:] + .5 * direction / np.max(np.abs(direction)), -1, 1)
        starts.append(center)
    starts = np.asarray(starts)
    generation_calls = calls
    values = np.linalg.norm(endpoints(starts) - task["goal"], axis=1)
    ix = int(np.argmin(values))
    best, best_cost = starts[ix].copy(), float(values[ix])
    incumbent = {"parameters": best.tolist(), "distance": best_cost}
    allocations = [(cap - calls) // 16 + (i < (cap - calls) % 16) for i in range(16)]
    rng = np.random.default_rng(2026112400 + index)
    trace = []
    for i, start in enumerate(starts):
        mean, std = start.copy(), np.full(4, .5)
        for step in range(2):
            n = allocations[2 * i + step]
            points = np.clip(mean + std * rng.standard_normal((n, 4)), -1, 1)
            scores = np.linalg.norm(endpoints(points) - task["goal"], axis=1)
            order = np.argsort(scores, kind="stable")
            if scores[order[0]] < best_cost:
                best, best_cost = points[order[0]].copy(), float(scores[order[0]])
            elite = points[order[:max(2, n // 8)]]
            mean, std = elite.mean(axis=0), np.maximum(elite.std(axis=0), .02)
            trace.append({"start": i, "round": step, "parameters": points.tolist(), "scores": scores.tolist()})
    actions = joint.decode(best[None])
    trajectory, blocked = base.rollout(task["start"], actions)
    end_cost = float(np.linalg.norm(trajectory[0, -1, :2] - task["goal"]))
    if calls != cap or not np.isclose(best_cost, end_cost, atol=1e-12, rtol=0):
        raise ValueError("final selected replay mismatch")
    initial = float(np.linalg.norm(np.asarray(task["start"])[:2] - task["goal"]))
    prefix_cost = float(np.linalg.norm(trajectory[0, 6, :2] - task["goal"]))
    suffixes = np.concatenate((factor.prior.probe_family(), actions[:, 6:]), axis=0)
    common = {}
    for name, prefix in (("selected", actions[0, :6]), ("zero", np.zeros((6, 2)))):
        scores = factor.prior.evaluate(task, prefix, suffixes)
        common[name] = {"distances": scores.tolist(), "active_change": (scores - scores[0]).tolist()}
    return {"method": method, "retained": retained, "starts": starts.tolist(), "response_evidence": evidence,
            "incumbent": incumbent, "selected_parameters": best.tolist(), "actions": actions[0].tolist(),
            "distance": end_cost, "trajectory": trajectory[0].tolist(), "blocked": blocked[0].tolist(),
            "initial_distance": initial, "prefix_distance": prefix_cost,
            "nonprogress": bool(prefix_cost >= initial - 1e-12), "common_suffix": common,
            "rollout_calls": calls, "generation_calls_including_screen": generation_calls,
            "svd_calls": 8 if method in ("goal", "singular") else 0, "solves": 8 if method == "ridge" else 0,
            "seconds": time.monotonic() - started, "optimization_trace": trace}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    seal = json.loads(factor.SEAL.read_text())
    if seal["prior_sha256"] != base.digest(factor.prior.OUTPUT) or seal["runner_sha256"] != base.digest(Path(factor.__file__)):
        raise ValueError("prior screen seal mismatch")
    paths = (PROTOCOL, Path(__file__), factor.SEAL, factor.prior.OUTPUT, Path(factor.__file__),
             Path(factor.prior.__file__), Path(joint.__file__), Path(base.__file__),
             Path(base.physics.__file__), Path(rank_goal_modes.__code__.co_filename))
    hashes = {str(p): base.digest(p) for p in paths}
    original = factor.load_prior()
    rows = []
    print("Diverse state-prefix responses and joint refinement; exposed exact tasks only.", flush=True)
    for priorrow, frozen in zip(original["rows"], seal["rows"], strict=True):
        index, task = priorrow["index"], priorrow["task"]
        retained = frozen["retained"]["diverse"]
        if index != frozen["index"] or retained != factor.diverse(task, priorrow["screen"]):
            raise ValueError("screen mismatch")
        rows.append({"index": index, "task": task, "plans": [plan(task, index, retained, cap, method)
                     for cap in previous.BUDGETS for method in METHODS]})
        print(f"task {index} complete", flush=True)
    if any(base.digest(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError("source changed")
    summary = {}
    for cap in previous.BUDGETS:
        summary[str(cap)] = {}
        for method in METHODS:
            selected = [p for r in rows for p in r["plans"] if p["rollout_calls"] == cap and p["method"] == method]
            summary[str(cap)][method] = {"mean": float(np.mean([p["distance"] for p in selected])),
                                          "successes": sum(p["distance"] < .05 for p in selected),
                                          "nonprogress": sum(p["nonprogress"] for p in selected)}
    with OUTPUT.open("x") as file:
        json.dump({"dependencies": hashes, "rows": rows, "summary": summary,
                   "scope": "exposed exact-state development; not RGB or independent",
                   "PC2_pass": False, "PC3_pass": False, "full_cost_certified": False},
                  file, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
