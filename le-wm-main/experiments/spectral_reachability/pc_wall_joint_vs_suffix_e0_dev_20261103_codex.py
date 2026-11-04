"""Paired CEM ablation: same starts, fixed prefix versus joint action parameters."""
import json
import time
from pathlib import Path

import numpy as np

from . import pc_wall_diverse_response_e0_dev_20261103_codex as diverse

factor, joint, base = diverse.factor, diverse.joint, diverse.base
PROTOCOL = diverse.previous.ROOT.parent / "docs/PC_WALL_JOINT_VS_SUFFIX_E0_DEV_PROTOCOL_20261103_CODEX.md"
OUTPUT = diverse.previous.ROOT / "results/pc_wall_joint_vs_suffix_e0_dev_20261103_codex.json"
METHODS = ("fixed", "joint")


def plan(task, index, retained, cap, method):
    if cap not in diverse.previous.BUDGETS or method not in METHODS or len(retained) != 8:
        raise ValueError("invalid protocol")
    started = time.monotonic()
    bank, probes = factor.prior.prefix_bank(), factor.prior.probe_family()
    calls = 33
    starts, start_scores = [], []
    for prefix_index in retained:
        prefix = bank[prefix_index]
        scores = factor.prior.evaluate(task, prefix, probes)
        calls += len(probes)
        suffix = probes[int(np.argmin(scores))]
        pre = diverse.prefix_parameters(prefix)
        point = np.array([*pre, suffix[0, 1] * 2 / np.pi, suffix[2, 0] / .3])
        starts.append(point)
        start_scores.append(float(np.min(scores)))
    best_index = int(np.argmin(start_scores))
    best, best_score = starts[best_index].copy(), start_scores[best_index]
    incumbent = {"point": best.tolist(), "distance": best_score}
    remaining = cap - calls
    allocations = [remaining // 16 + (i < remaining % 16) for i in range(16)]
    rng = np.random.default_rng(2026112500 + index)
    iterations = []
    for i, start in enumerate(starts):
        mean, std = start.copy(), np.full(4, .5)
        for iteration in range(2):
            n = allocations[2 * i + iteration]
            samples = np.clip(mean + std * rng.standard_normal((n, 4)), -1, 1)
            if method == "fixed":
                samples[:, :2] = start[:2]
            traces, _ = base.rollout(task["start"], joint.decode(samples))
            calls += n
            scores = np.linalg.norm(traces[:, -1, :2] - task["goal"], axis=1)
            order = np.argsort(scores, kind="stable")
            if scores[order[0]] < best_score:
                best, best_score = samples[order[0]].copy(), float(scores[order[0]])
            elite = samples[order[:max(2, n // 8)]]
            mean, std = elite.mean(axis=0), np.maximum(elite.std(axis=0), .02)
            iterations.append({"start": i, "round": iteration, "parameters": samples.tolist(), "scores": scores.tolist()})
    actions = joint.decode(best[None])
    trajectory, blocked = base.rollout(task["start"], actions)
    final = float(np.linalg.norm(trajectory[0, -1, :2] - task["goal"]))
    if calls != cap or not np.isclose(final, best_score, atol=1e-12, rtol=0):
        raise ValueError("budget or replay mismatch")
    return {"method": method, "rollout_calls": calls, "starts": np.asarray(starts).tolist(),
            "start_scores": start_scores, "incumbent": incumbent, "selected": best.tolist(),
            "actions": actions[0].tolist(), "distance": final, "trajectory": trajectory[0].tolist(),
            "blocked": blocked[0].tolist(), "iterations": iterations, "seconds": time.monotonic() - started}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sealed, old = json.loads(factor.SEAL.read_text()), factor.load_prior()
    if sealed["prior_sha256"] != base.digest(factor.prior.OUTPUT) or sealed["runner_sha256"] != base.digest(Path(factor.__file__)):
        raise ValueError("retention seal mismatch")
    paths = (PROTOCOL, Path(__file__), factor.SEAL, factor.prior.OUTPUT, Path(diverse.__file__),
             Path(factor.__file__), Path(factor.prior.__file__), Path(joint.__file__),
             Path(base.__file__), Path(base.physics.__file__))
    hashes = {str(p): base.digest(p) for p in paths}
    rows = []
    print("Identical starts and noise; fixed prefix vs full joint optimizer.", flush=True)
    for oldrow, row in zip(old["rows"], sealed["rows"], strict=True):
        index, task = oldrow["index"], oldrow["task"]
        retained = row["retained"]["diverse"]
        if index != row["index"] or retained != factor.diverse(task, oldrow["screen"]):
            raise ValueError("retention changed")
        plans = [plan(task, index, retained, cap, method) for cap in diverse.previous.BUDGETS for method in METHODS]
        rows.append({"index": index, "task": task, "retained": retained, "plans": plans})
        print(f"task {index} complete", flush=True)
    if any(base.digest(Path(p)) != h for p, h in hashes.items()):
        raise ValueError("dependency drift")
    summary = {}
    for cap in diverse.previous.BUDGETS:
        summary[str(cap)] = {}
        for method in METHODS:
            chosen = [p for row in rows for p in row["plans"] if p["rollout_calls"] == cap and p["method"] == method]
            summary[str(cap)][method] = {"mean": float(np.mean([p["distance"] for p in chosen])),
                                         "successes": sum(p["distance"] < .05 for p in chosen)}
    with OUTPUT.open("x") as handle:
        json.dump({"dependencies": hashes, "rows": rows, "summary": summary, "PC2_pass": False,
                   "PC3_pass": False, "full_cost_certified": False}, handle, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
