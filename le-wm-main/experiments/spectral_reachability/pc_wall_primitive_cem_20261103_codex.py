"""Flat versus state-blind primitive CEM on the exposed exact wall task."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .pc_wall_unicycle_atlas_cem_20261103 import Backend, GOAL, HIGH, LOW, ROOT, replay

REPO = Path(__file__).resolve().parents[3]
PROTOCOL = REPO / "docs/PC_WALL_PRIMITIVE_CEM_PROTOCOL_20261103_CODEX.md"
SEEDS = (20261103, 20261104, 20261105)
POP, ELITES, ROUNDS = 51, 6, 4
ARMS = ("flat", "primitive")


def primitive(params):
    p = np.asarray(params, dtype=float)
    bounds = np.array([HIGH[1], HIGH[0], HIGH[1], HIGH[0]])
    if p.shape != (4,) or not np.isfinite(p).all() or np.any(np.abs(p) > bounds):
        raise ValueError("invalid primitive coordinates")
    return np.asarray([[0., p[0]]] + [[p[1], 0.]] * 5 + [[0., p[2]]] * 2
                      + [[p[3], 0.]] * 5 + [[0., 0.]] * 3)


def run(seed, arm):
    if arm not in ARMS:
        raise ValueError("unknown parameterization")
    shape = (16, 2) if arm == "flat" else (4,)
    high = np.broadcast_to(HIGH, shape).copy() if arm == "flat" else np.array(
        [HIGH[1], HIGH[0], HIGH[1], HIGH[0]])
    rng, backend = np.random.default_rng(seed), Backend()
    mean, std = np.zeros(shape), high / 2
    best_score, best_actions, rounds = float("inf"), None, []
    start = perf_counter()
    for iteration in range(ROUNDS):
        samples = np.clip(mean + std * rng.standard_normal((POP, *shape)), -high, high)
        scores = []
        for sample in samples:
            actions = sample if arm == "flat" else primitive(sample)
            end = backend.rollout(ROOT, actions)["latent_history"][-1]
            scores.append(float(np.linalg.norm(np.asarray(end)[:2] - GOAL[:2])))
        scores = np.asarray(scores)
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite score")
        order = np.argsort(scores, kind="stable")
        index = int(order[0])
        if scores[index] < best_score:
            best_score = float(scores[index])
            best_actions = (samples[index] if arm == "flat" else primitive(samples[index])).copy()
        elite = samples[order[:ELITES]]
        mean, std = elite.mean(axis=0), np.maximum(elite.std(axis=0), 1e-5)
        rounds.append({"round": iteration, "samples": samples.tolist(), "scores": scores.tolist(),
                       "elite_indices": order[:ELITES].tolist(), "best_so_far": best_score})
    elapsed = perf_counter() - start
    checked = replay(best_actions)
    if not np.isclose(checked["terminal_error"], best_score, atol=1e-12, rtol=0):
        raise ValueError("selected replay mismatch")
    if backend.transitions != ROUNDS * POP * 16 or backend.rollouts != ROUNDS * POP:
        raise ValueError("unequal evaluator budget")
    return {"arm": arm, "seed": seed, "rounds": rounds, "selected": checked,
            "backend": vars(backend), "wall_seconds": elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = [Path(__file__), PROTOCOL,
               Path(__file__).with_name("pc_wall_unicycle_atlas_cem_20261103.py"),
               Path(__file__).with_name("pc_exact_wall_kill.py")]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    hashes = {str(p.relative_to(REPO)): digest(p) for p in sources}
    print("Exposed wall: compare flat and state-blind primitive CEM.", flush=True)
    rows = []
    for seed in SEEDS:
        group = [run(seed, arm) for arm in ARMS]
        rows.extend(group)
        print(f"seed {seed}: " + ", ".join(
            f"{r['arm']}={r['selected']['terminal_error']:.6g}" for r in group), flush=True)
    if hashes != {str(p.relative_to(REPO)): digest(p) for p in sources}:
        raise ValueError("source changed during run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump({"schema": "pc_wall_primitive_cem.v1", "source_sha256": hashes,
                   "scope": "exposed analytical task, privileged true objective, not spectral",
                   "numpy_version": np.__version__, "rows": rows},
                  stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
