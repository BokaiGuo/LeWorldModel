"""Matched suffix search for the sealed goal-aligned first bank; diagnostic only."""
import json
import time
from pathlib import Path

import numpy as np

from experiments import pc_goal_aligned_successor_dev_20261103_codex as source
from experiments.spectral_reachability.pc_cem_scale_floor_v1 import optimize

PROTOCOL = source.PROTOCOL.parent / "PC_GOAL_FIRSTBANK_MATCHED_SUFFIX_PROTOCOL_20261103_CODEX.md"
OUTPUT = source.PREFIX.with_name(source.PREFIX.name + "_firstbank_matched_suffix.json")
AUDIT = source.PREFIX.with_name(source.PREFIX.name + "_firstbank_matched_suffix_audit.json")
STARTS, SAMPLES = 4, 160


def dependencies():
    paths = (Path(__file__), PROTOCOL, source.PLANS, source.AUDIT, Path(optimize.__code__.co_filename))
    return {str(p): source.sha(p) for p in paths}


def run_prefix(task, root, prefix):
    goal = np.asarray(task["goal"])
    prefix = np.asarray(prefix, dtype=float)
    if prefix.shape != (5, 2) or not np.isfinite(prefix).all() or np.any(np.abs(prefix) > 1):
        raise ValueError("invalid prefix")
    starts, cumulative, incumbent = [], [], float("inf")
    for index in range(STARTS):
        seed = 2031110300 + root * STARTS + index
        counts = {"calls": 0, "chains": 0, "batch_sizes": []}

        def predict(actions):
            model = np.asarray(actions, dtype=np.float32)
            if model.ndim != 3 or model.shape[1:] != (15, 2) or not np.isfinite(model).all() or np.any(np.abs(model) > 1):
                raise ValueError("invalid actions")
            if not np.array_equal(model[:, :5], np.broadcast_to(prefix.astype(np.float32), model[:, :5].shape)):
                raise ValueError("prefix changed")
            counts["calls"] += 1
            counts["chains"] += len(model)
            counts["batch_sizes"].append(len(model))
            pos = source.source.previous.prior.exact.endpoints(task, model)
            return np.linalg.norm(pos - goal, axis=1)

        chain, score, scores = optimize(prefix, 15, SAMPLES, np.random.default_rng(seed), predict)
        if counts["chains"] != SAMPLES or len(scores) != SAMPLES or score != min(scores):
            raise ValueError("search accounting mismatch")
        replay = float(np.linalg.norm(source.source.previous.prior.exact.endpoints(
            task, chain[None].astype(np.float32))[0] - goal))
        if abs(replay - score) > 1e-12:
            raise ValueError("best chain replay mismatch")
        incumbent = min(incumbent, score)
        starts.append({"seed": seed, "scores": scores, "best_chain": chain.tolist(), "best_score": score,
                       "prediction_calls": counts["calls"], "prediction_chains": counts["chains"],
                       "batch_sizes": counts["batch_sizes"], "prediction_tokens": 3 * counts["chains"]})
        cumulative.append(incumbent)
    return {"starts": starts, "cumulative_best": cumulative}


def main():
    if OUTPUT.exists() or AUDIT.exists():
        raise FileExistsError("diagnostic outputs already exist")
    hashes = dependencies()
    data, audit = (json.loads(p.read_text()) for p in (source.PLANS, source.AUDIT))
    if data["dependencies"] != source.dependencies() or audit["plans_sha256"] != source.sha(source.PLANS):
        raise ValueError("source seal mismatch")
    panel = [row for row in data["rows"] if row["arm"] == "goal"]
    if len(panel) != 24 or len({r["root"] for r in panel}) != 24:
        raise ValueError("goal panel mismatch")
    print("Post-hoc matched suffix search for every sealed first-bank prefix.", flush=True)
    rows = []
    started = time.monotonic()
    for row in panel:
        root, result = row["root"], row["result"]
        bank = result["first_bank"]
        if len(bank) != 5:
            raise ValueError("unexpected first-bank size")
        prefixes = []
        for index, item in enumerate(bank):
            outcome = run_prefix(row["task"], root, item["prefix"])
            prefixes.append({"index": index, "name": item["name"], "prefix": item["prefix"], **outcome})
        greedy = np.argsort(result["prefix_scores"], kind="stable")[:2].tolist()
        random = np.random.default_rng(2030110300 + root).choice(len(bank), size=2, replace=False).tolist()
        sets = {"retained": result["retained"], "greedy": greedy, "random": random}
        for indices in sets.values():
            if len(set(indices)) != 2 or any(not 0 <= i < len(bank) for i in indices):
                raise ValueError("invalid retained set")
        rows.append({"root": root, "task": row["task"], "sets": sets,
                     "original_selected_score": result["selected"]["terminal_score"], "prefixes": prefixes})
        print(f"root {root}: {len(bank)} matched prefixes", flush=True)
    if hashes != dependencies() or data["dependencies"] != source.dependencies():
        raise ValueError("dependency drift")
    with OUTPUT.open("x") as stream:
        json.dump({"dependencies": hashes, "elapsed_seconds": time.monotonic() - started, "rows": rows},
                  stream, separators=(",", ":"), allow_nan=False)
    print("Sealed matched first-bank suffix diagnostic.", flush=True)


if __name__ == "__main__":
    main()
