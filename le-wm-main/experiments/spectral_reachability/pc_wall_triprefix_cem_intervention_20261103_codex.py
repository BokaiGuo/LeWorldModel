"""Fixed-prefix exact wall intervention, development only."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .pc_wall_unicycle_atlas_cem_20261103 import Backend, GOAL, HIGH, LOW, ROOT, replay

SEEDS = (20261103, 20261104, 20261105)
ROLES = ("selected", "zero", "progress")
ROUNDS, POPULATION, ELITES, TAIL = 4, 64, 8, 12
REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "le-wm-main/results/pc_wall_atlas_bound_normalized_20261103_codex_DEV01.json"
PROTOCOL = REPO / "docs/PC_WALL_TRIPREFIX_CEM_INTERVENTION_PROTOCOL_20261103_CODEX.md"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_prefixes(report):
    matches = [row for row in report["rows"]
               if row["seed"] == SEEDS[0] and row["arm"] == "spectral"]
    if len(matches) != 1:
        raise ValueError("source arm not unique")
    bank = matches[0]["root_bank"]
    if [item["name"] for item in bank] != ["zero", "progress", "extra_minus", "extra_plus"]:
        raise ValueError("unexpected source candidate order")
    if matches[0]["selected_root"] != "extra_minus":
        raise ValueError("source selection does not bind spectral minus")
    prefixes = {"selected": bank[2]["actions"], "zero": bank[0]["actions"],
                "progress": bank[1]["actions"]}
    for value in prefixes.values():
        action = np.asarray(value, dtype=float)
        if (action.shape != (4, 2) or not np.isfinite(action).all()
                or np.any(action < LOW) or np.any(action > HIGH)):
            raise ValueError("invalid prefix")
    return prefixes


def optimize(prefix, seed):
    prefix = np.asarray(prefix, dtype=float)
    prefix_backend = Backend()
    context = prefix_backend.rollout(ROOT, prefix)
    backend = Backend()
    rng = np.random.default_rng(seed)
    mean = np.zeros((TAIL, 2))
    std = np.broadcast_to(HIGH / 2, (TAIL, 2)).copy()
    best, best_distance, history = None, float("inf"), []
    start = perf_counter()
    for iteration in range(ROUNDS):
        proposals = np.clip(mean + std * rng.standard_normal((POPULATION, TAIL, 2)), LOW, HIGH)
        scores = np.empty(POPULATION)
        for index, tail in enumerate(proposals):
            successor = backend.rollout(context, tail)
            scores[index] = np.linalg.norm(np.asarray(successor["latent_history"][-1])[:2] - GOAL[:2])
        if not np.isfinite(scores).all():
            raise ValueError("non-finite suffix score")
        winner = int(np.argmin(scores))
        if scores[winner] < best_distance:
            best_distance, best = float(scores[winner]), proposals[winner].copy()
        elites = proposals[np.argsort(scores, kind="stable")[:ELITES]]
        mean, std = elites.mean(axis=0), np.maximum(elites.std(axis=0), 1e-5)
        history.append({"round": iteration, "winner_index": winner,
                        "winner_distance": float(scores[winner]), "best_distance": best_distance,
                        "elite_indices": np.argsort(scores, kind="stable")[:ELITES].tolist()})
    elapsed = perf_counter() - start
    checked = replay(np.concatenate([prefix, best]))
    if not np.isclose(checked["terminal_error"], best_distance, rtol=0, atol=1e-12):
        raise ValueError("suffix replay mismatch")
    if backend.transitions != ROUNDS * POPULATION * TAIL or prefix_backend.transitions != 4:
        raise ValueError("transition count mismatch")
    return {"prefix": prefix.tolist(), "prefix_state": context["latent_history"][-1],
            "prefix_distance": float(np.linalg.norm(
                np.asarray(context["latent_history"][-1])[:2] - GOAL[:2])),
            "suffix": best.tolist(), "history": history, "replay": checked,
            "suffix_backend": vars(backend), "prefix_backend": vars(prefix_backend),
            "wall_seconds": elapsed}


def evaluate_common(prefixes, suffix):
    out = {}
    for role in ROLES:
        prefix = np.asarray(prefixes[role])
        zero = replay(np.concatenate([prefix, np.zeros((TAIL, 2))]))
        common = replay(np.concatenate([prefix, np.asarray(suffix)]))
        out[role] = {"zero_tail": zero, "common_tail": common,
                     "distance_change_vs_zero_tail": common["terminal_error"] - zero["terminal_error"]}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = [Path(__file__), SOURCE, PROTOCOL,
               Path(__file__).with_name("pc_wall_unicycle_atlas_cem_20261103.py"),
               Path(__file__).with_name("pc_exact_wall_kill.py")]
    hashes = {str(p.relative_to(REPO)): sha(p) for p in sources}
    prefixes = get_prefixes(json.loads(SOURCE.read_text()))
    print("Triprefix wall development diagnostic: common suffix and independent CEM.", flush=True)
    rows = []
    for seed in SEEDS:
        optimized = {role: optimize(prefixes[role], seed) for role in ROLES}
        common = evaluate_common(prefixes, optimized["selected"]["suffix"])
        rows.append({"seed": seed, "optimized": optimized, "common": common})
        print(f"seed {seed}: " + ", ".join(
            f"{role}={optimized[role]['replay']['terminal_error']:.6f}" for role in ROLES), flush=True)
    if hashes != {str(p.relative_to(REPO)): sha(p) for p in sources}:
        raise ValueError("source changed during run")
    result = {"schema": "pc_wall_triprefix_cem_intervention.v1",
              "scope": "post-outcome single exact task, not independent physical or gate",
              "source_sha256": hashes, "numpy_version": np.__version__,
              "suffix_contract": {"rounds": ROUNDS, "population": POPULATION,
                                  "elites": ELITES, "horizon": TAIL},
              "prefixes": prefixes, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
