"""Privileged exact-state waypoint ceiling for the exposed wall triprefix."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .pc_wall_triprefix_cem_intervention_20261103_codex import ROLES, SEEDS, TAIL
from .pc_wall_unicycle_atlas_cem_20261103 import Backend, GOAL, HIGH, LOW, ROOT, replay

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "le-wm-main/results/pc_wall_triprefix_cem_intervention_20261103_codex_DEV01.json"
PROTOCOL = REPO / "docs/PC_WALL_TRIPREFIX_WAYPOINT_CEILING_PROTOCOL_20261103_CODEX.md"
COUNT = 256


def build(state, waypoint):
    state, waypoint = np.asarray(state, dtype=float), np.asarray(waypoint, dtype=float)
    if state.shape != (3,) or waypoint.shape != (2,) or not np.isfinite(state).all() or not np.isfinite(waypoint).all():
        raise ValueError("invalid pose or waypoint")
    actions, position, heading = [], state[:2].copy(), float(state[2])
    for target in (waypoint, GOAL[:2]):
        vector = target - position
        length = float(np.linalg.norm(vector))
        if length > 1e-12:
            angle = float(np.arctan2(vector[1], vector[0]))
            turn = (angle - heading + np.pi) % (2 * np.pi) - np.pi
            turns = int(np.ceil(abs(turn) / HIGH[1]))
            if turns:
                actions.extend([[0., turn / turns]] * turns)
            steps = int(np.ceil(length / HIGH[0]))
            actions.extend([[length / steps, 0.]] * steps)
            heading = angle
        position = target
    actions = actions[:TAIL]
    actions.extend([[0., 0.]] * (TAIL - len(actions)))
    result = np.asarray(actions, dtype=float)
    if np.any(result < LOW) or np.any(result > HIGH):
        raise ValueError("invalid bounded suffix")
    return result


def run(prefix, seed):
    prefix = np.asarray(prefix, dtype=float)
    if prefix.shape != (4, 2) or not np.isfinite(prefix).all():
        raise ValueError("invalid prefix")
    prefix_backend = Backend()
    context = prefix_backend.rollout(ROOT, prefix)
    state = np.asarray(context["latent_history"][-1])
    points = np.random.default_rng(seed).uniform([-.5, -1.5], [.5, 1.5], size=(COUNT, 2))
    backend, scores, best = Backend(), [], None
    start = perf_counter()
    for point in points:
        suffix = build(state, point)
        end = backend.rollout(context, suffix)["latent_history"][-1]
        score = float(np.linalg.norm(np.asarray(end)[:2] - GOAL[:2]))
        if best is None or score < scores[best]:
            best, selected = len(scores), suffix.copy()
        scores.append(score)
    elapsed = perf_counter() - start
    result = replay(np.concatenate([prefix, selected]))
    if not np.isclose(result["terminal_error"], scores[best], atol=1e-12, rtol=0):
        raise ValueError("replay mismatch")
    if backend.transitions != COUNT * TAIL or prefix_backend.transitions != 4:
        raise ValueError("budget mismatch")
    return {"seed": seed, "prefix": prefix.tolist(), "prefix_state": state.tolist(),
            "selected_index": best, "waypoints": points.tolist(), "scores": scores,
            "success_below_1e8": sum(score < 1e-8 for score in scores),
            "replay": result, "backend": vars(backend), "prefix_backend": vars(prefix_backend),
            "wall_seconds": elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = [Path(__file__), SOURCE, PROTOCOL,
               Path(__file__).with_name("pc_wall_triprefix_cem_intervention_20261103_codex.py"),
               Path(__file__).with_name("pc_wall_unicycle_atlas_cem_20261103.py"),
               Path(__file__).with_name("pc_exact_wall_kill.py")]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    hashes = {str(p.relative_to(REPO)): digest(p) for p in sources}
    source = json.loads(SOURCE.read_text())
    prefixes = source["prefixes"]
    if set(prefixes) != set(ROLES):
        raise ValueError("prefix roster mismatch")
    print("Privileged triprefix exact-state waypoint ceiling; exposed task.", flush=True)
    rows = []
    for seed in SEEDS:
        arms = {role: run(prefixes[role], seed) for role in ROLES}
        if any(arms[role]["waypoints"] != arms["selected"]["waypoints"] for role in ROLES):
            raise ValueError("unpaired waypoint samples")
        rows.append({"seed": seed, "arms": arms})
        print(f"seed {seed}: " + ", ".join(
            f"{role}={arms[role]['replay']['terminal_error']:.6g}" for role in ROLES), flush=True)
    if hashes != {str(p.relative_to(REPO)): digest(p) for p in sources}:
        raise ValueError("source changed during run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump({"schema": "pc_wall_triprefix_waypoint_ceiling.v1",
                   "scope": "privileged exact-state exposed-task diagnostic only",
                   "source_sha256": hashes, "numpy_version": np.__version__,
                   "rows": rows}, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
