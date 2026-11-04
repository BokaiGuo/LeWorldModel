"""Post-pilot random-waypoint exact wall-unicycle control."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from . import pc_wall_unicycle_atlas_cem_20261103 as wall

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/pc_wall_unicycle_atlas_cem_20261103_DEV01.json"
PROTOCOL = ROOT.parent / "docs/PC_WALL_UNICYCLE_WAYPOINT_CONTROL_PROTOCOL_20261103.md"
COUNT = 207


def build(waypoint):
    waypoint = np.asarray(waypoint, dtype=np.float64)
    if waypoint.shape != (2,) or not np.isfinite(waypoint).all():
        raise ValueError("invalid waypoint")
    actions, position, heading = [], wall.START[:2], float(wall.START[2])
    for target in (waypoint, wall.GOAL[:2]):
        vector = target - position
        length = float(np.linalg.norm(vector))
        if length > 1e-12:
            angle = float(np.arctan2(vector[1], vector[0]))
            turn = (angle - heading + np.pi) % (2 * np.pi) - np.pi
            turns = int(np.ceil(abs(turn) / wall.HIGH[1]))
            if turns:
                actions.extend([[0., turn / turns]] * turns)
            steps = int(np.ceil(length / wall.HIGH[0]))
            actions.extend([[length / steps, 0.]] * steps)
            heading = angle
        position = target
    actions = actions[:wall.HORIZON]
    actions.extend([[0., 0.]] * (wall.HORIZON - len(actions)))
    return np.asarray(actions, dtype=np.float64)


def run(seed, cap):
    if cap < COUNT * wall.HORIZON:
        raise ValueError("transition cap too small")
    rng = np.random.default_rng(seed)
    points = rng.uniform([-.5, -1.5], [.5, 1.5], size=(COUNT, 2))
    backend = wall.Backend()
    best, index, scores = None, None, []
    start = perf_counter()
    for i, point in enumerate(points):
        program = build(point)
        endpoint = backend.rollout(wall.ROOT, program)["latent_history"][-1]
        score = float(np.linalg.norm(np.asarray(endpoint)[:2] - wall.GOAL[:2]))
        scores.append(score)
        if index is None or score < scores[index]:
            index, best = i, program
    elapsed = perf_counter() - start
    outcome = wall.replay(best)
    if not np.isclose(outcome["terminal_error"], scores[index], rtol=0, atol=1e-12):
        raise ValueError("selected replay mismatch")
    if backend.transitions > cap:
        raise ValueError("transition cap exceeded")
    return {"seed": seed, "selected_index": index, "waypoints": points.tolist(), "scores": scores,
            "selected": outcome, "backend": vars(backend).copy(), "replay_transitions": wall.HORIZON,
            "wall_seconds": elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(SOURCE.read_text())
    if [r["seed"] for r in source["rows"]] != [20261103, 20261104, 20261105]:
        raise ValueError("source roster")
    print("Post-pilot random waypoint control; no wall geometry in program generation.", flush=True)
    rows = [run(r["seed"], r["atlas"]["backend"]["transitions"]) for r in source["rows"]]
    for row in rows:
        print(f"seed={row['seed']} error={row['selected']['terminal_error']:.6g}", flush=True)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = {"schema": "wall_unicycle_waypoint_control.v1", "rows": rows,
              "scope": "post-pilot exact single task, not learned model, independent or admission",
              "source_sha256": sha(SOURCE), "protocol_sha256": sha(PROTOCOL),
              "script_sha256": sha(Path(__file__)), "backend_sha256": sha(Path(wall.__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
