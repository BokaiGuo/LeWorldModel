"""Repair geometric candidate waypoint regression while retaining V1 evaluation."""
import json

import numpy as np

from experiments import pc_tworoom_rgb_geometric_candidates_20261107_codex as v1
from experiments.pc_input_permissions import file_hash

PROTOCOL = v1.ROOT.parent / "docs/PC_TWOROOM_RGB_GEOMETRIC_CANDIDATE_V2_PROTOCOL_20261107_CODEX.md"
OUTPUT = v1.ROOT / "results/pc_tworoom_rgb_geometric_candidates_v2_20261107_codex.json"
MANIFEST = v1.ROOT / "results/pc_tworoom_rgb_geometric_candidates_v2_manifest_20261107_codex.json"


def candidate_pool(current, goal, old_tail, remaining):
    if remaining not in (5, 10, 15, 20, 25):
        raise ValueError("invalid remaining horizon")
    old = v1.panel.source.cem.old_or_hold(old_tail)[:remaining]
    try:
        start, target, wall, door = v1.geometry(current, goal)
    except ValueError:
        return np.concatenate((old[None], np.zeros((15, remaining, 2), np.float32))), None
    chains = []
    for offset in (0., -3., 3., -6., 6.):
        for gain in (1., .8, 1.2):
            position, stage = start.copy(), 0
            side = -1. if start[0] < wall else 1.
            crossing = (start[0] - wall) * (target[0] - wall) < 0
            if crossing:
                safe_x = wall + side * 16.
                first_x = safe_x if side * (start[0] - safe_x) > 0 else start[0]
                points = [np.array([first_x, door + offset]),
                          np.array([wall - side * 16., door + offset]), target]
            else:
                points = [target]
            actions = []
            for _ in range(remaining):
                while stage < len(points) - 1 and np.max(np.abs(position - points[stage])) < 1e-4:
                    stage += 1
                action = np.clip((points[stage] - position) / 5. * gain, -1., 1.)
                position += 5. * action
                actions.append(action)
            chains.append(actions)
    pool = np.concatenate((old[None], np.asarray(chains, dtype=np.float32)))
    if pool.shape != (16, remaining, 2) or not np.isfinite(pool).all() or np.abs(pool).max() > 1:
        raise ValueError("illegal candidate")
    return pool, {"start": start.tolist(), "target": target.tolist(), "wall": wall, "door": door}


def main():
    if OUTPUT.exists() or MANIFEST.exists():
        raise FileExistsError("V2 result or manifest already exists")
    v1.PROTOCOL = PROTOCOL
    v1.OUTPUT = OUTPUT
    v1.candidate_pool = candidate_pool
    v1.main()
    with MANIFEST.open("x") as stream:
        json.dump({"wrapper_sha256": file_hash(__file__), "shared_runner_sha256": file_hash(v1.__file__),
                   "protocol_sha256": file_hash(PROTOCOL), "result_sha256": file_hash(OUTPUT)},
                  stream, indent=2)


if __name__ == "__main__":
    main()
