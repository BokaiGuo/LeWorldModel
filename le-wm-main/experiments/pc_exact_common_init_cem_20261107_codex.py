"""Common geometry initialization versus whole-chain primitive refinement."""
import json
import time
from pathlib import Path

import numpy as np

from experiments import pc_exact_primitive_cem_control_20261107_codex as primitive

ROOT = primitive.ROOT
PROTOCOL = ROOT.parent / "docs/PC_EXACT_COMMON_INIT_CEM_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_exact_common_init_cem_20261107_codex.json"
AUDIT = ROOT / "results/pc_exact_common_init_cem_audit_20261107_codex.json"
bridge = primitive.bridge
prior = primitive.prior


def waypoint_params(start, goal, waypoint):
    start, goal, waypoint = map(lambda x: np.asarray(x, float), (start, goal, waypoint))
    direct = goal - start[:2]
    distance = np.linalg.norm(direct)
    if distance <= 1e-12:
        raise ValueError("degenerate goal")
    bearing = np.arctan2(direct[1], direct[0])
    p = []
    for leg in (waypoint - start[:2], goal - waypoint):
        angle = (np.arctan2(leg[1], leg[0]) - bearing + np.pi) % (2 * np.pi) - np.pi
        p.extend((angle / (np.pi / 2), np.linalg.norm(leg) / distance - 1))
    result = np.asarray(p)
    if not np.isfinite(result).all() or np.any(np.abs(result) > 1):
        raise ValueError("waypoint outside primitive support")
    return result


def score(task, actions):
    return np.linalg.norm(prior.exact.endpoints(task, actions) - task["goal"], axis=1)


def run_one(old):
    task, root = old["task"], old["root"]
    started = time.perf_counter()
    offsets = np.random.default_rng(2718000 + root).uniform(-1.9, 1.9, 384)
    proposals = [bridge.make_chain(task["start"], task["goal"], x) for x in offsets]
    geometry = np.stack([p[0] for p in proposals])
    parameters = np.stack([waypoint_params(task["start"], task["goal"], p[1])
                           for p in proposals[:128]])
    reconstructed = np.stack([primitive.chain_from_parameters(task["start"], task["goal"], p)[0]
                              for p in parameters])
    np.testing.assert_allclose(geometry[:128], reconstructed, rtol=0, atol=1e-12)
    geometry_costs = score(task, geometry)
    geometry_seconds = time.perf_counter() - started
    started = time.perf_counter()
    elite = parameters[np.argsort(geometry_costs[:128], kind="stable")[:16]]
    mean, scale = elite.mean(axis=0), np.maximum(elite.std(axis=0), .08)
    rng = np.random.default_rng(2719000 + root)
    refined_params, refined_actions, refined_costs = [], [], []
    for _ in range(4):
        samples = np.clip(mean + rng.normal(size=(64, 4)) * scale, -1, 1)
        actions = np.stack([primitive.chain_from_parameters(task["start"], task["goal"], p)[0]
                            for p in samples])
        costs = score(task, actions)
        elite = samples[np.argsort(costs, kind="stable")[:16]]
        mean, scale = elite.mean(axis=0), np.maximum(elite.std(axis=0), .08)
        refined_params.extend(samples.tolist())
        refined_actions.extend(actions.tolist())
        refined_costs.extend(costs.tolist())
    joint_seconds = time.perf_counter() - started
    joint_costs = np.r_[geometry_costs[:128], refined_costs]
    return {"root": root, "task": task, "offsets": offsets.tolist(),
            "geometry_actions": geometry.tolist(), "geometry_costs": geometry_costs.tolist(),
            "geometry_best": int(np.argmin(geometry_costs)), "initial_parameters": parameters.tolist(),
            "refined_parameters": refined_params, "refined_actions": refined_actions,
            "refined_costs": refined_costs, "joint_best": int(np.argmin(joint_costs)),
            "counts": {"geometry": 384, "joint": 384},
            "timing": {"geometry_generation_and_score": geometry_seconds,
                       "joint_refinement_only": joint_seconds}}


def main():
    if OUTPUT.exists() or AUDIT.exists():
        raise FileExistsError("output exists")
    old = json.loads(bridge.OUTPUT.read_text())
    if old["dependencies"] != {p: prior.sha(p) for p in old["dependencies"]}:
        raise ValueError("sealed source drift")
    paths = (Path(__file__), PROTOCOL, Path(primitive.__file__), Path(bridge.__file__),
             bridge.OUTPUT, Path(prior.exact.__file__), Path(prior.exact.wall.__file__),
             Path(prior.exact.wall.physics.__file__))
    deps = {str(p): prior.sha(p) for p in paths}
    rows = []
    for row in old["rows"]:
        result = run_one(row)
        rows.append(result)
        joint = np.r_[result["geometry_costs"][:128], result["refined_costs"]]
        print(f"root {row['root']}: geometry={min(result['geometry_costs']):.5f}, "
              f"joint={joint.min():.5f}", flush=True)
    if deps != {p: prior.sha(p) for p in deps}:
        raise ValueError("dependency changed")
    with OUTPUT.open("x") as f:
        json.dump({"dependencies": deps, "rows": rows, "scope": "exact-state exposed development"},
                  f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
