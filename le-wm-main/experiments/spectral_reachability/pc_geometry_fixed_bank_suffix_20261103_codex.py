"""Privileged, fixed-bank diagnostic; physical outcomes never enter RGB selection."""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from experiments.spectral_reachability import pc_geometry_persistent_feasibility_20261103_codex as base

PROTOCOL = base.PROJECT / "docs/PC_GEOMETRY_FIXED_BANK_SUFFIX_DIAGNOSTIC_PROTOCOL_20261103_CODEX.md"
OUTPUT = base.ROOT / "results/pc_geometry_fixed_bank_suffix_20261103_codex.json"
SEED = 2026110317


def replay(model, start, actions):
    data = mujoco.MjData(model)
    data.qpos[:] = start
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    states = [np.r_[data.qpos[:2], data.qvel[:2]].copy()]
    for action in actions:
        base.step(model, data, action)
        states.append(np.r_[data.qpos[:2], data.qvel[:2]].copy())
    return np.asarray(states)


def optimize(model, start, prefix, noise, second_mean):
    evaluations = []

    def evaluate(actions):
        value = float(np.linalg.norm(replay(model, start, np.concatenate((prefix, actions)))[-1, :2] - base.GOAL))
        evaluations.append({"suffix": actions.tolist(), "distance": value})
        return value

    for initial in (np.zeros((10, 2)), second_mean):
        mean = initial.copy()
        std = np.full((10, 2), 0.5)
        for generation in range(3):
            samples = np.clip(mean + noise[generation] * std, -1, 1)
            scores = np.asarray([evaluate(sample) for sample in samples])
            elites = np.argsort(scores, kind="stable")[:4]
            mean = samples[elites].mean(axis=0)
            std = np.maximum(samples[elites].std(axis=0, ddof=1), 0.02)
        evaluate(mean)
        evaluate(np.zeros((10, 2)))
    index = int(np.argmin([item["distance"] for item in evaluations]))
    return {"evaluations": evaluations, "best_index": index,
            "best_distance": evaluations[index]["distance"],
            "best_suffix": evaluations[index]["suffix"]}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if mujoco.__version__ != "3.1.6":
        raise RuntimeError("fixed protocol requires MuJoCo 3.1.6")
    source = json.loads(base.OUTPUT.read_text())
    if source["xml_sha256"] != base.digest(base.XML):
        raise ValueError("source XML mismatch")
    model = mujoco.MjModel.from_xml_string(base.XML.read_text())
    rng = np.random.default_rng(SEED)
    common = np.concatenate((np.zeros((1, 10, 2)), rng.uniform(-1, 1, (5, 10, 2))))
    noises = rng.standard_normal((4, 2, 3, 16, 10, 2))
    means = rng.uniform(-1, 1, (4, 10, 2))
    rows = []
    for root_index, start in enumerate(base.obstacle_intersection_starts()):
        direct = [r for r in source["rows"] if r["start"] == start.tolist() and r["policy"] == "direct"]
        if len(direct) != 1 or len(direct[0]["trace"]) < 5:
            raise ValueError("missing fixed direct prefix")
        prefixes = [("zero", np.zeros((5, 2))),
                    ("direct", np.asarray([r["action"] for r in direct[0]["trace"][:5]]))]
        for direction in range(8):
            vector = np.array([np.cos(direction * np.pi / 4), np.sin(direction * np.pi / 4)])
            prefixes.append((f"direction_{direction}", np.tile(vector, (5, 1))))
        for name, prefix in prefixes:
            trajectory = replay(model, start, prefix)
            prefix_distance = float(np.linalg.norm(trajectory[-1, :2] - base.GOAL))
            initial_distance = float(np.linalg.norm(start - base.GOAL))
            fixed = [float(np.linalg.norm(replay(model, start, np.concatenate((prefix, suffix)))[-1, :2] - base.GOAL))
                     for suffix in common]
            search = optimize(model, start, prefix, noises[root_index], means[root_index])
            rows.append({"root_index": root_index, "start": start.tolist(), "prefix_name": name,
                         "prefix_actions": prefix.tolist(), "prefix_states": trajectory.tolist(),
                         "initial_distance": initial_distance, "prefix_distance": prefix_distance,
                         "non_improving": prefix_distance >= initial_distance,
                         "common_distances": fixed,
                         "common_change_vs_zero": [x - fixed[0] for x in fixed], "search": search})
        print(f"root {root_index + 1}/4 complete", flush=True)
    report = {"schema": "pc_geometry_fixed_bank_suffix.v1", "mujoco": mujoco.__version__,
              "protocol_sha256": base.digest(PROTOCOL), "source_sha256": base.digest(base.OUTPUT),
              "xml_sha256": base.digest(base.XML), "script_sha256": base.digest(Path(__file__)),
              "seed": SEED, "common_suffixes": common.tolist(), "rows": rows}
    with OUTPUT.open("x") as handle:
        json.dump(report, handle, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
