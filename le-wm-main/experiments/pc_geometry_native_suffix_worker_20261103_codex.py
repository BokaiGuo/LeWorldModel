"""Privileged native-dynamics diagnostic worker; never imported by RGB planning."""
import json
import sys
from pathlib import Path

import mujoco_py
import numpy as np


def main(request_path, xml_path, output_path):
    if mujoco_py.__version__ != "2.1.2.14":
        raise RuntimeError("wrong native binding")
    request = json.loads(Path(request_path).read_text())
    model = mujoco_py.load_model_from_xml(Path(xml_path).read_text())
    goal = np.array([1.8, 2.8])
    rng = np.random.default_rng(2026110318)
    bridges = rng.uniform(-1, 1, (2, 5, 2))
    common = np.concatenate((np.zeros((1, 10, 2)), rng.uniform(-1, 1, (5, 10, 2))))
    noise = rng.standard_normal((4, 2, 3, 16, 10, 2))
    means = rng.uniform(-1, 1, (4, 10, 2))
    bank = np.array([[m * np.cos(k*np.pi/4), m * np.sin(k*np.pi/4)]
                     for k in range(8) for m in (.25, .5, .75, 1.)])

    def step(sim, action):
        sim.data.qvel[:] = np.clip(sim.data.qvel, -5, 5)
        sim.forward()
        sim.data.ctrl[:] = np.clip(action, -1, 1)
        for _ in range(5):
            sim.step()

    def replay(start, actions):
        sim = mujoco_py.MjSim(model)
        sim.data.qpos[:], sim.data.qvel[:] = start, 0
        sim.forward()
        states = [np.r_[sim.data.qpos[:2], sim.data.qvel[:2]].copy()]
        for action in actions:
            step(sim, action)
            states.append(np.r_[sim.data.qpos[:2], sim.data.qvel[:2]].copy())
        return np.asarray(states)

    def greedy(start):
        actions = []
        for _ in range(5):
            current = replay(start, actions)[-1]
            scores = []
            for action in bank:
                sim = mujoco_py.MjSim(model)
                sim.data.qpos[:], sim.data.qvel[:] = current[:2], current[2:]
                sim.forward()
                step(sim, action)
                scores.append(float(np.linalg.norm(sim.data.qpos[:2] - goal)))
            actions.append(bank[int(np.argmin(scores))])
        return np.asarray(actions)

    rows = []
    for root, start in enumerate(request["starts"]):
        start = np.asarray(start)
        prefixes = [("zero", np.zeros((5, 2))), ("direct", greedy(start))]
        prefixes += [(f"direction_{k}", np.tile([np.cos(k*np.pi/4), np.sin(k*np.pi/4)], (5, 1)))
                     for k in range(8)]
        prefixes += [(f"random_{k}", bridges[k]) for k in range(2)]
        for name, prefix in prefixes:
            initial = float(np.linalg.norm(start-goal))
            states = replay(start, prefix)
            prefix_distance = float(np.linalg.norm(states[-1, :2]-goal))

            def score(suffix):
                return float(np.linalg.norm(replay(start, np.concatenate((prefix, suffix)))[-1, :2]-goal))
            fixed = [score(suffix) for suffix in common]
            evaluations = []
            for start_idx, loc in enumerate((np.zeros((10, 2)), means[root])):
                loc = loc.copy()
                scale = np.full((10, 2), .5)
                for generation in range(3):
                    samples = np.clip(loc + noise[root, start_idx, generation] * scale, -1, 1)
                    scores = []
                    for sample in samples:
                        value = score(sample)
                        scores.append(value)
                        evaluations.append({"suffix": sample.tolist(), "distance": value})
                    elites = np.argsort(scores, kind="stable")[:4]
                    loc = samples[elites].mean(axis=0)
                    scale = np.maximum(samples[elites].std(axis=0, ddof=1), .02)
                for final in (loc, np.zeros((10, 2))):
                    evaluations.append({"suffix": final.tolist(), "distance": score(final)})
            best = int(np.argmin([item["distance"] for item in evaluations]))
            rows.append({"root_index": root, "start": start.tolist(), "prefix_name": name,
                         "prefix_actions": prefix.tolist(), "prefix_states": states.tolist(),
                         "initial_distance": initial, "prefix_distance": prefix_distance,
                         "non_improving": prefix_distance >= initial,
                         "common_distances": fixed, "common_change_vs_zero": [x-fixed[0] for x in fixed],
                         "search": {"evaluations": evaluations, "best_index": best,
                                    "best_distance": evaluations[best]["distance"]}})
        print(f"native root {root+1}/4 complete", flush=True)
    with Path(output_path).open("x") as handle:
        json.dump({"binding": mujoco_py.__version__, "runtime": mujoco_py.get_version(),
                   "common_suffixes": common.tolist(), "rows": rows}, handle, allow_nan=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
