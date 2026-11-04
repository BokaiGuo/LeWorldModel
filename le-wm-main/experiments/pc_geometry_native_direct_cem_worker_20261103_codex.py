"""Privileged full-horizon native-dynamics CEM diagnostic."""
import json
import sys
from pathlib import Path

import mujoco_py
import numpy as np


def main(request, xml, output):
    if mujoco_py.__version__ != "2.1.2.14":
        raise ValueError("wrong MuJoCo binding")
    starts = json.loads(Path(request).read_text())["starts"]
    model = mujoco_py.load_model_from_xml(Path(xml).read_text())
    goal = np.array([1.8, 2.8])
    rng = np.random.default_rng(2026110319)
    means = rng.uniform(-1, 1, (4, 24, 15, 2))
    noise = rng.standard_normal((4, 24, 3, 16, 15, 2))
    rows = []
    for root, start in enumerate(starts):
        evaluations, points = [], []

        def score(actions):
            sim = mujoco_py.MjSim(model)
            sim.data.qpos[:], sim.data.qvel[:] = start, 0
            sim.forward()
            for action in actions:
                sim.data.qvel[:] = np.clip(sim.data.qvel, -5, 5)
                sim.forward()
                sim.data.ctrl[:] = np.clip(action, -1, 1)
                for _ in range(5):
                    sim.step()
            value = float(np.linalg.norm(sim.data.qpos[:2] - goal))
            evaluations.append({"actions": actions.tolist(), "distance": value})
            return value

        for iteration in range(24):
            loc = np.zeros((15, 2)) if iteration == 0 else means[root, iteration].copy()
            std = np.full((15, 2), .5)
            for generation in range(3):
                samples = np.clip(loc + noise[root, iteration, generation] * std, -1, 1)
                scores = np.asarray([score(sample) for sample in samples])
                elite = np.argsort(scores, kind="stable")[:4]
                loc = samples[elite].mean(axis=0)
                std = np.maximum(samples[elite].std(axis=0, ddof=1), .02)
            score(loc)
            score(np.zeros((15, 2)))
            if len(evaluations) in (100, 300, 600, 1200):
                index = int(np.argmin([item["distance"] for item in evaluations]))
                points.append({"evaluations": len(evaluations), "best_index": index,
                               "best_distance": evaluations[index]["distance"]})
        rows.append({"root": root, "start": start, "evaluations": evaluations, "budget_curve": points})
        print(f"direct CEM root {root+1}/4 complete", flush=True)
    with Path(output).open("x") as handle:
        json.dump({"binding": mujoco_py.__version__, "runtime": mujoco_py.get_version(),
                   "rows": rows}, handle, allow_nan=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
