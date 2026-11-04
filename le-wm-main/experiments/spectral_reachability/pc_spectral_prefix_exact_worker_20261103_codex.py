"""Privileged fixed-bank MuJoCo 2.2 CEM evaluator; never a planner input."""
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from audit_sra_pointmaze_replay_backend import replay


def main(request_path, xml_path, output_path):
    if output_path.exists():
        raise FileExistsError(output_path)
    if mujoco.__version__ != "2.2.0":
        raise RuntimeError("MuJoCo 2.2.0 required")
    request = json.loads(request_path.read_text())
    starts, prefixes = np.asarray(request["starts"]), np.asarray(request["prefixes"])
    mean, std, goal = np.asarray(request["action_mean"]), np.asarray(request["action_std"]), np.asarray(request["goal"])
    if starts.shape != (16, 4) or prefixes.shape != (16, 17, 2) or mean.shape != (2,) or std.shape != (2,) or np.any(std <= 0):
        raise ValueError("invalid request")
    model = mujoco.MjModel.from_xml_string(xml_path.read_text())
    mean10, std10 = np.tile(mean, 5), np.tile(std, 5)
    lower, upper, prior = (-1-mean10)/std10, (1-mean10)/std10, -mean10/std10
    noises = [np.random.default_rng(seed).standard_normal((2, 3, 16, 10))
              for seed in (2026110810, 2026110820)]
    rows = []
    for index, (start, prefix) in enumerate(zip(starts, prefixes)):
        def score(normalized):
            raw = (normalized * std10 + mean10).reshape(5, 2)
            return float(np.linalg.norm(replay(model, start, np.concatenate((prefix, raw)))[:2] - goal))
        prefix_distances = [float(np.linalg.norm(replay(model, start, prefix[:tick])[:2] - goal))
                            for tick in range(12, 18)]
        searches = []
        for start_index in range(2):
            loc, scale, traces = prior.copy(), np.full(10, 1/3), []
            for iteration in range(3):
                samples = np.clip(loc + noises[index // 8][start_index, iteration] * scale, lower, upper)
                samples[0] = np.clip(loc, lower, upper)
                scores = np.asarray([score(sample) for sample in samples])
                elite = np.argsort(scores, kind="stable")[:4]
                loc = samples[elite].mean(axis=0)
                scale = np.maximum(samples[elite].std(axis=0, ddof=1), 1e-5)
                traces.append({"samples": samples.tolist(), "scores": scores.tolist(),
                               "elite": elite.tolist(), "mean": loc.tolist(), "std": scale.tolist()})
            finals = np.stack((prior, loc)).clip(lower, upper)
            scores = [score(sample) for sample in finals]
            winner = int(scores[1] < scores[0])
            searches.append({"finals": finals.tolist(), "scores": scores, "winner": winner, "trace": traces})
        winner = int(min(searches[1]["scores"]) < min(searches[0]["scores"]))
        chosen = searches[winner]
        final_normalized = np.asarray(chosen["finals"][chosen["winner"]])
        rows.append({"index": index, "prefix_distances": prefix_distances,
                     "optimized_distance": float(chosen["scores"][chosen["winner"]]),
                     "optimized_raw_suffix": (final_normalized * std10 + mean10).reshape(5, 2).tolist(),
                     "winning_start": winner, "samples": 100, "searches": searches})
        print(f"fixed prefix {index + 1}/16 complete", flush=True)
    with output_path.open("x") as out:
        json.dump({"mujoco_version": mujoco.__version__, "rows": rows}, out, allow_nan=False)


if __name__ == "__main__":
    main(*(Path(x) for x in sys.argv[1:]))
