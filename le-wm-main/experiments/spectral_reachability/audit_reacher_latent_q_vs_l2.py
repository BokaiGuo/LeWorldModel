"""Fixed saved-data comparison of Q and unweighted latent goal distances."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup

REPLAY_SHA = "4a0c2774ee5569476c668301a53ba8ff6ea183b6a1c3ed1255637f2a65d0f27e"
GOAL_SHA = "5b36c1ac7b35923fcd8f5d28e22340371ab9eb67d96acd72644cb573d8e6f066"


def compare(a, b):
    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("nonfinite comparison")
    return "tie" if a == 0 or b == 0 else "reversal" if a * b < 0 else "agreement"


def analyze(row, goal):
    target = np.asarray(goal["target_coordinate"], dtype=np.float64)
    if (row["seed"] != goal["seed"] or target.shape != (192,) or not np.isfinite(target).all()):
        raise ValueError("goal identity or shape mismatch")
    branches = {(b["prefix_index"], b["tail_index"]): b for b in row["branches"]}
    if (len(row["branches"]) != 35 or set(branches) != {(i, j) for i in range(7) for j in range(5)}
            or any(b["status"] != "complete" or len(b["checkpoints"]) != 3 for b in branches.values())):
        raise ValueError("incomplete or duplicate branch")
    records = []
    counts = {name: {"agreement": 0, "reversal": 0, "tie": 0}
              for name in ("q_prediction_vs_rgb", "l2_prediction_vs_rgb", "q_rgb_vs_physical", "l2_rgb_vs_physical")}
    for i in range(1, 7):
        for j in range(5):
            a = branches[(i, j)]["checkpoints"][2]
            b = branches[(0, j)]["checkpoints"][2]
            if a["step"] != 15 or b["step"] != 15:
                raise ValueError("checkpoint time mismatch")
            def l2(entry, name):
                latent = np.asarray(entry[name], dtype=np.float64)
                if latent.shape != target.shape or not np.isfinite(latent).all():
                    raise ValueError("latent shape or finite mismatch")
                return float(np.mean((latent - target) ** 2))
            delta = {"pred_q": a["open_goal_q"] - b["open_goal_q"],
                     "rgb_q": a["observed_goal_q"] - b["observed_goal_q"],
                     "pred_l2": l2(a, "open_latent") - l2(b, "open_latent"),
                     "rgb_l2": l2(a, "observed_latent") - l2(b, "observed_latent"),
                     "physical": a["qpos_distance"] - b["qpos_distance"]}
            pairs = {"q_prediction_vs_rgb": ("pred_q", "rgb_q"),
                     "l2_prediction_vs_rgb": ("pred_l2", "rgb_l2"),
                     "q_rgb_vs_physical": ("rgb_q", "physical"),
                     "l2_rgb_vs_physical": ("rgb_l2", "physical")}
            labels = {name: compare(delta[left], delta[right]) for name, (left, right) in pairs.items()}
            for name, label in labels.items():
                counts[name][label] += 1
            records.append({"prefix_index": i, "tail_index": j, "deltas": delta, "comparisons": labels})
    return {"seed": row["seed"], "counts": counts, "paired_contrasts": records}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--goal-replay", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if setup._sha256_file(args.replay) != REPLAY_SHA or setup._sha256_file(args.goal_replay) != GOAL_SHA:
        raise ValueError("source SHA mismatch")
    replay, goal = json.loads(args.replay.read_text()), json.loads(args.goal_replay.read_text())
    if not replay["frozen_parameters_unchanged"] or not goal["frozen_parameters_unchanged"]:
        raise ValueError("model parameter postcheck failed")
    if replay["parameter_hash_before"] != goal["parameter_hash_before"]:
        raise ValueError("checkpoint mismatch")
    goals = {r["seed"]: r for r in goal["rows"]}
    if [r["seed"] for r in replay["rows"]] != [r["seed"] for r in goal["rows"]]:
        raise ValueError("root order mismatch")
    if replay["preflight_sha256"] != goal["preflight_sha256"]:
        raise ValueError("goal preflight identity mismatch")
    pf_path = (Path(__file__).resolve().parents[2] /
               "results/reacher_pc_p1_common_tail_preflight_20261102_codexexclusive_02.json")
    if setup._sha256_file(pf_path) != replay["preflight_sha256"]:
        raise ValueError("preflight SHA mismatch")
    expected_goals = {r["seed"]: r["goal_rgb_sha256"] for r in json.loads(pf_path.read_text())["roots"]}
    if any(r["goal_rgb_sha256"] != expected_goals[r["seed"]] for r in goal["rows"]):
        raise ValueError("goal RGB hash mismatch")
    rows = [analyze(r, goals[r["seed"]]) for r in replay["rows"]]
    result = {"scope": "saved outcome-exposed development comparisons, no new inference or selection",
              "replay_sha256": REPLAY_SHA, "goal_replay_sha256": GOAL_SHA,
              "script_sha256": setup._sha256_file(Path(__file__)), "rows": rows,
              "independent_confirmation": False, "P1_admission": False}
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": setup._sha256_file(args.output)}))


if __name__ == "__main__":
    main()
