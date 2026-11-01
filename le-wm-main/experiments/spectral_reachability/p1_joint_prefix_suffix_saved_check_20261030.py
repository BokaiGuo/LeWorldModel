"""Check saved joint-refinement arrays against source rollout and selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_joint_prefix_suffix_cuda_20261030 as runner
from . import p1_backtrack_action_inventory_20261030 as inventory

NAME = "p1_joint_prefix_suffix_saved_check_20261030"
ROOT = runner.ROOT
REPORT_SHA = "de579ff2c8e4873e74dfa89bf0ebf86f4af8fc1a6803bbe904f6427a95a94d54"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(path):
    require(sha(path) == REPORT_SHA, "run hash")
    result = json.loads(path.read_bytes())
    source, saved = runner.base.one.source, runner.base.one.saved
    source_report = saved.read_verified(source.REPORT, source.REPORT_SHA)
    prepared = saved.read_verified(source_report["prepared"], source_report["prepare_sha256"])
    backtrack = saved.read_verified(ROOT / "results/p1_bounded_backtrack_host_20261030_MAIN01.json", inventory.SHA)
    inventory.verifier.verify(backtrack, source_report, prepared)
    require(len(result["rows"]) == 160 and result["calls"] ==
            {"predict": 2400, "action_encoder": 2400, "reverse": 640}
            and result["model_before"] == result["model_after"] == source_report["model_before"]
            and not result["P1_admission"] and not result["physical_release"], "run roster")
    improved = changed = monotonic = 0
    winners = []
    for arm_index, arm in enumerate(source_report["rows"]):
        goal = np.asarray(prepared["requests"][arm_index // 2]["task"]["goal"], np.float32)
        group = result["rows"][20 * arm_index:20 * (arm_index + 1)]
        for i, (row, candidate) in enumerate(zip(group, arm["result"]["candidates"], strict=True)):
            continuation = backtrack["rows"][arm_index * 20 + i]
            require((row["slot"], row["arm"], row["candidate_index"], row["start"], row["iteration"])
                    == (arm["slot"], arm["arm"], i, candidate["start"], candidate["iteration"]),
                    "candidate identity")
            trajectory = row["trajectory"]
            require(len(trajectory) == 5 and [x["iteration"] for x in trajectory] == list(range(5)),
                    "iteration roster")
            initial = np.asarray(trajectory[0]["actions"], np.float32)
            expected = np.concatenate((np.asarray(candidate["prefix"], np.float32),
                                       np.asarray(continuation["final"]["raw"], np.float32).reshape(10, 2)))
            require(initial.shape == (3, 5, 2) and initial.reshape(15, 2).tobytes() == expected.tobytes()
                    and np.array_equal(np.asarray(trajectory[0]["prefix_latent"], np.float32),
                                       np.asarray(candidate["mid"]["latent_history"][-1], np.float32))
                    and np.array_equal(np.asarray(trajectory[0]["terminal_latent"], np.float32),
                                       np.asarray(continuation["final"]["endpoint"], np.float32)),
                    "source rollout binding")
            scores = []
            for step in trajectory:
                action = np.asarray(step["actions"], np.float32)
                end = np.asarray(step["terminal_latent"], np.float32)
                require(action.shape == (3, 5, 2) and np.isfinite(action).all() and (abs(action) <= 1).all()
                        and end.shape == (192,) and np.isfinite(end).all(), "finite action/endpoint")
                score = float(np.mean((end - goal) ** 2))
                require(np.isclose(score, step["score"], rtol=2e-6, atol=2e-7), "score binding")
                scores.append(step["score"])
            best = min(range(5), key=lambda n: (scores[n], n))
            require(row["best_iteration"] == best and row["best_score"] == scores[best]
                    and row["best_actions"] == trajectory[best]["actions"], "best binding")
            improved += scores[best] < scores[0]
            changed += np.asarray(row["best_actions"], np.float32)[0].tobytes() != initial[0].tobytes()
            monotonic += all(a >= b for a, b in zip(scores, scores[1:]))
        best_row = min(group, key=lambda r: (r["best_score"], r["candidate_index"]))
        winners.append({"slot": best_row["slot"], "arm": best_row["arm"],
                        "candidate_index": best_row["candidate_index"], "best_iteration": best_row["best_iteration"],
                        "best_score": best_row["best_score"]})
    require(winners == result["winners"] and sha(path) == REPORT_SHA, "winner/no drift")
    return {"schema": NAME, "report_sha256": REPORT_SHA, "candidates": 160,
            "initial_rollout_exact": 160, "best_prefix_changed": changed, "best_score_improved": improved,
            "fully_monotonic_scores": monotonic, "winners": winners, "P1_admission": False,
            "scope": "saved arrays, score and selection; gradients and runtime hooks not independently attested"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    require(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_")
            and not out.exists(), "exclusive output")
    result = check(ROOT / "results/p1_joint_prefix_suffix_cuda_20261030_RUN01.json")
    result["code_sha256"] = sha(Path(__file__))
    content = json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    with out.open("xb") as stream:
        require(stream.write(content) == len(content), "short write")
    print(json.dumps({"sha256": hashlib.sha256(content).hexdigest(), **result}))


if __name__ == "__main__":
    main()
