"""Descriptive same-time candidate comparisons; future RGB is not a selector."""
import argparse
import json
import math
from pathlib import Path

from . import p1_candidate_step_rgb_cuda_20261031 as source

NAME = "p1_candidate_step_rgb_analysis_20261031"
INPUT = source.ROOT / "results/p1_candidate_step_rgb_cuda_20261031_RUN01.json"
SHA = "bd320a45486dc6f78223f141681ff69c8d4a28fc2f8204e0ea5d3cdaa930880d"


def analyze(data):
    if data["status"] != "complete" or len(data["rows"]) != 96 or data["P1_admission"] is not False:
        raise ValueError("fixed result roster")
    groups = {}
    for row in data["rows"]:
        key = (row["index"], row["candidate"])
        if key in groups or not 0 <= row["index"] < 24 or not 0 <= row["candidate"] < 4:
            raise ValueError("candidate identity")
        if [s["step"] for s in row["steps"]] != list(range(1, len(row["steps"])+1)):
            raise ValueError("step roster")
        if any(not math.isfinite(s["actual_goal_mse"]) for s in row["steps"]):
            raise ValueError("nonfinite score")
        groups[key] = row
    if set(groups) != {(i, j) for i in range(24) for j in range(4)}:
        raise ValueError("candidate coverage")
    comparisons = []
    for i in range(24):
        candidates = [groups[i, j] for j in range(4)]
        winners = [row for row in candidates if row["kind"] == 0]
        if len(winners) != 1:
            raise ValueError("one fixed winner")
        winner = winners[0]
        for row in candidates:
            if row["kind"] == 0 or not row["success"]:
                continue
            steps = [s["step"] for s in row["steps"] if s["terminated"] and not s["truncated"]]
            if len(steps) != 1 or steps[0] != len(row["steps"]):
                raise ValueError("success termination")
            step = steps[0]
            same_time = winner["steps"][step-1] if len(winner["steps"]) >= step else None
            success_frame = row["steps"][step-1]
            comparisons.append({"index": i, "role": row["role"], "logical_pair": row["logical_pair"],
                                "winner": winner["candidate"], "success_candidate": row["candidate"],
                                "success_step": step, "winner_action_sha256": winner["action_sha256"],
                                "success_action_sha256": row["action_sha256"],
                                "winner_model_terminal_score": winner["model_terminal_score"],
                                "success_model_terminal_score": row["model_terminal_score"],
                                "winner_real_same_time_goal_mse": None if same_time is None else same_time["actual_goal_mse"],
                                "success_real_goal_mse": success_frame["actual_goal_mse"],
                                "winner_stopped_before": same_time is None})
    return {"schema": NAME, "comparisons": comparisons, "source_sha256": SHA,
            "P1_admission": False, "independent_confirmation": False,
            "limitations": "Post-hoc successful candidates, real future RGB not deployable for selection; "
                           "same-root repeated candidates are not independent."}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    out = Path(p.parse_args().output)
    if not out.is_absolute() or out.parent != source.ROOT / "results" or \
            not out.name.startswith(NAME + "_") or out.exists() or out.is_symlink():
        raise ValueError("exclusive output")
    if source.base.sha(INPUT) != SHA:
        raise ValueError("source hash")
    result = analyze(json.loads(INPUT.read_bytes()))
    result["analysis_sha256"] = source.base.sha(Path(__file__).resolve())
    if source.base.sha(INPUT) != SHA:
        raise ValueError("source drift")
    digest = source.base.write(out, result)
    print(json.dumps({"comparisons": len(result["comparisons"]), "sha256": digest}), flush=True)


if __name__ == "__main__":
    main()
