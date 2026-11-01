"""Retrospectively compare fixed predicted L10 and max(L5, L10) selectors."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from . import p1_candidate_short_horizon_alignment_20261031 as binding

NAME = "p1_worst_two_time_selector_20261031"
ROOT = binding.ROOT


def analyze(mid, actual):
    # Reuse the independent physical-frame, score, candidate, and winner bindings.
    checked = binding.analyze(mid, actual)
    by_key = {(row["index"], row["candidate"]): row for row in mid["rows"]}
    prior = {row["index"]: row for row in checked["selectors"]}
    summary = {role: Counter() for role in ("selected", "zero", "greedy")}
    rows = []
    for index in range(24):
        choices = [by_key[index, candidate] for candidate in range(4)]
        for choice in choices:
            if not all(math.isfinite(choice[k]) and choice[k] >= 0
                       for k in ("predicted_midpoint_goal_mse", "open_terminal_goal_mse")):
                raise ValueError("invalid model score")
        original = min(choices, key=lambda x: (x["open_terminal_goal_mse"], x["candidate"]))
        robust = min(choices, key=lambda x: (
            max(x["predicted_midpoint_goal_mse"], x["open_terminal_goal_mse"]),
            x["candidate"]))
        role = original["role"]
        if role not in summary or any(c["role"] != role for c in choices):
            raise ValueError("role identity")
        if original["candidate"] != prior[index]["open10"]:
            raise ValueError("original winner identity")
        counts = summary[role]
        counts["records"] += 1
        counts["original_success"] += int(original["success"])
        counts["worst_two_time_success"] += int(robust["success"])
        counts["changed"] += int(original["candidate"] != robust["candidate"])
        rows.append({"index": index, "role": role,
                     "logical_pair": original["logical_pair"],
                     "original_candidate": original["candidate"],
                     "worst_two_time_candidate": robust["candidate"],
                     "original_success": original["success"],
                     "worst_two_time_success": robust["success"],
                     "scores": [{"candidate": c["candidate"],
                                 "predicted5": c["predicted_midpoint_goal_mse"],
                                 "open10": c["open_terminal_goal_mse"],
                                 "success": c["success"]} for c in choices]})
    if any(summary[r]["records"] != 8 for r in summary):
        raise ValueError("role roster")
    return {"schema": NAME, "status": "complete",
            "selector": "min(max(predicted_midpoint_goal_mse, open_terminal_goal_mse), candidate_index)",
            "summary": {r: dict(c) for r, c in summary.items()}, "rows": rows,
            "P1_admission": False, "independent_confirmation": False,
            "equal_budget_certified": False,
            "limitations": "Post-hoc known-root selection on previously executed candidates; "
                           "not an online experiment, independent confirmation or cost comparison."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    if not out.is_absolute() or out.parent != ROOT / "results" or not out.name.startswith(NAME + "_") \
            or out.exists() or out.is_symlink():
        raise ValueError("exclusive output")
    inputs = ((binding.MID, binding.MID_SHA), (binding.real.INPUT, binding.real.SHA))
    if any(binding.prior.base.sha(path) != sha for path, sha in inputs):
        raise ValueError("input hash")
    print("Compare fixed L10 and worst-of-L5/L10 model selectors on all saved physical candidates.", flush=True)
    result = analyze(*(json.loads(path.read_bytes()) for path, _ in inputs))
    result["sources"] = {str(path): sha for path, sha in inputs}
    result["code_sha256"] = binding.prior.base.sha(Path(__file__).resolve())
    if any(binding.prior.base.sha(path) != sha for path, sha in inputs):
        raise ValueError("source drift")
    digest = binding.prior.base.write(out, result)
    print(json.dumps({"summary": result["summary"], "sha256": digest}), flush=True)


if __name__ == "__main__":
    main()
