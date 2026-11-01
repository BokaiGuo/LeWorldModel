"""Summarize pinned real-context CUDA replay without treating repeats as roots."""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from . import p3_h3_model_physical_direction_20261031 as prior

ROOT = prior.ROOT
NAME = "p3_h3_real_context_analysis_20261031"
REPLAY = ROOT / "results/p3_h3_real_context_replay_20261031_RUN01.json"
REPLAY_SHA = "9ab124dff2aa202552acc393e6534f1464d7eb2ea6615988852cd4d981171a1f"
GROUPS = ROOT / "results/p3_h3_model_physical_direction_20261031_RUN01.json"
GROUPS_SHA = "ed4b92f97af2e761a0ae58bded7d838693b85a8285098418a3d34a352585ee6e"


def summarize(replay, groups):
    need = prior.require
    need(replay["model_before"] == replay["model_after"] and len(replay["rows"]) == 52
         and len(replay["excluded"]) == 10 and len(groups["groups"]) == 18, "fixed roster")
    rows = {(row["logical_pair"], row["variant"]): row for row in replay["rows"]}
    need(len(rows) == 52, "unique row identity")
    chosen, comparison = [], Counter()
    keys = ("open_goal_cost", "refreshed_goal_cost", "actual_goal_cost", "open_terminal_mse",
            "refreshed_terminal_mse", "open_latent", "refreshed_latent", "actual_latent", "goal_latent")
    for group in groups["groups"]:
        if group["representative"]["physical_new_minus_old"] is None:
            continue
        one, two = [r["logical_pair"] for r in group["duplicates"]]
        for variant in ("old", "new"):
            a, b = rows[(one, variant)], rows[(two, variant)]
            need(all(a[key] == b[key] for key in keys), "duplicate replay differs")
            chosen.append(a)
        pair = [rows[(one, variant)] for variant in ("old", "new")]
        direction = tuple(prior.sign(pair[1][k] - pair[0][k]) for k in
                          ("open_goal_cost", "refreshed_goal_cost", "actual_goal_cost"))
        comparison["/".join(direction)] += 1
    need(len(chosen) == 26 and sum(comparison.values()) == 13, "unique complete denominator")
    return {"schema": NAME, "unique_complete_action_pairs": 13, "unique_complete_trajectories": 26,
            "logical_complete_pairs": 26, "censored_logical_pairs": 10,
            "mean_open_terminal_mse": statistics.mean(r["open_terminal_mse"] for r in chosen),
            "mean_refreshed_terminal_mse": statistics.mean(r["refreshed_terminal_mse"] for r in chosen),
            "refreshed_lower_mse": sum(r["refreshed_terminal_mse"] < r["open_terminal_mse"] for r in chosen),
            "directions_open_refreshed_actual": dict(comparison),
            "limitations": "Post-hoc known roots; duplicate action pairs, not independent roots. "
                           "Real history replacement is not a closed-loop control outcome.",
            "P1_admission": False, "P3_admission": False, "independent_confirmation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    output = Path(parser.parse_args().output)
    prior.require(output.is_absolute() and output.parent == ROOT / "results"
                  and output.name.startswith(NAME + "_") and not output.exists()
                  and not output.is_symlink(), "exclusive output")
    sources = ((REPLAY, REPLAY_SHA), (GROUPS, GROUPS_SHA))
    prior.require(all(prior.digest(p) == h for p, h in sources), "source hash")
    result = summarize(*(json.loads(p.read_text()) for p, _ in sources))
    prior.require(all(prior.digest(p) == h for p, h in sources), "source drift")
    result["sources"] = {str(p): h for p, h in sources}
    result["analysis_sha256"] = prior.digest(Path(__file__))
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"sha256": prior.digest(output), **{k: result[k] for k in (
        "unique_complete_action_pairs", "mean_open_terminal_mse", "mean_refreshed_terminal_mse",
        "refreshed_lower_mse", "directions_open_refreshed_actual")}}))


if __name__ == "__main__":
    main()
