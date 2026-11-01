"""Evaluate a fixed rank-based selector on all already-executed candidates."""
import argparse
import json
import math
from pathlib import Path

from . import p1_adam_checkpoint_physical_pair_20261031 as pair

ROOT = pair.ROOT
NAME = "p1_two_horizon_rank_20261031"
SOURCE = ROOT / "results/p1_uphill_all_candidates_midpoint_20261031_RUN02.json"
SOURCE_SHA = "e0e3062b6a0ef46a94b5be0c159e19547da5182745a7a4c343a88c4e6d2f3de0"
PROTOCOL = ROOT.parent / "docs/D2P_P1_TWO_HORIZON_RANK_PROTOCOL_20261031_MAIN.md"
need = pair.pairing.need


def choose(candidate_scores):
    need(len(candidate_scores) == 4 and sorted(x[0] for x in candidate_scores) == list(range(4)),
         "candidate identities")
    need(all(math.isfinite(mid) and math.isfinite(end) for _, mid, end in candidate_scores),
         "finite scores")
    mid_order = sorted(candidate_scores, key=lambda x: (x[1], x[0]))
    end_order = sorted(candidate_scores, key=lambda x: (x[2], x[0]))
    mid_rank = {entry[0]: i for i, entry in enumerate(mid_order)}
    end_rank = {entry[0]: i for i, entry in enumerate(end_order)}
    combined = min(range(4), key=lambda i: (mid_rank[i]+end_rank[i], end_rank[i], i))
    return {"mid": mid_order[0][0], "end": end_order[0][0], "combined": combined}


def analyze(data):
    need(data["status"] == "complete" and len(data["rows"]) == 96
         and data["model_before"] == data["model_after"], "source roster")
    rows = []
    for i in range(24):
        group = [r for r in data["rows"] if r["index"] == i]
        need(len(group) == 4 and len({r["role"] for r in group}) == 1
             and len({r["logical_pair"] for r in group}) == 1, "group identity")
        scores = [(r["candidate"], r["predicted_midpoint_goal_mse"], r["open_terminal_goal_mse"]) for r in group]
        choices = choose(scores)
        selected = {r["candidate"]: r for r in group}
        rows.append({"index": i, "role": group[0]["role"], "logical_pair": group[0]["logical_pair"],
                     "choices": choices,
                     "outcomes": {method: {"success": selected[choice]["success"],
                                           "status": selected[choice]["status"],
                                           "returned_suffix": selected[choice]["returned_suffix"]}
                                  for method, choice in choices.items()}})
    summary = {role: {method: sum(r["outcomes"][method]["success"] is True for r in rows if r["role"] == role)
                      for method in ("end", "mid", "combined")}
               for role in ("selected", "zero", "greedy")}
    return {"schema": NAME, "rows": rows, "summary": summary, "source_sha256": SOURCE_SHA,
            "P1_admission": False, "independent_confirmation": False, "equal_budget_certified": False,
            "limitations": "Post-outcome development on repeated known roots; no new physics or online selection."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME+"_")
         and not out.exists() and not out.is_symlink(), "exclusive output")
    data = pair.pairing.read(SOURCE, SOURCE_SHA)
    result = analyze(data)
    result["protocol_sha256"] = pair.physical.sha(PROTOCOL)
    result["code_sha256"] = pair.physical.sha(Path(__file__).resolve())
    need(pair.physical.sha(SOURCE) == SOURCE_SHA, "source drift")
    digest = pair.physical.prior.physical.design.run._write_new(out, result)
    print(json.dumps({"sha256": digest, "summary": result["summary"]}), flush=True)


if __name__ == "__main__":
    main()
