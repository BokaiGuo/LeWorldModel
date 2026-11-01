"""Link the complete common-suffix panel to verified midpoint-feedback outcomes."""
import argparse
import json
from pathlib import Path

from . import p1_response_h10_control_binding_20261031 as binding

ROOT, need, sha = binding.panel.ROOT, binding.need, binding.sha
NAME = "p1_response_integrated_evidence_20261031"
FEEDBACK = ROOT / "results/p1_response_midpoint_controls_20261031_RUN01.json"
FEEDBACK_SHA = "3582b0b7896d3bfda92bdf35c137455c24a4f880cb5f30c58dd201781921de3b"


def integrate(panel, feedback):
    need(len(panel["rows"]) == 588 and panel["fixed_prefixes"] == 16
         and panel["fixed_branches"] == 588 and panel["P1_admission"] is False,
         "complete H10 panel")
    need(feedback["fixed_rows"] == 16 and feedback["P1_admission"] is False
         and len(feedback["rows"]) == 13 and feedback["excluded"] == [1, 3, 5],
         "feedback roster")
    mapped = {row["index"]: row for row in feedback["rows"]}
    need(len(mapped) == 13, "unique feedback")
    anchors = {}
    for row in panel["rows"]:
        index = row["anchor"]
        entry = anchors.setdefault(index, {"logical_pair": row["logical_pair"],
                                           "role": row["role"], "root_slot": row["root_slot"],
                                           "condition": row["condition"], "arm": row["arm"],
                                           "probe_indices": []})
        need(all(entry[k] == row[k] for k in ("logical_pair", "role", "root_slot", "condition", "arm")),
             "anchor consistency")
        entry["probe_indices"].append(row["probe_index"])
    need(len(anchors) == 14 and set(anchors) == set(range(16)) - {1, 5}
         and all(item["probe_indices"] == list(range(42)) for item in anchors.values()), "panel roster")
    need(set(mapped) == set(anchors) - {3}, "feedback coverage")
    outcomes = []
    for index, info in sorted(anchors.items()):
        value = mapped.get(index)
        if value is None:
            outcomes.append({"anchor": index, "status": "excluded_prior_midpoint_physical_incomplete",
                             "feedback": None})
            continue
        need(all(info[k] == value[k] for k in ("logical_pair", "role", "root_slot", "condition", "arm")),
             "feedback task identity")
        outcomes.append({"anchor": index, "status": "measured", "feedback": value})
    return {"schema": NAME, "common_suffix_rows": panel["rows"], "feedback_by_anchor": outcomes,
            "candidate_comparisons": panel["candidate_comparisons"], "censored_before_cp5": [1, 5],
            "fixed_branches": 588, "fixed_feedback_roster": 16,
            "limitations": "Known roots and exploratory selection; model_selected is not a certified "
            "deployment winner. Common suffix and individually optimized feedback answer different "
            "questions; shared rows are not independent samples and total budget is not certified.",
            "P1_admission": False, "independent_confirmation": False, "equal_budget_certified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--panel-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    panel_path, out = Path(args.panel), Path(args.output)
    need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_")
         and not out.exists() and not out.is_symlink(), "exclusive output")
    need(sha(panel_path) == args.panel_sha and sha(FEEDBACK) == FEEDBACK_SHA, "source hashes")
    panel, feedback = json.loads(panel_path.read_bytes()), json.loads(FEEDBACK.read_bytes())
    need(panel["input_sha256"]["prepare"] == binding.NEW_SHA, "sealed panel source")
    result = integrate(panel, feedback)
    need(sha(panel_path) == args.panel_sha and sha(FEEDBACK) == FEEDBACK_SHA, "source drift")
    result["input_sha256"] = {"panel": args.panel_sha, "feedback": FEEDBACK_SHA, "code": sha(__file__)}
    with out.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"rows": len(result["common_suffix_rows"]), "sha256": sha(out)}), flush=True)


if __name__ == "__main__":
    main()
