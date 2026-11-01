"""Exploratory same-action comparison of sealed P3 H3 model and physical pairs."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAME = "p3_h3_model_physical_direction_20261031"
SOURCES = {
    "prepare": ("p3_raw_h3_physical_prepare_20261031_RUN01.json",
                "3f2b8c34defb942242d2f66d1b46f83251125f07b871440b2ec6d8acde1189fd"),
    "model": ("p3_raw_chart_nonlinear_20261030_main_RUN01.json",
              "44cc243acc1c260ff84b9263227cad14508d8704cd5292244b894eba9b0d7839"),
    "physical": ("p3_raw_h3_panel_summary_20261031_RUN02.json",
                 "a55a291d387c32e33b6104f4b6431da90fb1f499ea661be709fd51e22b29e211"),
}
METRICS = ("object_position_error", "wrapped_object_angle_error", "native_distance")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sign(value):
    return "lower" if value < -1e-9 else "higher" if value > 1e-9 else "tie"


def analyze(prepared, model, physical):
    plans = prepared["plans"]
    rows = [r for r in model["rows"] if r["horizon"] == 3]
    require(len(plans) == 72 and len(rows) == 36 and len(physical["groups"]) == 18,
            "fixed rosters")
    model_by_key = {(r["decision_index"], r["candidate"], r["policy"], r["root"]): r for r in rows}
    require(len(model_by_key) == 36, "unique model identities")
    result = []
    for group in physical["groups"]:
        ids = group["logical_pair_indices"]
        require(len(ids) == 2 and group["outcomes_exact"] is True, "duplicate group")
        comparisons = []
        for logical, outcome in zip(ids, group["records"], strict=True):
            old, new = plans[2 * logical:2 * logical + 2]
            require(old["variant"] == "old" and new["variant"] == "new"
                    and all(old[k] == new[k] for k in ("candidate", "decision_index", "policy",
                                                         "root_slot", "task_id", "root_sha256")),
                    "paired plan identity")
            mr = model_by_key[(old["decision_index"], old["candidate"], old["policy"], old["root_slot"])]
            for variant, plan in (("old", old), ("new", new)):
                actual = np.asarray(plan["commands"], dtype="<f4")
                imagined = np.asarray(mr[variant + "_action"], dtype="<f4")
                require(actual.shape == (20, 2) and imagined.shape == (5, 2)
                        and actual[15:].tobytes() == imagined.tobytes(), "exact suffix action")
            comparable = outcome["old_status"] == outcome["new_status"] == "horizon_completed"
            require(not comparable or outcome["step15_exact"] is True, "actual shared prefix")
            differences = outcome["new_minus_old_endpoint"] if comparable else None
            comparisons.append({
                "logical_pair": logical, "decision_index": old["decision_index"],
                "candidate": old["candidate"], "policy": old["policy"], "root_slot": old["root_slot"],
                "model_new_minus_old": mr["new_cost"] - mr["old_cost"],
                "old_status": outcome["old_status"], "new_status": outcome["new_status"],
                "physical_new_minus_old": differences,
            })
        require(comparisons[0]["model_new_minus_old"] == comparisons[1]["model_new_minus_old"]
                and comparisons[0]["physical_new_minus_old"] == comparisons[1]["physical_new_minus_old"],
                "repeat outcome agreement")
        result.append({"duplicates": comparisons, "representative": comparisons[0]})
    counts = {metric: Counter() for metric in METRICS}
    for group in result:
        row = group["representative"]
        if row["physical_new_minus_old"] is None:
            continue
        for metric in METRICS:
            counts[metric][sign(row["model_new_minus_old"]) + "/" +
                           sign(row["physical_new_minus_old"][metric])] += 1
    return {"schema": NAME, "groups": result, "direction_counts_model_then_physical":
            {metric: dict(value) for metric, value in counts.items()},
            "unique_action_pairs": len(result),
            "complete_pairs": sum(g["representative"]["physical_new_minus_old"] is not None for g in result),
            "limitations": "Post-hoc exploratory comparison; two known roots and duplicated actions. "
                           "Model context is imagined, not real cp15; scores and physical metrics differ. "
                           "Counts are not independent samples or evidence of equal budget.",
            "P1_admission": False, "P3_admission": False, "independent_confirmation": False}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    require(output.is_absolute() and output.parent == ROOT / "results"
            and output.name.startswith(NAME + "_") and not output.exists()
            and not output.is_symlink(), "exclusive output")
    sources = {name: ROOT / "results" / entry[0] for name, entry in SOURCES.items()}
    for name, path in sources.items():
        require(digest(path) == SOURCES[name][1], "source hash")
    payloads = {name: json.loads(path.read_text()) for name, path in sources.items()}
    result = analyze(payloads["prepare"], payloads["model"], payloads["physical"])
    for name, path in sources.items():
        require(digest(path) == SOURCES[name][1], "source drift")
    result["source_sha256"] = {name: SOURCES[name][1] for name in SOURCES}
    result["analysis_sha256"] = digest(Path(__file__))
    with output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output_sha256": digest(output), "complete_pairs": result["complete_pairs"],
                      "counts": result["direction_counts_model_then_physical"]}))


if __name__ == "__main__":
    main()
