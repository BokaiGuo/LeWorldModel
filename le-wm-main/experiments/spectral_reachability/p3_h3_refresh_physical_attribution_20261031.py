"""Join fixed P3 H3 real-context predictions to the same physical action pairs."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from . import p3_h3_model_physical_direction_20261031 as original

ROOT = original.ROOT
NAME = "p3_h3_refresh_physical_attribution_20261031"
REPLAY = ROOT / "results/p3_h3_real_context_replay_20261031_RUN01.json"
DIRECTION = ROOT / "results/p3_h3_model_physical_direction_20261031_RUN01.json"
SOURCES = {
    REPLAY: "9ab124dff2aa202552acc393e6534f1464d7eb2ea6615988852cd4d981171a1f",
    DIRECTION: "ed4b92f97af2e761a0ae58bded7d838693b85a8285098418a3d34a352585ee6e",
}
require, sign = original.require, original.sign


def analyze(replay, physical):
    require(len(replay["rows"]) == 52 and len(physical["groups"]) == 18,
            "fixed source roster")
    by_key = {(row["logical_pair"], row["variant"]): row for row in replay["rows"]}
    require(len(by_key) == 52, "unique replay identities")
    records = []
    for group in physical["groups"]:
        raw = group["representative"]
        if raw["physical_new_minus_old"] is None:
            continue
        pair = raw["logical_pair"]
        a, b = by_key[(pair, "old")], by_key[(pair, "new")]
        require(all(a[k] == b[k] == raw[k] for k in
                    ("decision_index", "candidate", "policy", "root_slot")),
                "task identity")
        values = {key: b[key] - a[key] for key in
                  ("open_goal_cost", "refreshed_goal_cost", "actual_goal_cost")}
        require(abs(values["open_goal_cost"] - raw["model_new_minus_old"]) < 1e-5,
                "open model score binding")
        records.append({"logical_pair": pair, "root_slot": raw["root_slot"],
                        "candidate": raw["candidate"], "policy": raw["policy"],
                        "prediction_differences": values,
                        "physical_differences": raw["physical_new_minus_old"]})
    require(len(records) == 13, "complete pair denominator")
    counts = {}
    for source in ("open_goal_cost", "refreshed_goal_cost", "actual_goal_cost"):
        counts[source] = {}
        for metric in original.METRICS:
            count = Counter(sign(r["prediction_differences"][source]) + "/" +
                            sign(r["physical_differences"][metric]) for r in records)
            counts[source][metric] = dict(count)
    return {"schema": NAME, "complete_action_pairs": 13, "records": records, "counts": counts,
            "limitations": "Post-hoc known roots, repeated actions, no independent samples. "
            "Prediction, real encoded distance and physical metrics are different quantities; "
            "this is not a control experiment or evidence of a budget advantage.",
            "P1_admission": False, "P3_admission": False, "independent_confirmation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    require(output.is_absolute() and output.parent == ROOT / "results"
            and output.name.startswith(NAME + "_") and not output.exists()
            and not output.is_symlink(), "exclusive output")
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    require(all(digest(path) == expected for path, expected in SOURCES.items()), "source hashes")
    result = analyze(*(json.loads(path.read_text()) for path in SOURCES))
    require(all(digest(path) == expected for path, expected in SOURCES.items()), "source drift")
    result["source_sha256"] = {str(path): expected for path, expected in SOURCES.items()}
    result["analysis_sha256"] = digest(Path(__file__))
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"sha256": digest(output), "counts": result["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
