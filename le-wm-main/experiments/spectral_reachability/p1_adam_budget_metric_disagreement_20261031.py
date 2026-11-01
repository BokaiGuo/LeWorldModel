"""Describe paired model-score and physical-metric direction disagreements."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from . import p1_adam_checkpoint_physical_pair_20261031 as source

NAME = "p1_adam_budget_metric_disagreement_20261031"
INPUT = source.ROOT / "results/p1_adam_checkpoint_physical_pair_20261031_RUN01.json"
INPUT_SHA = "6b1d9fd80a4465a1d9e0cdb6ded0092773ec11b1d19f04217a385e0fc1378051"
METRICS = ("object_position_error", "object_angle_error_wrapped",
           "pusher_position_error", "native_joint_xy_error", "native_angle_term")
EPS = 1e-6  # Post-hoc descriptive numerical tolerance, not a gate.


def direction(value):
    if not math.isfinite(value):
        raise ValueError("nonfinite difference")
    return -1 if value < -EPS else 1 if value > EPS else 0


def analyze(result):
    source.pairing.need(result["schema"] == source.NAME and len(result["rows"]) == 24
                        and not result["P1_admission"] and not result["equal_budget_certified"],
                        "source contract")
    pairs, counts = [], {}
    for left, right in (("5", "10"), ("10", "20"), ("5", "20")):
        for metric in METRICS:
            counts[f"{left}_to_{right}:{metric}"] = Counter()
    for i, row in enumerate(result["rows"]):
        source.pairing.need(row["index"] == i and set(row["budgets"]) == {"5", "10", "20"},
                            "row roster")
        for left, right in (("5", "10"), ("10", "20"), ("5", "20")):
            a, b = row["budgets"][left], row["budgets"][right]
            model_delta = b["model_best_score"] - a["model_best_score"]
            model_direction = direction(model_delta)
            item = {"index": i, "logical_pair": row["logical_pair"], "root_slot": row["root_slot"],
                    "condition": row["condition"], "role": row["role"], "from": left, "to": right,
                    "model_score_delta": model_delta, "statuses": [a["status"], b["status"]],
                    "success": [a["last_native_success"], b["last_native_success"]], "metrics": {}}
            for metric in METRICS:
                group = counts[f"{left}_to_{right}:{metric}"]
                if a["endpoint"] is None or b["endpoint"] is None:
                    group["missing_fixed_endpoint"] += 1
                    item["metrics"][metric] = None
                    continue
                delta = b["endpoint"][metric] - a["endpoint"][metric]
                actual_direction = direction(delta)
                relation = ("tie" if 0 in (model_direction, actual_direction) else
                            "agree" if model_direction == actual_direction else "opposite")
                group[relation] += 1
                item["metrics"][metric] = {"delta": delta, "direction_relation": relation}
            pairs.append(item)
    return {"schema": NAME, "source_sha256": INPUT_SHA, "tolerance": EPS,
            "counts": {key: dict(sorted(value.items())) for key, value in counts.items()},
            "pairs": pairs, "P1_admission": False, "independent_confirmation": False,
            "limitations": "Descriptive post-hoc comparisons on two shared known roots; "
                           "different actions, missing fixed endpoints, no independent inference."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    source.pairing.need(out.is_absolute() and out.parent == source.ROOT / "results"
                        and out.name.startswith(NAME + "_") and not out.exists()
                        and not out.is_symlink(), "exclusive output")
    data = source.pairing.read(INPUT, INPUT_SHA)
    result = analyze(data)
    result["analysis_sha256"] = source.physical.sha(Path(__file__).resolve())
    source.pairing.need(source.physical.sha(INPUT) == INPUT_SHA, "source drift")
    digest = source.physical.prior.physical.design.run._write_new(out, result)
    print(json.dumps({"pairs": len(result["pairs"]), "counts": result["counts"], "sha256": digest}))


if __name__ == "__main__":
    main()
