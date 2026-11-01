"""All-arm search revision results, preserving failed/censored execution."""
import argparse
import hashlib
import json
from pathlib import Path

from . import p1_return_variants_analysis_20261101_main as prior

ROOT = prior.ROOT
VARIANTS = {
    "q_winner": "p1_q_return_shard_20261101_main",
    "q_horizon": "p1_horizon_q_20261101_main",
    "q_exchange": "p1_suffix_exchange_q_20261101_main",
    "q_normalized_exchange": "p1_normalized_exchange_q_20261101_main",
}


def analyze():
    sources, rows = {}, []

    def read(path):
        blob = path.read_bytes()
        sources[str(path)] = hashlib.sha256(blob).hexdigest()
        return json.loads(blob)

    target = read(ROOT / "results/p1_return_target_cuda_20261101_main_MAIN06.json")["rows"][0]
    for variant, prefix in VARIANTS.items():
        for arm in prior.ARMS:
            directory = ROOT / f"results/{prefix}_{arm}_RUN_MAIN01"
            report = read(directory / "report.json")
            if len(report["rows"]) != 1 or report["rows"][0]["arm"] != arm:
                raise ValueError("fixed single arm")
            row = report["rows"][0]
            points = []
            for cp in row["checkpoints"]:
                if cp["goal"] != target["goal_state"] or (cp["step"] == 0 and cp["state"] != target["root_state"]):
                    raise ValueError("same initial task")
                result = prior.qualification.native(cp["goal"], cp["state"])
                points.append(dict(step=cp["step"], **prior.metrics(cp["state"], cp["goal"]),
                                   joint_xy=result["joint_xy_error"], native7=result["native7_distance"]))
            cp0 = next(p for p in points if p["step"] == 0)
            cp5 = next((p for p in points if p["step"] == 5), None)
            columns = ("pusher_xy", "object_xy", "object_angle", "joint_xy", "native7")
            counts = {}
            decisions = []
            for node in report["nodes"][row["episode_id"]]:
                if node["status"] != "sealed":
                    continue
                receipt = node["receipt"]
                for key, value in receipt["account"]["charged_upper_bounds"].items():
                    counts[key] = counts.get(key, 0) + value
                decisions.append(dict(boundary=node["boundary"], selected_index=receipt["selected_index"]))
            rows.append(dict(variant=variant, arm=arm, report_status=report["status"],
                             status=row["status"], native_success=row["native_success"],
                             physical_steps=row["control_returned"], first_error=row["first_error"],
                             prefix_delta=prior.delta({k: cp5[k] for k in columns},
                                                      {k: cp0[k] for k in columns}) if cp5 else None,
                             points=points, endpoint20=next((p for p in points if p["step"] == 20), None),
                             terminal_observed=dict(step=row["control_returned"],
                                 **prior.metrics(row["final_state"], target["goal_state"]),
                                 joint_xy=prior.qualification.native(
                                     target["goal_state"], row["final_state"])["joint_xy_error"])
                             if row.get("final_state") is not None else None,
                             sealed_planner_node_counts=counts, decisions=decisions))
    return dict(rows=rows, sources=sources, eligible_roots=1, development_only=True,
                P1_admission=False, independent_confirmation=False,
                accounting_scope="sealed planner nodes only; RGB encoding and Q construction separate",
                caution="No endpoint imputation. Four internal arms are not certified external strong baselines.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(rows=len(result["rows"]), output=str(args.output))))
