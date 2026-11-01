"""Independent fixed-roster physical metrics and saved-search checks."""
import argparse
import hashlib
import json
from pathlib import Path

from . import p1_return_variants_analysis_20261101_main as metrics
from . import p1_normalized_exchange_saved_20261101_main as saved
from . import p1_reverse_metric_cuda_20261101_main as target

ROOT = metrics.ROOT
PREFIX = "p1_reverse_control_20261101_main"


def analyze():
    sources, rows = {}, []

    def read(path):
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    receipt = read(target.TARGET)
    if sources[str(target.TARGET)] != target.TARGET_SHA:
        raise ValueError("fixed target")
    task = receipt["rows"][0]
    for arm in metrics.ARMS:
        directory = ROOT / f"results/{PREFIX}_{arm}_RUN_MAIN01"
        report = read(directory / "report.json")
        verification = saved.check(directory)
        if len(report["rows"]) != 1 or report["rows"][0]["arm"] != arm:
            raise ValueError("fixed arm")
        row = report["rows"][0]
        points = []
        for cp in row["checkpoints"]:
            if cp["goal"] != task["goal_state"] or cp["goal_rgb_sha256"] != task["pixel_sha256"]:
                raise ValueError("task identity")
            if cp["step"] == 0 and cp["state"] != task["root_state"]:
                raise ValueError("initial state identity")
            points.append(dict(step=cp["step"], **metrics.metrics(cp["state"], cp["goal"]),
                               joint_xy=metrics.qualification.native(cp["goal"], cp["state"])["joint_xy_error"]))
        start = next(p for p in points if p["step"] == 0)
        first = next((p for p in points if p["step"] == 5), None)
        columns = ("pusher_xy", "object_xy", "object_angle", "joint_xy")
        final = row.get("final_state")
        nodes, counts = [], {}
        for node in report["nodes"][row["episode_id"]]:
            if node["status"] != "sealed":
                continue
            node_receipt = node["receipt"]
            for key, value in node_receipt["account"]["charged_upper_bounds"].items():
                counts[key] = counts.get(key, 0) + value
            nodes.append(dict(boundary=node["boundary"], selected_index=node_receipt["selected_index"]))
        rows.append(dict(arm=arm, status=row["status"], native_success=row["native_success"],
                         control_returned=row["control_returned"], first_error=row["first_error"],
                         prefix_delta=metrics.delta({k: first[k] for k in columns},
                                                    {k: start[k] for k in columns}) if first else None,
                         points=points, decisions=nodes, sealed_node_costs=counts,
                         terminal_observed=dict(step=row["control_returned"],
                             **metrics.metrics(final, task["goal_state"]),
                             joint_xy=metrics.qualification.native(task["goal_state"], final)["joint_xy_error"])
                         if final is not None else None,
                         verification=verification))
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in sources.items()):
        raise ValueError("analysis source drift")
    return dict(rows=rows, sources=sources, P1_admission=False,
                independent_confirmation=False, developmental_tasks=1,
                privileged_task_generation=True, physical_outcomes_used_in_selection=False,
                accounting_scope="sealed planning nodes; goal-metric construction and RGB encoding separate",
                caveat="No endpoint padding. Same optimizer opportunity is not equal actual total cost.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(rows=[{k: row[k] for k in ("arm", "status", "native_success", "terminal_observed")}
                               for row in result["rows"]], output=str(args.output))), flush=True)
