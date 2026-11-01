"""Apply the fixed common-direction diversity rule to an existing model panel."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p1_joint_suffix_fd_host_20261029 as source
from experiments.spectral_reachability import p1_task_response_diversity_20261030_main as diversity


def analyze(report, prepared):
    if len(report["rows"]) != 8 or len(prepared["requests"]) != 4:
        raise ValueError("fixed panel dimensions")
    expected_arms = {"projected_ascent", "outward_free_tangent"}
    for slot in range(4):
        pair = report["rows"][2 * slot:2 * slot + 2]
        if any(row["slot"] != slot for row in pair) or {row["arm"] for row in pair} != expected_arms:
            raise ValueError("fixed panel identities")
    results = []
    for idx, arm in enumerate(report["rows"]):
        slot = idx // 2
        if arm["slot"] != slot or arm["arm"] not in ("projected_ascent", "outward_free_tangent"):
            raise ValueError("fixed panel identities")
        task = prepared["requests"][slot]["task"]
        projection = np.asarray(task["projection"], dtype=np.float64)
        goal = np.asarray(task["goal"], dtype=np.float64)
        root = np.asarray(task["root_context"]["latent_history"][-1], dtype=np.float64)
        if projection.shape != (192, 192) or not np.array_equal(projection, np.eye(192)):
            raise ValueError("requires fixed identity target metric")
        candidates = arm["result"]["candidates"]
        if len(candidates) != 20:
            raise ValueError("fixed candidate denominator")
        jacobians = [c["signal"]["jacobian"] for c in candidates]
        scores = [c["stage_Q"] for c in candidates]
        picked = diversity.select(jacobians, goal - root, scores, 4)
        results.append(dict(slot=slot, arm=arm["arm"], selected_indices=picked["selected_indices"],
                            selection_min_squared_distance=picked["selection_min_squared_distance"],
                            response_features=picked["response_features"], stage_scores=picked["stage_scores"],
                            root_direction=picked["task_direction"]))
    return dict(schema="p1_task_response_saved_panel_20261030_main.v1", arms=results,
                source_report_sha256=source.REPORT_SHA, source_receipt_sha256=source.RECEIPT_SHA,
                fresh=False, physical_execution=False, P1_admission=False,
                limitation="Known outcomes, post-hoc fixed panel; selected candidates are not new actions or equal-budget evidence.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if (not path.is_absolute() or path.parent != source.ROOT / "results"
            or not path.name.startswith("p1_task_response_saved_panel_20261030_main_")
            or path.exists() or path.is_symlink()):
        raise ValueError("exclusive output")
    report = source.saved.read_verified(source.REPORT, source.REPORT_SHA)
    receipt = source.saved.read_verified(source.RECEIPT, source.RECEIPT_SHA)
    if receipt.get("report_sha256") != source.REPORT_SHA:
        raise ValueError("receipt identity")
    prepared = source.saved.read_verified(report["prepared"], report["prepare_sha256"])
    source.saved.analyze(report, receipt, prepared)
    source.host.verify(report["prepared"], report["prepare_sha256"])
    result = analyze(report, prepared)
    data = json.dumps(result, sort_keys=True, allow_nan=False, separators=(",", ":")).encode() + b"\n"
    with path.open("xb") as stream:
        stream.write(data)
    print(json.dumps(dict(path=str(path), sha256=hashlib.sha256(data).hexdigest())))


if __name__ == "__main__":
    main()
