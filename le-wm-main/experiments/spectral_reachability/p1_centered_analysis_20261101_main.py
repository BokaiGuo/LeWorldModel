"""Fixed-roster physical metrics for the centered development panel."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import p1_centered_panel_20261101_main as panel
from . import p1_centered_metric_cuda_20261101_main as target
from . import p1_return_variants_analysis_20261101_main as metrics


def analyze(tag):
    saved = panel.saved
    summary_path = panel.ROOT / "results" / f"p1_centered_panel_20261101_main_{tag}.json"
    summary = json.loads(summary_path.read_bytes())
    if [r["arm"] for r in summary["rows"]] != list(panel.ARMS):
        raise ValueError("complete fixed roster")
    if saved.sha(target.TARGET) != target.TARGET_SHA:
        raise ValueError("fixed target receipt")
    goal = json.loads(target.TARGET.read_bytes())["rows"][0]
    sources = {str(summary_path): saved.sha(summary_path), str(target.TARGET): target.TARGET_SHA}
    metric_path = panel.ROOT / "results/p1_centered_metric_cuda_20261101_main_MAIN01.json"
    if saved.sha(metric_path) != "ef9ae31f143a18d73432fb45bf861db13f9f852baa776fe977962ec9cb2a4818":
        raise ValueError("fixed metric")
    matrix = np.asarray(json.loads(metric_path.read_bytes())["sqrt_matrix"], float)
    goal_latent = np.asarray(goal["latent"], np.float32)
    for path in (metric_path, Path(__file__).resolve(), Path(metrics.__file__).resolve()):
        sources[str(path)] = saved.sha(path)
    rows = []
    for item in summary["rows"]:
        path = Path(item["report"])
        if saved.sha(path) != item["report_sha256"]:
            raise ValueError("report hash")
        sources[str(path)] = item["report_sha256"]
        report = json.loads(path.read_bytes())
        row = report["rows"][0]
        if row["arm"] != item["arm"] or row["first_error"]:
            raise ValueError("arm identity or infrastructure error")
        def measure(state):
            native = metrics.qualification.native(goal["goal_state"], state)
            return dict(**metrics.metrics(state, goal["goal_state"]),
                        joint_xy=native["joint_xy_error"],
                        native7_distance=native["native7_distance"],
                        native_success=native["initial_success"])
        points = []
        for point in row["checkpoints"]:
            if point["goal"] != goal["goal_state"] or point["goal_rgb_sha256"] != goal["pixel_sha256"]:
                raise ValueError("actual goal identity")
            if point["step"] == 0 and point["state"] != goal["root_state"]:
                raise ValueError("actual translated root")
            points.append(dict(step=point["step"], **measure(point["state"])))
        for encoding in row["encodings"]:
            latent_path = Path(encoding["latent"]["path"])
            if (not latent_path.resolve().is_relative_to(path.parent)
                    or saved.sha(latent_path) != encoding["latent"]["file_sha256"]):
                raise ValueError("actual RGB latent artifact")
            sources[str(latent_path)] = saved.sha(latent_path)
            latent = np.load(latent_path, allow_pickle=False)
            if latent.shape != (192,) or latent.dtype != np.float32 or not np.isfinite(latent).all():
                raise ValueError("finite RGB latent")
            point = next(p for p in points if p["step"] == encoding["step"])
            residual = (latent-goal_latent).astype(float)
            point["actual_RGB_Q"] = float(np.mean((residual @ matrix.T)**2))
        initial = next(p for p in points if p["step"] == 0)
        prefix = next((p for p in points if p["step"] == 5), None)
        final = measure(row["final_state"]) if row.get("final_state") is not None else None
        if final is not None and final["native_success"] != row["native_success"]:
            raise ValueError("native success recomputation")
        cost = {}
        for node in report["nodes"][row["episode_id"]]:
            if node["status"] == "sealed":
                for axis, amount in node["receipt"]["account"]["charged_upper_bounds"].items():
                    cost[axis] = cost.get(axis, 0) + amount
        first_node = report["nodes"][row["episode_id"]][0]
        temporal = None
        if first_node["status"] == "sealed":
            trace = Path(first_node["receipt"]["trace"])
            if saved.sha(trace) != first_node["receipt"]["trace_sha256"]:
                raise ValueError("initial node trace")
            sources[str(trace)] = saved.sha(trace)
            events = [json.loads(line) for line in trace.read_bytes().splitlines()]
            banks = [e["payload"]["raw"] for e in events if e["kind"] == "initial_candidates"]
            if len(banks) != 1:
                raise ValueError("one initial bank")
            bank = np.asarray(banks[0], np.float32)
            if bank.shape != (8, 4, 5, 2):
                raise ValueError("full initial candidate bank")
            numerator = np.linalg.norm(bank.sum(axis=2), axis=-1)
            denominator = np.linalg.norm(bank, axis=-1).sum(axis=2)
            ratios = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                               where=denominator > 0)
            temporal = dict(unique_prefixes=len({p.tobytes() for p in bank[:, 0]}),
                            packet_direction_ratios=ratios.tolist(),
                            zero_packets=int(np.count_nonzero(denominator == 0)),
                            nonzero_packet_mean_ratio=float(ratios[denominator > 0].mean())
                            if (denominator > 0).any() else None,
                            caution="Action coherence only; not a measured cause of failure.")
        rows.append(dict(arm=row["arm"], status=row["status"], steps=row["control_returned"],
            final=final, points=points, prefix_delta=None if prefix is None else
            {k: prefix[k]-initial[k] for k in (
                "object_xy", "object_angle", "pusher_xy", "joint_xy", "actual_RGB_Q")},
            actual_charged_costs=cost, initial_candidate_temporal=temporal,
            verification=item["verification"]))
    return dict(rows=rows, sources=sources, P1_admission=False, independent_confirmation=False,
                caution="Known translated development root; no endpoint imputation or equal-actual-cost claim.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.tag)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps([{k: r[k] for k in ("arm", "status", "steps", "final", "prefix_delta")}
                      for r in result["rows"]]), flush=True)
