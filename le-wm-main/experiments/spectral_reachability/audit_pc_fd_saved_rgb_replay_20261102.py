"""Independent saved-data checks and paired descriptions for FD RGB replay."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def mse(a, b):
    if not a or len(a) != len(b) or any(not math.isfinite(v) for v in a + b):
        raise ValueError("invalid latent vectors")
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


def analyze(replay, physical):
    if replay["status"] != "complete" or not replay["frozen_parameters_unchanged"]:
        raise ValueError("incomplete replay or mutable model")
    if replay["parameter_hash_before"] != replay["parameter_hash_after"]:
        raise ValueError("model hash changed")
    if [r["seed"] for r in replay["rows"]] != [20263160, 20263161]:
        raise ValueError("root mismatch")
    if [r["seed"] for r in physical["rows"]] != [20263160, 20263161]:
        raise ValueError("physical root mismatch")
    results = []
    for row, truth in zip(replay["rows"], physical["rows"]):
        if len(row["branches"]) != 10 or len(truth["branches"]) != 10:
            raise ValueError("branch count mismatch")
        expected = [(arm, tail) for arm in ("selected", "parameter_zero") for tail in range(5)]
        if [(b["arm"], b["tail_index"]) for b in row["branches"]] != expected:
            raise ValueError("branch roster mismatch")
        for b, source in zip(row["branches"], truth["branches"]):
            if b["status"] != "complete" or (b["arm"], b["prefix_index"], b["tail_index"]) != (
                    source["arm"], source["prefix_index"], source["tail_index"]):
                raise ValueError("source branch mismatch")
            if [c["step"] for c in b["checkpoints"]] != [5, 10, 15]:
                raise ValueError("checkpoint mismatch")
            for c in b["checkpoints"]:
                pc = source["trajectory"][c["step"]]
                if c["rgb_sha256"] != pc["rgb"]["sha256"] or c["qpos_distance"] != pc["qpos_distance"]:
                    raise ValueError("physical checkpoint binding mismatch")
                for prefix in ("open", "refreshed"):
                    value = mse(c[prefix + "_latent"], c["observed_latent"])
                    if not math.isclose(value, c[prefix + "_mse"], rel_tol=1e-5, abs_tol=1e-9):
                        raise ValueError("latent MSE mismatch")
                if any(not math.isfinite(c[x]) for x in ("observed_goal_q", "open_goal_q", "refreshed_goal_q")):
                    raise ValueError("nonfinite goal Q")
        steps = []
        for i, step in enumerate((5, 10, 15)):
            points = [b["checkpoints"][i] for b in row["branches"]]
            steps.append({"step": step,
                          "open_mse_mean": sum(c["open_mse"] for c in points) / 10,
                          "refreshed_mse_mean": sum(c["refreshed_mse"] for c in points) / 10,
                          "refresh_strictly_lower_count": sum(c["refreshed_mse"] < c["open_mse"] for c in points)})
        comparisons = []
        for i in range(5):
            s = row["branches"][i]["checkpoints"][-1]
            z = row["branches"][i + 5]["checkpoints"][-1]
            comparisons.append({"tail_index": i, **{name: s[key] - z[key] for name, key in (
                ("open_q_delta", "open_goal_q"), ("refreshed_q_delta", "refreshed_goal_q"),
                ("observed_q_delta", "observed_goal_q"), ("qpos_delta", "qpos_distance"))}})
        def reversals(a, b):
            return sum(c[a] * c[b] < 0 for c in comparisons)
        results.append({"seed": row["seed"], "steps": steps, "pairs": comparisons,
                        "open_vs_observed_reversals": reversals("open_q_delta", "observed_q_delta"),
                        "refreshed_vs_observed_reversals": reversals("refreshed_q_delta", "observed_q_delta"),
                        "observed_vs_qpos_reversals": reversals("observed_q_delta", "qpos_delta")})
    return {"status": "saved_mse_and_pairing_verified",
            "limits": ["Q scores not independently recalculated from goal latent",
                       "no model inference or physical replay", "no P1 admission"], "roots": results}


def verify_goal_q(replay, metric_report):
    if [r["seed"] for r in metric_report["rows"]] != [r["seed"] for r in replay["rows"]]:
        raise ValueError("metric root mismatch")
    count = 0
    for row, source in zip(replay["rows"], metric_report["rows"]):
        gram = np.asarray(source["goal_metric"]["gram"], dtype=np.float64)
        target = np.asarray(row["goal_coordinate"], dtype=np.float64)
        if (gram.shape != (target.size, target.size) or target.ndim != 1
                or not np.isfinite(gram).all() or not np.isfinite(target).all()
                or not np.allclose(gram, gram.T, rtol=1e-12, atol=1e-12)):
            raise ValueError("invalid goal metric")
        values, vectors = np.linalg.eigh(gram)
        if values.min() < -1e-10 * max(1.0, np.abs(values).max()):
            raise ValueError("goal metric is not positive semidefinite")
        weights = 0.01 / (np.maximum(values, 0.0) + 0.01)
        for branch in row["branches"]:
            for point in branch["checkpoints"]:
                for role in ("open", "refreshed", "observed"):
                    coord = np.asarray(point[role + "_coordinate"], dtype=np.float64)
                    if coord.shape != target.shape or not np.isfinite(coord).all():
                        raise ValueError("invalid saved coordinate")
                    # The model computes the coordinate subtraction in float32
                    # before converting that difference to the metric's float64.
                    delta = (coord.astype(np.float32) - target.astype(np.float32)).astype(np.float64)
                    rotated = vectors.T @ delta
                    expected = float(np.sum(weights * rotated**2) / target.size)
                    if not math.isclose(expected, point[role + "_goal_q"], rel_tol=1e-8, abs_tol=1e-10):
                        raise ValueError("goal Q mismatch")
                    count += 1
    return count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--physical", type=Path, required=True)
    p.add_argument("--metric-report", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    data = args.replay.read_bytes()
    source = args.physical.read_bytes()
    replay, physical = json.loads(data), json.loads(source)
    if replay["physical_sha256"] != hashlib.sha256(source).hexdigest():
        raise ValueError("physical digest mismatch")
    result = analyze(replay, physical)
    if args.metric_report:
        metric_data = args.metric_report.read_bytes()
        metric_sha = hashlib.sha256(metric_data).hexdigest()
        if any(row["goal_metric_source_sha256"] != metric_sha for row in replay["rows"]):
            raise ValueError("metric source digest mismatch")
        result["verified_goal_q_values"] = verify_goal_q(replay, json.loads(metric_data))
        result["limits"][0] = "Q recomputed from saved coordinates; coordinate generation not independently replayed"
        result["metric_report_sha256"] = metric_sha
    result.update(replay_sha256=hashlib.sha256(data).hexdigest(),
                  physical_sha256=hashlib.sha256(source).hexdigest())
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps({"status": result["status"], "roots": len(result["roots"])}))


if __name__ == "__main__":
    main()
