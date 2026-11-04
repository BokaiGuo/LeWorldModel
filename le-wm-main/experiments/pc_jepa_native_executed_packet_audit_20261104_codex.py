"""Recompute executed-packet metrics from saved arrays without neural calls."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pc_jepa_native_v7_feedback_dev_20261104_codex"
OUT = ROOT / "results/pc_jepa_native_executed_packet_diag_20261104_codex"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def main():
    report_path = OUT / "result.json"
    report = json.loads(report_path.read_text())
    if report["status"] != "completed" or len(report["rows"]) != 6 or not report["model_unchanged"]:
        raise ValueError("diagnostic incomplete")
    for path, expected in report["bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError("source binding mismatch")
    rows = []
    for row in report["rows"]:
        arm, step = row["arm"], row["step"]
        decision_path = SOURCE / f"{arm}_{step}/output/decision.json"
        input_path = SOURCE / f"{arm}_{step}/input/context.npz"
        actual_path = SOURCE / f"{arm}_physical/frame_{15 + 5 * step:02d}/rgb.npy"
        if (sha(decision_path), sha(input_path), sha(actual_path)) != (
                row["decision_sha256"], row["input_sha256"], row["actual_rgb_sha256"]):
            raise ValueError("input SHA mismatch")
        decision = json.loads(decision_path.read_text())
        chain = np.asarray(decision["decision"]["selected"]["chain"], np.float32)
        for j in range(5):
            cmd = json.loads((SOURCE / f"{arm}_physical/command_{11 + 5 * step + j:02d}.json").read_text())
            if not np.array_equal(chain[j], np.asarray(cmd["action"], np.float32)):
                raise ValueError("actual action mismatch")
        with np.load(input_path, allow_pickle=False) as f:
            if set(f.files) != {"rgb_history", "goal_rgb", "normalized_proprio",
                                "normalized_history_packets", "action_mean", "action_std", "carry"}:
                raise ValueError("unexpected planning field")
        arr_path = OUT / row["array_file"]
        if sha(arr_path) != row["array_sha256"]:
            raise ValueError("array SHA mismatch")
        with np.load(arr_path, allow_pickle=False) as f:
            if set(f.files) != {"current", "predicted", "actual", "goal", "full_terminal"}:
                raise ValueError("array names mismatch")
            arrays = {k: f[k].astype(np.float64).reshape(-1) for k in f.files}
        c, p, a, g = (arrays[x] for x in ("current", "predicted", "actual", "goal"))
        if any(x.shape != (256 * 384,) or not np.isfinite(x).all() for x in arrays.values()):
            raise ValueError("patch dimensions or finite")
        mse = lambda x, y: float(np.mean(np.square(x - y)))
        dp, da = p - c, a - c
        norm_p, norm_a = float(np.linalg.norm(dp)), float(np.linalg.norm(da))
        m = {"pred_vs_actual_mse": mse(p, a), "persistence_vs_actual_mse": mse(c, a),
             "current_goal_mse": mse(c, g), "predicted_goal_mse": mse(p, g), "actual_goal_mse": mse(a, g),
             "predicted_goal_change": mse(p, g) - mse(c, g), "actual_goal_change": mse(a, g) - mse(c, g),
             "predicted_displacement_l2": norm_p, "actual_displacement_l2": norm_a,
             "displacement_error_l2": float(np.linalg.norm(dp - da)),
             "displacement_cosine": float(np.dot(dp, da) / (norm_p * norm_a)) if min(norm_p, norm_a) > 1e-12 else None}
        differences = {k: abs(m[k] - row["metrics"][k]) if m[k] is not None else 0 for k in m}
        if max(differences.values()) > 1e-10:
            raise ValueError("independent metric mismatch")
        rows.append({"arm": arm, "step": step, "max_metric_absolute_error": max(differences.values()),
                     "full_terminal_score_float64": mse(arrays["full_terminal"], g),
                     "original_float32_score": row["selected_score_original"],
                     "recomputed_float32_score": row["full_terminal_score_recomputed_float32"]})
    audit = {"status": "saved_array_metrics_verified", "result_sha256": sha(report_path), "rows": rows,
             "scope": "independent CPU arithmetic, hashes and action binding; no neural or physical rerun"}
    with (OUT / "audit.json").open("x") as f:
        json.dump(audit, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": audit["status"], "rows": len(rows),
                      "max_metric_error": max(row["max_metric_absolute_error"] for row in rows)}))


if __name__ == "__main__":
    main()
