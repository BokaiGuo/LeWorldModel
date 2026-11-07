"""Post-hoc decomposition of a sealed, state-assisted action panel."""
import json
import numpy as np

from experiments.pc_input_permissions import file_hash
from experiments import pc_pointmaze_same_renderer_three_way_20261107_codex as panel

ROOT = panel.ROOT
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_ACTION_ERROR_DECOMPOSITION_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_pointmaze_action_error_decomposition_20261107_codex.json"


def analyze(prediction, actual, goal, distances):
    p, a, g = (np.asarray(x, dtype=np.float64) for x in (prediction, actual, goal))
    d = np.asarray(distances, dtype=np.float64)
    if p.shape != a.shape or p.shape[0] < 2 or p.shape[1:] != g.shape or d.shape != (p.shape[0],):
        raise ValueError("incompatible action panel")
    if not all(np.isfinite(x).all() for x in (p, a, g, d)):
        raise ValueError("nonfinite input")
    pc = p - p.mean(axis=0)
    ac = a - a.mean(axis=0)
    bias = (p - a).mean(axis=0)
    mse = lambda v: float(np.mean(np.square(v)))
    raw = mse(p - a)
    common = mse(bias)
    residual = mse(pc - ac)
    if not np.isclose(raw, common + residual, rtol=1e-10, atol=1e-12):
        raise AssertionError("error decomposition failed")
    norm = float(np.linalg.norm(pc.ravel()) * np.linalg.norm(ac.ravel()))
    cosine = None if norm == 0 else float(np.dot(pc.ravel(), ac.ravel()) / norm)
    axes = tuple(range(1, p.ndim))
    pred_cost = np.mean((p - g) ** 2, axis=axes)
    truth_cost = np.mean((a - g) ** 2, axis=axes)
    # Unavailable to a planner: uses the mean of all actual future encodings.
    oracle_cost = np.mean((p - bias - g) ** 2, axis=axes)
    selected = {name: int(np.argmin(cost)) for name, cost in
                (("model", pred_cost), ("actual_encoding", truth_cost), ("oracle_bias", oracle_cost))}
    return {"raw_mse": raw, "common_bias_mse": common, "centered_error_mse": residual,
            "zero_change_baseline_mse": mse(ac), "centered_prediction_mse": mse(pc),
            "effect_cosine": cosine, "selected": selected,
            "physical_regret": {key: float(d[i] - d.min()) for key, i in selected.items()}}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan = json.loads(panel.PLAN.read_text())
    result = json.loads(panel.RESULT.read_text())
    if (plan["predictions_sha256"] != file_hash(panel.PRED)
            or result["plan_sha256"] != file_hash(panel.PLAN)
            or result["actual_sha256"] != file_hash(panel.ACTUAL)
            or result["prediction_sha256"] != file_hash(panel.PRED)
            or plan["track_id"] != "official_state_assisted"):
        raise ValueError("sealed input mismatch")
    rows = []
    with np.load(panel.PRED, allow_pickle=False) as p, np.load(panel.ACTUAL, allow_pickle=False) as a:
        for prior in result["rows"]:
            root, camera = prior["root"], prior["camera"]
            key = f"r{root - 1880}_v{camera}"
            row = analyze(p[key + "_pred"], a[key], p[key + "_goal"], prior["distances"])
            row.update(root=root, camera=camera)
            if row["selected"]["model"] != prior["selected"]["predicted"]:
                raise ValueError("sealed choice mismatch")
            rows.append(row)
    if {(r["root"], r["camera"]) for r in rows} != {(r, c) for r in range(1880, 1884) for c in range(2)}:
        raise ValueError("incomplete task/camera panel")
    metrics = ("raw_mse", "common_bias_mse", "centered_error_mse", "zero_change_baseline_mse",
               "centered_prediction_mse", "effect_cosine")
    root_summary = []
    for root in range(1880, 1884):
        subset = [r for r in rows if r["root"] == root]
        summary = {"root": root}
        for metric in metrics:
            values = [r[metric] for r in subset if r[metric] is not None]
            summary[metric] = float(np.mean(values)) if values else None
        summary["physical_regret"] = {name: float(np.mean([r["physical_regret"][name] for r in subset]))
                                      for name in ("model", "actual_encoding", "oracle_bias")}
        root_summary.append(summary)
    record = {"track": "official_state_assisted", "scope": "post-hoc exposed panel; oracle not deployable",
              "protocol_sha256": file_hash(PROTOCOL), "script_sha256": file_hash(__file__),
              "plan_sha256": file_hash(panel.PLAN), "result_sha256": file_hash(panel.RESULT),
              "prediction_sha256": file_hash(panel.PRED), "actual_sha256": file_hash(panel.ACTUAL),
              "rows": rows, "roots": root_summary}
    with OUTPUT.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps({"roots": root_summary}, indent=2))


if __name__ == "__main__":
    main()
