"""Decompose saved JEPA MSE into chart and orthogonal contributions."""
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pc_jepa_v12_contact_score_bank_posthoc_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_JEPA_CHART_SCORE_DECOMPOSITION_POSTHOC_PROTOCOL_20261104_CODEX.md"
WORKER = ROOT / "experiments/pc_jepa_v12_contact_score_predict_worker_20261104_codex.py"
OUT = ROOT / "results/pc_jepa_chart_score_decomposition_posthoc_20261104_codex.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordering(a, b, tolerance=1e-8):
    a, b = np.asarray(a), np.asarray(b)
    i, j = np.triu_indices(len(a), 1)
    da, db = a[i] - a[j], b[i] - b[j]
    valid = (np.abs(da) > tolerance) & (np.abs(db) > tolerance)
    count = int(valid.sum())
    agree = int(np.sum((da[valid] * db[valid]) > 0))
    return {"comparable_pairs": count, "agree": agree, "ties_or_excluded": int(len(i) - count),
            "agreement": agree / count if count else None, "tolerance": tolerance}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    decision_path = SOURCE / "neural/predict/output/decision.json"
    report_path = SOURCE / "neural/report.json"
    result_path = SOURCE / "result.json"
    audit_path = SOURCE / "audit.json"
    source_audit = json.loads(audit_path.read_text())
    report, data = (json.loads(path.read_text()) for path in (report_path, decision_path))
    if (source_audit["status"] != "passed" or source_audit["result_sha256"] != digest(result_path)
            or source_audit["report_sha256"] != digest(report_path)
            or report["source_result_sha256"] != digest(result_path)
            or report["predict_sha256"] != digest(decision_path) or data["worker_sha256"] != digest(WORKER)):
        raise ValueError("source evidence binding mismatch")
    deps = seal.capture([Path(__file__), PROTOCOL, WORKER, result_path, audit_path, report_path, decision_path])
    scores = np.asarray(data["decision"]["scores"], np.float64)
    positions = np.asarray(data["decision"]["positions"], np.float64)
    target = np.asarray(data["chart_goal"], np.float64)
    encoded = np.asarray(report["actual_encoded"], np.float64)
    if (scores.shape != (64,) or positions.shape != (64, 2) or target.shape != (2,)
            or encoded.shape != (64,) or not all(np.isfinite(x).all() for x in (scores, positions, target, encoded))
            or not np.array_equal(scores, np.asarray(report["predicted"]))):
        raise ValueError("unexpected saved neural data")
    projected = np.sum((positions - target) ** 2, axis=1)
    residual = scores - projected
    groups = {}
    for name, sl in (("planner_slice", slice(0, 56)), ("privileged_witness_variants", slice(56, 64))):
        total, proj, rest = scores[sl], projected[sl], residual[sl]
        fraction = np.divide(proj, total, out=np.full_like(proj, np.nan), where=total > 0)
        groups[name] = {"count": len(total), "full_mse_range": [float(total.min()), float(total.max())],
            "projected_range": [float(proj.min()), float(proj.max())],
            "residual_range": [float(rest.min()), float(rest.max())],
            "projection_fraction_range": [float(np.nanmin(fraction)), float(np.nanmax(fraction))],
            "chart_vs_full": ordering(proj, total), "chart_vs_actual_encoded": ordering(proj, encoded[sl]),
            "full_vs_actual_encoded": ordering(total, encoded[sl])}
    seal.verify(deps)
    with OUT.open("x") as stream:
        json.dump({"status": "complete", "dependencies": deps, "groups": groups,
                   "projected": projected.tolist(), "residual": residual.tolist(),
                   "scope": "post-hoc saved predictions; not physical or independent evidence"},
                  stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
