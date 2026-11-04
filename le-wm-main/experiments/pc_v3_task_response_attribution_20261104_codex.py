"""Task-distance sensitivity from the sealed tail-local intervention."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_response_temporal_v3_tail_local_mechanism_20261104_codex as source

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V3_TASK_RESPONSE_ATTRIBUTION_PROTOCOL_20261104_CODEX.md"
OUTPUT = ROOT / "results/pc_v3_task_response_attribution_20261104_codex.json"
NAMES = ("selected", "zero", "greedy", "random_bridge", "lookahead")


def calculate(row):
    goal = np.asarray(row["task"]["goal"], dtype=float)
    result = {}
    for eps in (.01, .05):
        controls = {}
        for name in NAMES:
            record = row["controls"][name]["tail_local_response"][str(eps)]
            points = np.asarray(record["center_and_probe_endpoints"], dtype=float)
            matrix = np.asarray(record["response"], dtype=float)
            if points.shape != (21, 2) or matrix.shape != (2, 10) or not np.isfinite(points).all():
                raise ValueError("invalid recorded response")
            np.testing.assert_allclose(matrix, ((points[1:11] - points[11:21]) / (2 * eps)).T,
                                       rtol=0, atol=1e-12)
            distances = np.linalg.norm(points - goal, axis=1)
            gradient_fd = (distances[1:11] - distances[11:21]) / (2 * eps)
            gradient = None if distances[0] <= 1e-10 else matrix.T @ ((points[0] - goal) / distances[0])
            controls[name] = {"distance": float(distances[0]), "distance_probe_fd": gradient_fd.tolist(),
                              "distance_gradient": None if gradient is None else gradient.tolist(),
                              "gradient_fd_error_l2": None if gradient is None else float(
                                  np.linalg.norm(gradient - gradient_fd)),
                              "branch_stable": record["branch_stable"]}
        selected = controls["selected"]["distance_gradient"]
        if selected is not None and np.linalg.norm(selected) > 1e-10:
            reference = -np.asarray(selected) / np.linalg.norm(selected)
            for entry in controls.values():
                gradient = entry["distance_gradient"]
                entry["directional_derivative"] = None if gradient is None else float(np.dot(gradient, reference))
                entry["gradient_gap_to_selected"] = None if gradient is None else float(
                    np.linalg.norm(np.asarray(gradient) - selected))
            reference = reference.tolist()
        else:
            reference = None
        result[str(eps)] = {"selected_descent_direction": reference, "controls": controls}
    return {"root": row["root"], "task": row["task"], "nonprogress": row["nonprogress"],
            "original_intersection": row["tail_all_intersection"], "scales": result}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = source.prior.base.sha
    data = json.loads(source.INTERVENTION.read_text())
    rebuilt = json.loads(source.REBUILD.read_text())
    audit = json.loads(source.AUDIT.read_text())
    if (rebuilt["intervention_sha256"] != sha(source.INTERVENTION)
            or audit["plans_sha256"] != sha(source.PLANS)
            or data["plans_sha256"] != sha(source.PLANS)):
        raise ValueError("sealed source mismatch")
    rows = [calculate(row) for row in data["rows"]]
    if len(rows) != 24 or {row["root"] for row in rows} != set(range(24)):
        raise ValueError("incomplete panel")
    source.prior.base.write(OUTPUT, {"protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
                                    "source_sha256": sha(source.INTERVENTION),
                                    "rebuild_sha256": sha(source.REBUILD), "rows": rows,
                                    "scope": "post-result attribution; not prospective confirmation"})
    print("Recorded task-distance responses for all 24 exposed roots.", flush=True)


if __name__ == "__main__":
    main()
