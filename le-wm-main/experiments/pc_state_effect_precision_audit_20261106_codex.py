"""Recompute precision diagnostic summaries without loading the model."""
import json

import numpy as np

from experiments.pc_input_permissions import file_hash
from experiments.pc_state_effect_precision_cause_20261106_codex import OUT, SOURCE, PROTOCOL, STEPS


def reconstruct(scan):
    center = scan["center"]
    for item in scan["scan"]:
        h = item["step"]
        np.testing.assert_allclose(item["central_derivative"],
                                   (item["plus"] - item["minus"]) / (2*h), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(item["curvature"],
                                   (item["plus"] + item["minus"] - 2*center) / h**2, rtol=1e-12, atol=1e-12)
    by_step = {item["step"]: item for item in scan["scan"]}
    near = by_step[.003]
    predictions, labels = [], []
    for h in (.03, .1):
        for sign, field in ((-1, "minus"), (1, "plus")):
            x = sign * h
            predictions.append(center + near["central_derivative"] * x + .5 * near["curvature"] * x*x)
            labels.append(by_step[h][field])
    return float(np.abs(np.asarray(predictions) - labels).mean())


def main():
    output = OUT.with_name(OUT.stem + "_audit.json")
    if output.exists():
        raise FileExistsError(output)
    record = json.loads(OUT.read_text())
    if record["protocol_sha256"] != file_hash(PROTOCOL) or record["original_result_sha256"] != file_hash(SOURCE):
        raise ValueError("upstream record drift")
    rows = []
    for row in record["rows"]:
        values = {precision: reconstruct(row[precision]) for precision in ("fp32", "fp64")}
        for precision in ("fp32", "fp64"):
            if [item["step"] for item in row[precision]["scan"]] != list(STEPS):
                raise ValueError("unexpected scan support")
        rows.append({"root": row["root"], "camera": row["camera"],
                     "quadratic_heldout_mae": values,
                     "gradient_abs_diff": abs(row["fp32"]["gradient"] - row["fp64"]["gradient"]),
                     "curvature_at_003": {p: row[p]["scan"][3]["curvature"] for p in ("fp32", "fp64")},
                     "curvature_at_01": {p: row[p]["scan"][6]["curvature"] for p in ("fp32", "fp64")}})
    result = {"source_sha256": file_hash(OUT), "code_sha256": file_hash(__file__),
              "rows": rows, "mean_quadratic_mae": {p: float(np.mean([r["quadratic_heldout_mae"][p] for r in rows]))
                                                   for p in ("fp32", "fp64")},
              "max_gradient_abs_diff": max(row["gradient_abs_diff"] for row in rows)}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
