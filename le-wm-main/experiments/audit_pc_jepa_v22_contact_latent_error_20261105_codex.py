"""Independent per-row CPU arithmetic on the saved contact latent arrays."""
import json

import numpy as np

from experiments import pc_jepa_v22_contact_latent_error_20261105_codex as run


def main():
    run.verify()
    out = run.OUT
    if (out / "audit.json").exists():
        raise FileExistsError(out / "audit.json")
    report = json.loads((out / "report.json").read_text())
    receipt = json.loads((out / "output/receipt.json").read_text())
    if (report["status"] != "completed" or report["latents_sha256"] != run.sha(out / "output/latents.npz")
            or report["worker_receipt_sha256"] != run.sha(out / "output/receipt.json")
            or receipt["latents_sha256"] != report["latents_sha256"] or len(report["rows"]) != 64):
        raise ValueError("report binding")
    physical = json.loads((run.SOURCE / "physical.json").read_text())
    max_error = 0.
    with np.load(out / "output/latents.npz", allow_pickle=False) as data:
        current = data["current"].astype(np.float64)
        goal = data["goal"].astype(np.float64)
        for i, row in enumerate(report["rows"]):
            pred = data["predicted"][i].astype(np.float64)
            actual = data["actual"][i].astype(np.float64)
            values = {"prediction_mse": np.mean(np.square(pred - actual)),
                      "persistence_mse": np.mean(np.square(current - actual)),
                      "predicted_goal_mse": np.mean(np.square(pred - goal)),
                      "actual_goal_mse": np.mean(np.square(actual - goal))}
            if row["index"] != i or row["physical"] != physical["rows"][i]["metrics"]:
                raise ValueError("row binding mismatch")
            max_error = max(max_error, *(abs(float(value) - row[key]) for key, value in values.items()))
    if max_error > 1e-12:
        raise ValueError("metric arithmetic mismatch")
    result = {"status": "passed", "rows": 64, "max_abs_arithmetic_error": max_error,
              "report_sha256": run.sha(out / "report.json"), "latents_sha256": run.sha(out / "output/latents.npz"),
              "scope": "independent CPU arithmetic; no independent neural or physical replay"}
    with (out / "audit.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
