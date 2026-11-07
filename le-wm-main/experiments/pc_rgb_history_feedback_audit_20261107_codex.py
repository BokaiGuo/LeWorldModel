"""Post-run verification of history packet receipts against dispatched actions."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pc_input_permissions import file_hash, tensor_receipt
from experiments import pc_rgb_true_history_alignment_20261107_codex as source


def main():
    path = source.ROOT / "results/pc_rgb_history_feedback_20261107_codex.json"
    output = source.ROOT / "results/pc_rgb_history_feedback_audit_20261107_codex.json"
    if output.exists():
        raise FileExistsError(output)
    record = json.loads(path.read_text())
    if record["runner_sha256"] != file_hash(Path(__file__).with_name(
            "pc_rgb_history_feedback_20261107_codex.py")):
        raise ValueError("runner changed")
    _, scaler, _, action_hash = source.model_and_scaler()
    if action_hash != record["normalizer_source_sha256"]:
        raise ValueError("normalizer changed")
    rows = []
    for row in record["rows"]:
        prior = []
        if len(row["receipts"]) != 3 * len(row["plans"]):
            raise ValueError("unexpected model query count")
        for iteration, plan in enumerate(row["plans"]):
            packets = prior[-2:] if row["mode"] == "history" else []
            if packets:
                normal = scaler.transform(np.asarray(packets, np.float32).reshape(-1, 2))
                tensor = torch.tensor(normal.reshape(1, 1, len(packets), 10), dtype=torch.float32)
            else:
                tensor = torch.empty((1, 1, 0, 10), dtype=torch.float32)
            expected = tensor_receipt(tensor)
            receipts = row["receipts"][iteration * 3:(iteration + 1) * 3]
            if any(receipt["inputs"]["action"] != expected for receipt in receipts):
                raise ValueError("scoring history does not match dispatched actions")
            if plan["history_frames"] != len(packets) + 1 or plan["scored_candidates"] != 66:
                raise ValueError("history or budget mismatch")
            steps = plan["executed_steps"]
            dispatched = np.asarray([x["action"] for x in steps], np.float32)
            if not np.isfinite(dispatched).all() or np.any(np.abs(dispatched) > 1):
                raise ValueError("invalid dispatched action")
            selected = np.asarray(plan["normalized_selected_chain"], np.float32)
            decoded = scaler.inverse_transform(selected.reshape(-1, 2)).reshape(3, 5, 2)
            np.testing.assert_allclose(dispatched, decoded[0, :len(steps)], atol=2e-7, rtol=1e-6)
            if len(steps) == 5:
                prior.append(dispatched)
        rows.append({"root": row["root"], "mode": row["mode"],
                     "history_frames": [p["history_frames"] for p in row["plans"]],
                     "executed_steps": sum(len(p["executed_steps"]) for p in row["plans"]),
                     "success_ever": any(s["terminated"] for p in row["plans"] for s in p["executed_steps"]),
                     "scored_candidates": sum(p["scored_candidates"] for p in row["plans"]),
                     "final_reward": row["plans"][-1]["executed_steps"][-1]["reward"]})
    for root in (0, 8):
        pair = [r for r in record["rows"] if r["root"] == root]
        a = np.asarray(pair[0]["plans"][0]["normalized_selected_chain"])
        b = np.asarray(pair[1]["plans"][0]["normalized_selected_chain"])
        np.testing.assert_array_equal(a, b)
    result = {"rows": rows, "run_sha256": file_hash(path), "audit_sha256": file_hash(__file__),
              "scope": "history/dispatch receipt consistency; no independent performance claim"}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
