"""Recompute sealed history predictions from explicit encode/predict calls."""
import json

import numpy as np
import torch

from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments import pc_rgb_true_history_alignment_20261107_codex as run

OUT = run.ROOT / "results/pc_rgb_true_history_predictor_audit_20261107_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    record = json.loads(run.RECORD.read_text())
    if (record["input_sha256"] != file_hash(run.INPUT) or record["pred_sha256"] != file_hash(run.PRED)
            or record["runner_sha256"] != file_hash(run.__file__)):
        raise ValueError("sealed prediction inputs changed")
    model, scaler, transform, action_hash = run.model_and_scaler()
    if model_fingerprint(model) != record["model_sha256"] or action_hash != record["action_array_sha256"]:
        raise ValueError("model or normalizer mismatch")
    rows = []
    with np.load(run.INPUT, allow_pickle=False) as data, np.load(run.PRED, allow_pickle=False) as pred, torch.inference_mode():
        for root in (0, 1, 8, 9):
            for side in (-1, 1):
                key = f"r{root}_s{side}"
                frames = torch.stack([transform(x) for x in data[key+"_frames"]]).cuda()
                packets = torch.tensor(scaler.transform(data[key+"_actions"].reshape(15, 2)).reshape(1, 3, 10),
                                       device="cuda", dtype=torch.float32)
                encoded = model.encode({"pixels": frames[None]})["emb"]
                expected = model.predict(encoded, model.action_encoder(packets))[:, -1][0]
                err = float((expected - torch.tensor(pred[key+"_true_history"], device="cuda")).abs().max())
                altered = packets.clone()
                altered[:, :2] = 0
                zero = model.predict(encoded, model.action_encoder(altered))[:, -1][0]
                zero_err = float((zero - torch.tensor(pred[key+"_zero_history"], device="cuda")).abs().max())
                rows.append({"root": root, "side": side, "true_history_abs": err,
                             "zero_history_abs": zero_err,
                             "history_action_change_l2": float(torch.linalg.vector_norm(expected-zero))})
    maximum = max(max(r["true_history_abs"], r["zero_history_abs"]) for r in rows)
    if maximum > 1e-4 or len(rows) != 8:
        raise ValueError("explicit predictor mismatch")
    output = {"rows": rows, "max_abs_error": maximum,
              "history_action_changes": sum(r["history_action_change_l2"] > 1e-7 for r in rows),
              "record_sha256": file_hash(run.RECORD), "runner_sha256": file_hash(__file__),
              "scope": "RGB-only numerical history alignment, not performance confirmation"}
    with OUT.open("x") as file:
        json.dump(output, file, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in output.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
