"""Compare explicit history assembly against sealed real-RGB predictions."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments import pc_rgb_true_history_alignment_20261107_codex as source
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_history_contract_20261107_codex import score_future

OUTPUT = source.ROOT / "results/pc_rgb_history_contract_gpu_20261107_codex.json"
PROTOCOL = source.ROOT.parent / "docs/PC_RGB_HISTORY_CONTRACT_DEV_PROTOCOL_20261107_CODEX.md"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    prior = json.loads(source.RECORD.read_text())
    if prior["input_sha256"] != file_hash(source.INPUT) or prior["pred_sha256"] != file_hash(source.PRED):
        raise ValueError("previous sealed inputs changed")
    model, scaler, transform, action_hash = source.model_and_scaler()
    fingerprint = model_fingerprint(model)
    if fingerprint != prior["model_sha256"] or action_hash != prior["action_array_sha256"]:
        raise ValueError("model or normalization differs")
    audit = freeze_audit(RGB, "lewm", [Path(__file__), Path(__file__).with_name(
        "pc_rgb_history_contract_20261107_codex.py"), PROTOCOL], preprocessing="same sealed PushT transform",
        clock="three frames and two actual past packets, then one future packet")
    wrapper = RGBLeWM(model, audit)
    rows = []
    print("Check explicit real-history packet assembly against sealed GPU predictions.", flush=True)
    with np.load(source.INPUT, allow_pickle=False) as data, np.load(source.PRED, allow_pickle=False) as prior_pred, \
            torch.inference_mode():
        for root in (0, 1, 8, 9):
            for side in (-1, 1):
                key = f"r{root}_s{side}"
                frames = torch.stack([transform(x) for x in data[key + "_frames"]]).cuda()
                goal = transform(data[key + "_goal"]).cuda()
                packets = torch.tensor(scaler.transform(data[key + "_actions"].reshape(15, 2))
                                       .reshape(1, 1, 3, 10), dtype=torch.float32, device="cuda")
                _, context = score_future(wrapper, pixels=frames[None, None], goal=goal[None, None, None],
                                          history=packets[:, :, :2], future=packets[:, :, 2:])
                predicted = context["predicted_emb"][0, 0, -1].cpu().numpy()
                reference = prior_pred[key + "_true_history"]
                np.testing.assert_allclose(predicted, reference, atol=1e-6, rtol=1e-5)
                rows.append({"root": root, "side": side,
                             "max_abs_difference": float(np.max(np.abs(predicted - reference)))})
    if model_fingerprint(model) != fingerprint:
        raise ValueError("frozen model changed")
    record = {"track": RGB, "rows": rows, "model_sha256": fingerprint,
              "action_array_sha256": action_hash, "input_sha256": file_hash(source.INPUT),
              "prior_record_sha256": file_hash(source.RECORD), "prior_predictions_sha256": file_hash(source.PRED),
              "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
              "implementation_sha256": file_hash(Path(__file__).with_name("pc_rgb_history_contract_20261107_codex.py")),
              "scope": "exposed RGB development history; no physical or closed-loop claim"}
    with OUTPUT.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps({"rows": len(rows), "max_abs_difference": max(x["max_abs_difference"] for x in rows)}))


if __name__ == "__main__":
    main()
