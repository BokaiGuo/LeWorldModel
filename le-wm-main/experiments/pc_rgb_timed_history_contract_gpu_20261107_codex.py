"""Compare strict timed histories with previously sealed RGB-only predictions."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments import pc_rgb_true_history_isolated_20261107_codex as old
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_single_frame_contract_20261107_codex import SingleFrameRGB, model_digest
from experiments.pc_rgb_timed_history_contract_20261107_codex import TimedHistoryRGB

OUTPUT = old.DIRECTORY / "pc_rgb_timed_history_contract_gpu_20261107_codex.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    prior = json.loads(old.RECORD.read_text())
    if prior["input_sha256"] != file_hash(old.base.INPUT) or prior["pred_sha256"] != file_hash(old.PREDICTIONS):
        raise ValueError("sealed input or prediction drift")
    model, scaler, transform, action_hash = old.base.model_and_scaler()
    if model_fingerprint(model) != prior["model_sha256"] or action_hash != prior["action_array_sha256"]:
        raise ValueError("model or normalizer drift")
    fingerprint = model_digest(model)
    audit = freeze_audit(RGB, "lewm", [__file__, old.base.INPUT, old.PREDICTIONS,
                         Path(__file__).with_name("pc_rgb_timed_history_contract_20261107_codex.py")],
                         preprocessing="PushT ImageNet and original scaler",
                         clock="0,5,10 observed; 10-to-15 future")
    kwargs = dict(expected_model_digest=fingerprint, action_mean=scaler.mean_, action_scale=scaler.scale_,
                  packet_steps=5, physical_low=-1., physical_high=1.)
    timed = TimedHistoryRGB(model, audit, max_frames=3, **kwargs)
    single = SingleFrameRGB(model, audit, **kwargs)
    rows = []
    with np.load(old.base.INPUT, allow_pickle=False) as data, \
            np.load(old.PREDICTIONS, allow_pickle=False) as predictions, torch.inference_mode():
        for root in (0, 1, 8, 9):
            for side in (-1, 1):
                key = f"r{root}_s{side}"
                frames = torch.stack([transform(frame) for frame in data[key + "_frames"]]).cuda()
                goal = transform(data[key + "_goal"]).cuda()[None, None, None]
                packets = torch.tensor(scaler.transform(data[key + "_actions"].reshape(15, 2))
                                       .reshape(1, 1, 3, 10), dtype=torch.float32, device="cuda")
                value, context = timed.score_timed(pixels=frames[None, None], goal=goal,
                                                   history=packets[:, :, :2], future=packets[:, :, 2:],
                                                   frame_steps=(0, 5, 10), current_step=10)
                pred = context["predicted_emb"][0, 0, -1].cpu().numpy()
                truth = predictions[key + "_true_history"]
                np.testing.assert_allclose(pred, truth, atol=1e-6, rtol=1e-5)
                empty = torch.empty(1, 1, 0, 10, device="cuda")
                single_value, _ = single.score({"pixels": frames[-1:][None, None], "goal": goal,
                                                "action": empty}, packets[:, :, 2:])
                timed_value, single_context = timed.score_timed(
                    pixels=frames[-1:][None, None], goal=goal, history=empty, future=packets[:, :, 2:],
                    frame_steps=(10,), current_step=10)
                np.testing.assert_allclose(timed_value.cpu(), single_value.cpu(), atol=1e-6, rtol=1e-5)
                np.testing.assert_allclose(single_context["predicted_emb"][0, 0, -1].cpu(),
                                           predictions[key + "_single"], atol=1e-6, rtol=1e-5)
                rows.append({"root": root, "side": side,
                             "history_prediction_max_abs": float(np.max(np.abs(pred-truth))),
                             "single_score_max_abs": float((single_value-timed_value).abs().max()),
                             "history_score": float(value.item())})
    timed.verify_unchanged()
    single.verify_unchanged()
    result = {"rows": rows, "model_digest": fingerprint, "prior_record_sha256": file_hash(old.RECORD),
              "prior_prediction_sha256": file_hash(old.PREDICTIONS), "input_sha256": file_hash(old.base.INPUT),
              "contract_sha256": file_hash(Path(__file__).with_name("pc_rgb_timed_history_contract_20261107_codex.py")),
              "runner_sha256": file_hash(__file__),
              "scope": "exposed RGB physical histories; implementation parity, no performance claim"}
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"rows": len(rows),
                      "max_history_prediction_error": max(r["history_prediction_max_abs"] for r in rows),
                      "max_single_score_error": max(r["single_score_max_abs"] for r in rows)}))


if __name__ == "__main__":
    main()
