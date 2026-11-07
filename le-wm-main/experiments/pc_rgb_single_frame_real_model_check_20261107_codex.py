"""Real-checkpoint contract smoke check; not planning performance evidence."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments import pc_tworoom_rgb_retention_fixed_pool_20261107_codex as source
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_single_frame_contract_20261107_codex import SingleFrameRGB, model_digest

OUT = source.ROOT / "results/pc_rgb_single_frame_real_model_check_20261107_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    plan = json.loads(source.PLAN.read_text())
    if plan["input_sha256"] != file_hash(source.inputs.INPUT):
        raise ValueError("input drift")
    model, scaler, transform, action_hash = source.common.model_setup()
    if model_fingerprint(model) != plan["model_sha256"] or action_hash != plan["normalizer_action_sha256"]:
        raise ValueError("checkpoint or normalizer differs from sealed plan")
    digest = model_digest(model)
    audit = freeze_audit(RGB, "lewm", [__file__, source.inputs.INPUT,
                         source.common.CHECKPOINT / "weights.pt", source.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224; action scaler", clock="five five-step packets")
    strict = SingleFrameRGB(model, audit, expected_model_digest=digest, action_mean=scaler.mean_,
                            action_scale=scaler.scale_, packet_steps=5,
                            physical_low=-1., physical_high=1.)
    row = plan["rows"][0]
    with np.load(source.inputs.INPUT, allow_pickle=False) as data:
        idx = int(np.flatnonzero(data["rows"] == row["row"])[0])
        context = {"pixels": transform(data["current"][idx])[None, None, None].cuda(),
                   "goal": transform(data["goal"][idx])[None, None, None].cuda(),
                   "action": torch.empty(1, 1, 0, 10, device="cuda")}
    candidate = torch.tensor(np.asarray(row["generation_actions"][:1], np.float32).reshape(1, 1, 5, 10),
                             device="cuda", requires_grad=True)
    with torch.inference_mode():
        reference, _ = RGBLeWM(model, audit).score(context, candidate.detach())
    output, _ = strict.score(context, candidate)
    gradient, = torch.autograd.grad(output.sum(), candidate)
    torch.cuda.synchronize()
    diff = float((reference - output.detach()).abs().max())
    if diff > 1e-3 or not torch.isfinite(gradient).all() or not (gradient.abs() > 0).any():
        raise ValueError("real-model score parity or action gradients failed")
    history_rejected = False
    try:
        strict.score({**context, "action": torch.zeros(1, 1, 1, 10, device="cuda")}, candidate)
    except ValueError:
        history_rejected = True
    if not history_rejected or not strict.verify_unchanged():
        raise ValueError("history or frozen model contract failed")
    result = {"row": row["row"], "device": torch.cuda.get_device_name(),
              "score_abs_diff": diff, "score": float(output.detach().item()),
              "action_gradient_max_abs": float(gradient.abs().max()),
              "action_gradient_finite": bool(torch.isfinite(gradient).all()),
              "nonempty_history_rejected": history_rejected,
              "model_digest_before_after": digest, "prior_model_fingerprint": plan["model_sha256"],
              "normalizer_action_sha256": action_hash, "plan_sha256": file_hash(source.PLAN),
              "contract_sha256": file_hash(Path(__file__).with_name("pc_rgb_single_frame_contract_20261107_codex.py")),
              "runner_sha256": file_hash(__file__),
              "scope": "one real checkpoint and saved RGB/action; implementation check only"}
    with OUT.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
