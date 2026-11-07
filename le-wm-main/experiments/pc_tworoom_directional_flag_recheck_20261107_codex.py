"""Post-hoc small-step numerical checks for predeclared AD/FD flags."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments import pc_tworoom_rgb_directional_alignment_20261107_codex as prior
from experiments.pc_input_permissions import RGB, freeze_audit, file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_single_frame_contract_20261107_codex import SingleFrameRGB, model_digest

OUT = prior.source.ROOT / "results/pc_tworoom_directional_flag_recheck_20261107_codex.json"
AUDIT = prior.source.ROOT / "results/pc_tworoom_rgb_directional_alignment_audit_20261107_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    plan, audit_record = json.loads(prior.PLAN.read_text()), json.loads(AUDIT.read_text())
    if audit_record["plan_sha256"] != file_hash(prior.PLAN):
        raise ValueError("audit does not bind plan")
    flagged = [(r["root"], d) for r in audit_record["rows"] for d, value in
               enumerate(r["small_scale_flagged"]) if value]
    if flagged != [(2, 3), (4, 0)]:
        raise ValueError("unexpected predeclared flags")
    model, scaler, transform, action_hash = prior.source.common.model_setup()
    if model_fingerprint(model) != plan["model_sha256"] or action_hash != plan["normalizer_sha256"]:
        raise ValueError("model drift")
    digest = model_digest(model)
    audit = freeze_audit(RGB, "lewm", [__file__, prior.PLAN, AUDIT, prior.source.inputs.INPUT],
                         preprocessing="ImageNet 224; action scaler", clock="five five-step packets")
    guard = SingleFrameRGB(model, audit, expected_model_digest=digest, action_mean=scaler.mean_,
                           action_scale=scaler.scale_, packet_steps=5, physical_low=-1., physical_high=1.)
    scales = (.001, .002, .004, .008)
    rows = []
    with np.load(prior.source.inputs.INPUT, allow_pickle=False) as data:
        for root, direction in flagged:
            r = next(item for item in plan["rows"] if item["root"] == root)
            ix = int(np.flatnonzero(data["rows"] == r["row"])[0])
            context = {"pixels": transform(data["current"][ix])[None, None, None].cuda(),
                       "goal": transform(data["goal"][ix])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            center = scaler.transform(np.zeros((25, 2))).reshape(1, 1, 5, 10)
            x = torch.tensor(center, dtype=torch.float32, device="cuda", requires_grad=True)
            value, _ = guard.score(context, x)
            gradient, = torch.autograd.grad(value.sum(), x)
            ad = float((gradient.detach().cpu().numpy().reshape(25, 2) /
                        scaler.scale_ * prior.directions()[direction]).sum())
            raws = np.stack([prior.directions()[direction] * h * s for h in scales for s in (1, -1)])
            normalized = scaler.transform(raws.reshape(-1, 2)).reshape(1, 8, 5, 10)
            with torch.inference_mode():
                batch, _ = guard.score(context, torch.tensor(normalized, device="cuda", dtype=torch.float32))
                individual = []
                for candidate in normalized[0]:
                    cost, _ = guard.score(context, torch.tensor(candidate[None, None], device="cuda",
                                                               dtype=torch.float32))
                    individual.append(float(cost.item()))
            batch = batch[0].cpu().numpy()
            individual = np.asarray(individual)
            rows.append({"root": root, "direction": direction, "ad": ad,
                         "batch_costs": batch.tolist(), "individual_costs": individual.tolist(),
                         "batch_individual_max_abs": float(np.max(np.abs(batch - individual))),
                         "batch_fd": [prior.finite_difference(batch[i*2], batch[i*2+1], h)
                                      for i, h in enumerate(scales)],
                         "individual_fd": [prior.finite_difference(individual[i*2], individual[i*2+1], h)
                                           for i, h in enumerate(scales)]})
    guard.verify_unchanged()
    with OUT.open("x") as f:
        json.dump({"scales": scales, "rows": rows, "audit_sha256": file_hash(AUDIT),
                   "runner_sha256": file_hash(__file__),
                   "scope": "post-hoc diagnostics only; no altered thresholds"}, f, indent=2, allow_nan=False)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
