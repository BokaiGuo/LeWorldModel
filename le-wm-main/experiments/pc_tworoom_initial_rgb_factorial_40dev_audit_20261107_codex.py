"""Audit all physical trace bindings and GPU-rescore pixel-sensitive roots."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments import pc_tworoom_initial_rgb_factorial_40dev_20261107_codex as exp
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit

OUTPUT = Path(str(exp.PREFIX) + "_audit.json")
ROOTS = (14, 42, 49)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan, phys, result = map(lambda p: json.loads(p.read_text()), (exp.PLAN, exp.PHYSICAL, exp.RESULT))
    if (result["plan_sha256"] != file_hash(exp.PLAN) or result["physical_sha256"] != file_hash(exp.PHYSICAL)
            or plan["pool_sha256"] != file_hash(exp.POOL)
            or plan["source_pool_sha256"] != file_hash(exp.source.base.source.POOL)):
        raise ValueError("artifact drift")
    model, scaler, transform, normalizer = exp.source.base.source.block.prior.common.model_setup()
    if exp.source.model_fingerprint(model) != plan["model_sha256"] or normalizer != plan["normalizer_sha256"]:
        raise ValueError("model mismatch")
    audit = freeze_audit(RGB, "lewm", [__file__, exp.PLAN, exp.POOL, exp.PROTOCOL],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="five packets of five steps")
    checked, inversion_roots = [], []
    with (np.load(exp.POOL, allow_pickle=False) as pool,
          np.load(exp.source.base.source.POOL, allow_pickle=False) as old,
          torch.inference_mode()):
        for item, physical in zip(plan["rows"], phys["rows"], strict=True):
            root = item["root"]
            if root != physical["root"]:
                raise ValueError("root mismatch")
            actions = pool[f"r{root}_actions"]
            for index, identity in enumerate(item["identities"]):
                np.testing.assert_array_equal(actions[index],
                                              old[f"r{root}_{identity['bank']}_actions"][identity["index"]])
            scores = pool[f"r{root}_scores"]
            for arm, cost in zip(exp.ARMS, scores, strict=True):
                choice, margin = exp.select(cost)
                if choice != item["choices"][arm] or margin != item["margins"][arm]:
                    raise ValueError("choice mismatch")
                same_arm = next(other for other in exp.ARMS if item["choices"][other] == choice)
                if physical["outcomes"][arm]["trace"] != physical["outcomes"][same_arm]["trace"]:
                    raise ValueError("identical action physical trajectory differs")
            n = sum((scores[0, i]-scores[0, j]) * (scores[1, i]-scores[1, j]) < 0
                    for i in range(32) for j in range(i+1, 32))
            if n:
                inversion_roots.append({"root": root, "strict_pair_inversions": int(n)})
            if root in ROOTS:
                images = pool[f"r{root}_images"]
                norm = scaler.transform(actions.reshape(-1, 2)).reshape(1, 32, 5, 10)
                batch = torch.tensor(norm, dtype=torch.float32, device="cuda")
                errors = []
                for index, (current, goal) in enumerate(((0, 2), (1, 2), (0, 3), (1, 3))):
                    ctx = {"pixels": transform(images[current])[None, None, None].cuda(),
                           "goal": transform(images[goal])[None, None, None].cuda(),
                           "action": torch.empty((1, 1, 0, 10), device="cuda")}
                    cost, _ = RGBLeWM(model, audit).score(ctx, batch)
                    errors.append(float(np.abs(cost[0].cpu().numpy() - scores[index]).max()))
                if max(errors) > 1e-3:
                    raise ValueError(f"score mismatch {root}: {errors}")
                checked.append({"root": root, "max_gpu_score_error": max(errors)})
                print(f"root={root} verified", flush=True)
    if exp.source.model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("weights changed")
    with OUTPUT.open("x") as f:
        json.dump({"gpu_replayed": checked, "pair_inversions": inversion_roots,
                   "physical_roots": len(phys["rows"]), "result_sha256": file_hash(exp.RESULT),
                   "scope": "same model/environment, fixed candidate panel"}, f, indent=2)


if __name__ == "__main__":
    main()
