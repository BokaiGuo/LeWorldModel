"""Cross-check saved serial refinement scores with native batched RGB scoring."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments import pc_tworoom_joint_random_dev_20261107_codex as src
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_rgb_observation_cache_20261107_codex import RGBObservationScoringSession

OUT = src.ref.ROOT / "results/pc_tworoom_refinement_score_crosscheck_20261107_codex.json"
PROTOCOL = src.ref.ROOT.parent / "docs/PC_TWOROOM_REFINEMENT_SCORE_CROSSCHECK_PROTOCOL_20261107_CODEX.md"


def compare(original, other, actions, tolerance=0.005):
    original, other = np.asarray(original), np.asarray(other)
    actions = np.asarray(actions)
    if original.shape != (9,) or other.shape != (9,) or actions.shape != (9, 5, 2):
        raise ValueError("expected nine aligned candidates")
    if not np.isfinite(original).all() or not np.isfinite(other).all():
        raise ValueError("nonfinite score")
    i, j = int(np.argmin(original)), int(np.argmin(other))
    return {"max_abs_error": float(np.max(np.abs(original - other))),
            "within_tolerance": bool(np.max(np.abs(original - other)) <= tolerance),
            "old_index": i, "new_index": j, "index_changed": i != j,
            "action_changed": not np.array_equal(actions[i], actions[j]),
            "old_margin": float(np.sort(original, kind="stable")[1] - original[i]),
            "new_margin": float(np.sort(other, kind="stable")[1] - other[j]),
            "rescored": other.tolist()}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    p = json.loads(src.PLAN.read_text())
    if file_hash(src.POOL) != p["pool_sha256"]:
        raise ValueError("sealed pool changed")
    for name, digest in p["source_sha256"].items():
        if file_hash(name) != digest:
            raise ValueError(f"source changed: {name}")
    prior = json.loads(src.q.PLAN.read_text())
    common = src.q.old.base.source.block.prior.common
    input_path = src.q.old.base.source.block.prior.inputs.INPUT
    model, scaler, transform, normalizer = common.model_setup()
    if src.ref.model_fingerprint(model) != p["model_sha256"] or normalizer != p["normalizer_sha256"]:
        raise ValueError("model or normalizer changed")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, src.PLAN, src.POOL, input_path,
                         common.CHECKPOINT / "weights.pt"], preprocessing="ImageNet 224",
                         clock="five packets each of five raw environment steps")
    rows = []
    with np.load(src.POOL, allow_pickle=False) as pool, np.load(input_path, allow_pickle=False) as data, \
            src.backend.precise_action_derivatives(), torch.inference_mode():
        for item in prior["rows"]:
            root = item["root"]
            ctx = {"pixels": transform(data["current"][root])[None, None, None].cuda(),
                   "goal": transform(data["goal"][root])[None, None, None].cuda(),
                   "action": torch.empty((1, 1, 0, 10), device="cuda")}
            session = RGBObservationScoringSession(model, audit)
            row = {"root": root, "arms": {}}
            for arm in ("adam", "random"):
                blocks = pool[f"r{root}_{arm}_actions"]
                raw = np.repeat(blocks, 5, axis=1)
                normalized = scaler.transform(raw.reshape(-1, 2)).reshape(1, 9, 5, 10)
                values, _ = session.score(ctx, torch.as_tensor(normalized, dtype=torch.float32, device="cuda"))
                row["arms"][arm] = compare(pool[f"r{root}_{arm}_scores"], values[0].cpu().numpy(), blocks)
            rows.append(row)
            print(f"root={root} checked", flush=True)
    if src.ref.model_fingerprint(model) != p["model_sha256"]:
        raise ValueError("model mutated")
    summary = {arm: {"max_abs_error": max(r["arms"][arm]["max_abs_error"] for r in rows),
                     "exceed_tolerance_roots": [r["root"] for r in rows if not r["arms"][arm]["within_tolerance"]],
                     "index_change_roots": [r["root"] for r in rows if r["arms"][arm]["index_changed"]],
                     "action_change_roots": [r["root"] for r in rows if r["arms"][arm]["action_changed"]]}
               for arm in ("adam", "random")}
    with OUT.open("x") as f:
        json.dump({"protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "plan_sha256": file_hash(src.PLAN), "pool_sha256": file_hash(src.POOL),
                   "tolerance": 0.005, "summary": summary, "rows": rows}, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
