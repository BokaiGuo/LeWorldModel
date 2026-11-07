"""Recompute sealed temporal candidates and audit descriptor implementation."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments import pc_temporal_effect_retention_20261107_codex as src
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

OUTPUT = src.ROOT / "results/pc_temporal_effect_integrity_audit_20261107_codex.json"
PROTOCOL = src.ROOT.parent / "docs/PC_TEMPORAL_EFFECT_INTEGRITY_AUDIT_PROTOCOL_20261107_CODEX.md"


def centered_effect(costs):
    values = np.asarray(costs, dtype=np.float64)
    if values.shape != (8, 9, 5) or not np.isfinite(values).all():
        raise ValueError("expected finite scores")
    post = values[:, :, 2:].min(axis=-1)
    effect = post[:, :1] - post[:, 1:]
    return effect - effect.mean(axis=1, keepdims=True)


def record_root(root, saved_a, actual_a, saved_b, actual_b, saved_inc, actual_inc, ids, decisions):
    old_keep, new_keep = src.retention(saved_a, 29114000 + root), src.retention(actual_a, 29114000 + root)
    rows = {}
    for method in src.METHODS:
        old_choice = src.choose(saved_a[:, 1:].min(-1), saved_b.min(-1), ids, old_keep[method],
                                saved_inc.min(-1))
        new_choice = src.choose(actual_a[:, 1:].min(-1), actual_b.min(-1), ids, new_keep[method],
                                actual_inc.min(-1))
        declared = decisions[method]
        if (old_choice[1:] != (declared["bank"], declared["index"])
                or old_keep[method] != declared["retained"]
                or abs(old_choice[0] - declared["cost"]) > 1e-9):
            raise ValueError("sealed decision cannot be reproduced")
        rows[method] = {"retained_changed": old_keep[method] != new_keep[method],
                        "choice_changed": old_choice[1:] != new_choice[1:],
                        "sealed": list(old_choice[1:]), "recomputed": list(new_choice[1:])}
    probes = saved_a[:, 1:, 2:].min(-1).astype(np.float64)
    reference_cancellation = float(np.max(np.abs(centered_effect(saved_a)
                                                 + probes - probes.mean(axis=1, keepdims=True))))
    return {"root": root, "max_abs_a": float(np.max(np.abs(saved_a - actual_a))),
            "max_abs_b": float(np.max(np.abs(saved_b - actual_b))),
            "max_abs_incumbent": float(np.max(np.abs(saved_inc - actual_inc))),
            "reference_cancellation_error": reference_cancellation, "methods": rows}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan = json.loads(src.PLAN.read_text())
    paths = {"pool": src.POOL, "old_plan": src.old.PLAN, "old_pool": src.old.POOL,
             "initial_pool": src.old.base.source.POOL, "temporal_plan": src.temporal.PLAN,
             "temporal_scores": src.temporal.SCORES, "inputs": src.old.base.source.block.prior.inputs.INPUT,
             "protocol": src.PROTOCOL, "runner": src.__file__}
    if any(plan["sha256"][k] != file_hash(v) for k, v in paths.items()):
        raise ValueError("source binding drift")
    common = src.old.base.source.block.prior.common
    model, scaler, transform, normalizer = common.model_setup()
    if model_fingerprint(model) != plan["model_sha256"] or normalizer != plan["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, src.PLAN, *paths.values()],
                         preprocessing="ImageNet 224 and action scaler",
                         clock="five packets of five environment steps")
    rows = []
    with (np.load(src.old.POOL, allow_pickle=False) as old,
          np.load(src.old.base.source.POOL, allow_pickle=False) as initial,
          np.load(src.temporal.SCORES, allow_pickle=False) as temporal,
          np.load(src.POOL, allow_pickle=False) as saved,
          np.load(paths["inputs"], allow_pickle=False) as inputs, torch.inference_mode()):
        wrapper = RGBLeWM(model, audit)
        for task in plan["rows"]:
            root = task["root"]
            a = src.temporal.source.chains_for(old, root)
            b_ids = saved[f"r{root}_b_ids"]
            b = old[f"r{root}_b"].reshape(64, 25, 2)[b_ids]
            inc = np.asarray([initial[f"r{root}_{arm}_actions"][
                int(np.argmin(initial[f"r{root}_{arm}_scores"]))] for arm in ("block_cem", "block_random")])
            all_actions = np.concatenate((a, b, inc))
            if not np.isfinite(all_actions).all() or np.any(np.abs(all_actions) > 1):
                raise ValueError("invalid action")
            normalized = scaler.transform(all_actions.reshape(-1, 2)).reshape(-1, 5, 10)
            ctx = {"pixels": transform(inputs["current"][root])[None, None, None].cuda(),
                   "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                   "action": torch.empty((1, 1, 0, 10), device="cuda")}
            chunks = []
            for i in range(0, len(normalized), 32):
                batch = torch.as_tensor(normalized[None, i:i+32], dtype=torch.float32, device="cuda")
                _, out = wrapper.score(ctx, batch)
                if out["predicted_emb"].shape[2] != 6:
                    raise ValueError("prediction clock mismatch")
                costs = (out["predicted_emb"][:, :, 1:] - out["goal_emb"][:, -1, None, None]).square().sum(-1)
                chunks.append(costs[0].cpu().numpy())
            fresh = np.concatenate(chunks)
            for i, method in enumerate(src.METHODS):
                choice = task["decisions"][method]
                selected = inc[choice["index"]] if choice["bank"] == "incumbent" else old[
                    f"r{root}_{choice['bank']}"].reshape(64, 25, 2)[choice["index"]]
                if (src.digest(selected) != choice["action_sha256"]
                        or not np.array_equal(selected, saved[f"r{root}_actions"][i])):
                    raise ValueError("sealed action does not match source")
            rows.append(record_root(root, temporal[f"r{root}_scores"], fresh[:72].reshape(8, 9, 5),
                                    saved[f"r{root}_b_times"], fresh[72:-2],
                                    saved[f"r{root}_inc_times"], fresh[-2:], b_ids, task["decisions"]))
            print(f"root={root}: verified {len(all_actions)} chains", flush=True)
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("model weights changed")
    if [x["root"] for x in rows] != list(range(20, 50)):
        raise ValueError("incomplete roots")
    max_error = max(max(r["max_abs_a"], r["max_abs_b"], r["max_abs_incumbent"]) for r in rows)
    changes = {m: {"retained": sum(r["methods"][m]["retained_changed"] for r in rows),
                   "choice": sum(r["methods"][m]["choice_changed"] for r in rows)} for m in src.METHODS}
    result = {"rows": rows, "max_abs_score_error": max_error,
              "max_reference_cancellation_error": max(r["reference_cancellation_error"] for r in rows),
              "changes": changes, "numerically_consistent": max_error <= 1e-3
              and all(changes[m]["choice"] == 0 for m in src.METHODS),
              "plan_sha256": file_hash(src.PLAN), "protocol_sha256": file_hash(PROTOCOL),
              "runner_sha256": file_hash(__file__), "scope": "exposed development numerical audit"}
    with OUTPUT.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
