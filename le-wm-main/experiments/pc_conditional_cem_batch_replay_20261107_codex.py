"""Rescore sealed conditional CEM candidates with controlled batch sizes."""
import json
import time

import numpy as np
import torch

from experiments import pc_rgb_conditional_temporal_cem_20261107_codex as src
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = src.old.ROOT
PROTOCOL = ROOT.parent / "docs/PC_CONDITIONAL_CEM_BATCH_REPLAY_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_conditional_cem_batch_replay_20261107_codex.json"
SCORES = ROOT / "results/pc_conditional_cem_batch_replay_20261107_codex.npz"
SIZES = (16, 32, 64)


def decisions(inc, actions, inc_scores, scores, mode):
    """Selection on an already scored fixed population, not a new CEM run."""
    inc = np.asarray(inc)
    actions = np.asarray(actions)
    inc_scores = np.asarray(inc_scores)
    scores = np.asarray(scores)
    if (inc.shape != (2, 25, 2) or actions.shape != (src.ROUNDS * src.N, 25, 2)
            or inc_scores.shape != (2, 5) or scores.shape != (src.ROUNDS * src.N, 5)
            or mode not in src.METHODS
            or not all(np.isfinite(x).all() for x in (inc, actions, inc_scores, scores))):
        raise ValueError("invalid fixed population")
    values = scores.reshape(src.ROUNDS, src.N, 5)
    objective = values[:, :, -1] if mode == "endpoint" else values[:, :, 2:].min(axis=-1)
    elites = [np.argsort(row, kind="stable")[:src.K].tolist() for row in objective]
    chosen, index, cost = src.choose(inc, inc_scores, actions, scores)
    return {"elite": elites, "index": index, "action_sha256": src.digest(chosen), "cost": cost}


def compare(reference, other):
    return {"elite_set_changed": [set(a) != set(b) for a, b in zip(reference["elite"], other["elite"], strict=True)],
            "choice_index_changed": reference["index"] != other["index"],
            "choice_action_changed": reference["action_sha256"] != other["action_sha256"]}


def main():
    if OUTPUT.exists() or SCORES.exists():
        raise FileExistsError("output already exists")
    plan = json.loads(src.PLAN.read_text())
    source = src.old.base.source
    inputs_path = source.block.prior.inputs.INPUT
    paths = {"pool": src.POOL, "old_plan": src.old.PLAN, "source_pool": source.POOL,
             "inputs": inputs_path, "protocol": src.PROTOCOL, "runner": src.__file__}
    if any(plan["sha256"][k] != file_hash(v) for k, v in paths.items()):
        raise ValueError("sealed experiment dependency drift")
    common = source.block.prior.common
    model, scaler, transform, normalizer = common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != plan["model_sha256"] or normalizer != plan["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, src.PLAN, *paths.values(),
                                      common.CHECKPOINT / "weights.pt"],
                         preprocessing="ImageNet 224 and action scaler",
                         clock="five packets of five environment steps")
    rows, arrays = [], {}
    print("Rescoring identical sealed actions; fixed-pool analysis is not an adaptive CEM rerun.", flush=True)
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    with (np.load(src.POOL, allow_pickle=False) as pool,
          np.load(inputs_path, allow_pickle=False) as data, torch.inference_mode()):
        wrapper = RGBLeWM(model, audit)
        for task in plan["rows"]:
            root = task["root"]
            inc = pool[f"r{root}_inc"]
            actions = pool[f"r{root}_actions"]
            original_inc = pool[f"r{root}_inc_scores"]
            original_scores = pool[f"r{root}_scores"]
            ctx = {"pixels": transform(data["current"][root])[None, None, None].cuda(),
                   "goal": transform(data["goal"][root])[None, None, None].cuda(),
                   "action": torch.empty((1, 1, 0, 10), device="cuda")}

            def score(chains, batch_size):
                if (chains.ndim != 3 or chains.shape[1:] != (25, 2)
                        or not np.isfinite(chains).all() or np.any(np.abs(chains) > 1)):
                    raise ValueError("invalid actions")
                norm = scaler.transform(chains.reshape(-1, 2)).reshape(-1, 5, 10)
                output = []
                for offset in range(0, len(norm), batch_size):
                    batch = torch.as_tensor(norm[None, offset:offset + batch_size],
                                            dtype=torch.float32, device="cuda")
                    endpoint, context = wrapper.score(ctx, batch)
                    pred, goal = context["predicted_emb"], context["goal_emb"]
                    if pred.shape[2] != 6:
                        raise ValueError("prediction clock mismatch")
                    values = (pred[:, :, 1:] - goal[:, -1, None, None]).square().sum(-1)
                    if not torch.allclose(endpoint, values[:, :, -1], rtol=0, atol=0.001):
                        raise ValueError("endpoint mismatch")
                    output.append(values[0].cpu().numpy())
                return np.concatenate(output)

            new = {}
            for size in SIZES:
                inc_scores = score(inc, size)
                arm_scores = np.stack([
                    np.concatenate([score(actions[i, r * src.N:(r + 1) * src.N], size)
                                    for r in range(src.ROUNDS)])
                    for i in range(len(src.METHODS))])
                arrays[f"r{root}_inc_{size}"] = inc_scores
                arrays[f"r{root}_scores_{size}"] = arm_scores
                new[size] = {"inc": inc_scores, "scores": arm_scores}
            method_rows = {}
            for i, method in enumerate(src.METHODS):
                base = decisions(inc, actions[i], original_inc, original_scores[i], method)
                if (base["index"] != task["decisions"][method]["index"]
                        or base["action_sha256"] != task["decisions"][method]["action_sha256"]
                        or base["elite"] != [r["elite"] for r in task["decisions"][method]["iterations"]]):
                    raise ValueError("original decision mismatch")
                by_size = {size: decisions(inc, actions[i], new[size]["inc"], new[size]["scores"][i], method)
                           for size in SIZES}
                method_rows[method] = {
                    str(size): {"max_abs_vs_saved": float(max(
                        np.max(np.abs(new[size]["inc"] - original_inc)),
                        np.max(np.abs(new[size]["scores"][i] - original_scores[i])))),
                        "max_abs_vs_32": float(max(
                            np.max(np.abs(new[size]["inc"] - new[32]["inc"])),
                            np.max(np.abs(new[size]["scores"][i] - new[32]["scores"][i])))),
                        "vs_saved": compare(base, by_size[size]),
                        "vs_32": compare(by_size[32], by_size[size]),
                        "decision": by_size[size]} for size in SIZES}
            rows.append({"root": root, "methods": method_rows})
            print(f"root={root}: fixed action pools scored at batch sizes {SIZES}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    if [r["root"] for r in rows] != list(range(20, 50)):
        raise ValueError("incomplete panel")
    with SCORES.open("xb") as file:
        np.savez_compressed(file, **arrays)
    summary = {}
    for size in SIZES:
        for ref in ("vs_saved", "vs_32"):
            cells = [r["methods"][m][str(size)] for r in rows for m in src.METHODS]
            summary[f"{size}_{ref}"] = {
                "max_abs_score_error": max(c["max_abs_" + ref] for c in cells),
                "changed_elite_rounds": sum(sum(c[ref]["elite_set_changed"]) for c in cells),
                "changed_choice_index": sum(c[ref]["choice_index_changed"] for c in cells),
                "changed_choice_action": sum(c[ref]["choice_action_changed"] for c in cells)}
    result = {"summary": summary, "rows": rows, "track": RGB,
              "source_plan_sha256": file_hash(src.PLAN), "source_pool_sha256": file_hash(src.POOL),
              "scores_sha256": file_hash(SCORES), "protocol_sha256": file_hash(PROTOCOL),
              "runner_sha256": file_hash(__file__), "model_sha256": fingerprint,
              "normalizer_sha256": normalizer, "gpu_seconds_synchronized": elapsed,
              "scope": "fixed-pool exposed development audit, not adaptive search or physical replay"}
    with OUTPUT.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
