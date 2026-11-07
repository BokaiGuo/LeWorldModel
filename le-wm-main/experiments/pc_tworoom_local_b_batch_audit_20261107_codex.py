"""Re-score both fixed-search arms with identical GPU batching."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments import pc_tworoom_local_b_tail_20261107_codex as local
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_observation_cache_20261107_codex import RGBObservationScoringSession

ROOT = local.q.ROOT
OUTPUT = ROOT / "results/pc_tworoom_local_b_batch_audit_20261107_codex.json"
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_LOCAL_B_BATCH_AUDIT_PROTOCOL_20261107_CODEX.md"


def select(scores, b_start):
    """Inputs are 2 incumbents, 64 A, 16 uniform B, 16 local B."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (98,) or not np.isfinite(scores).all() or b_start not in (66, 82):
        raise ValueError("invalid scores")
    eligible = np.concatenate((np.arange(66), np.arange(b_start, b_start + 16)))
    order = np.argsort(scores[eligible], kind="stable")
    return int(eligible[order[0]]), float(scores[eligible[order[1]]] - scores[eligible[order[0]]])


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    q = local.q
    plan = json.loads(local.PLAN.read_text())
    previous = json.loads(q.PLAN.read_text())
    if (plan["pool_sha256"] != file_hash(local.POOL)
            or any(file_hash(Path(path)) != digest for path, digest in plan["source_sha256"].items())
            or previous["source_pool_sha256"] != file_hash(q.old.base.source.POOL)):
        raise ValueError("source drift")
    common = q.old.base.source.block.prior.common
    inputs_path = q.old.base.source.block.prior.inputs.INPUT
    model, scaler, transform, normalizer = common.model_setup()
    if model_fingerprint(model) != plan["model_sha256"] or normalizer != plan["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, local.PLAN, local.POOL, q.PLAN, q.POOL,
                         q.old.POOL, q.old.base.source.POOL, inputs_path, common.CHECKPOINT / "weights.pt"],
                         preprocessing="ImageNet 224, TwoRoom scaler", clock="five five-step packets")
    rows = []
    with (np.load(local.POOL, allow_pickle=False) as localpool,
          np.load(q.POOL, allow_pickle=False) as oldcost,
          np.load(q.old.POOL, allow_pickle=False) as banks,
          np.load(q.old.base.source.POOL, allow_pickle=False) as initial,
          np.load(inputs_path, allow_pickle=False) as inputs, torch.inference_mode()):
        for entry, prior in zip(plan["rows"], previous["rows"], strict=True):
            root = entry["root"]
            if root != prior["root"]:
                raise ValueError("root mismatch")
            incumbent = []
            for name in ("block_cem", "block_random"):
                costs = initial[f"r{root}_{name}_scores"]
                incumbent.append(initial[f"r{root}_{name}_actions"][int(np.argmin(costs))])
            a = banks[f"r{root}_a"].reshape(64, 25, 2)
            b = banks[f"r{root}_b"].reshape(64, 25, 2)
            ids = np.sort((np.asarray(entry["keep"])[:, None] * 8 + np.arange(8)).ravel())
            prior_ids = oldcost[f"r{root}_b_ids"]
            locations = np.searchsorted(prior_ids, ids)
            if np.any(locations >= len(prior_ids)) or not np.array_equal(prior_ids[locations], ids):
                raise ValueError("missing uniform candidate scores")
            new = localpool[f"r{root}_local_candidates"]
            chains = np.concatenate((np.asarray(incumbent), a, b[ids], new)).astype(np.float32)
            historic = np.concatenate((oldcost[f"r{root}_inc_cost"], oldcost[f"r{root}_a_cost"].ravel(),
                                       oldcost[f"r{root}_b_cost"][locations],
                                       localpool[f"r{root}_local_cost"]))
            normalized = scaler.transform(chains.reshape(-1, 2)).reshape(98, 5, 10)
            context = {"pixels": transform(inputs["current"][root])[None, None, None].cuda(),
                       "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            session = RGBObservationScoringSession(model, audit)
            values = []
            for offset in range(0, 98, 32):
                packet = torch.as_tensor(normalized[offset:offset+32][None], dtype=torch.float32, device="cuda")
                value, _ = session.score(context, packet)
                values.extend(value[0].cpu().tolist())
            values = np.asarray(values, dtype=np.float64)
            choices = {}
            for name, start in (("uniform", 66), ("local", 82)):
                index, gap = select(values, start)
                selected_hash = q.digest(chains[index])
                choices[name] = {"selected_index": index, "score": float(values[index]), "gap": gap,
                                 "action_sha256": selected_hash,
                                 "same_as_sealed": selected_hash == entry[f"{name}_action_sha256"]}
            rows.append({"root": root, "scores": values.tolist(), "historical_scores": historic.tolist(),
                         "max_abs_score_diff": float(np.max(np.abs(values - historic))), "choices": choices})
            print(f"root={root} max_diff={rows[-1]['max_abs_score_diff']:.8g}", flush=True)
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("model changed")
    paths = [Path(__file__), PROTOCOL, local.PLAN, local.POOL, q.PLAN, q.POOL, q.old.POOL,
             q.old.base.source.POOL, inputs_path]
    with OUTPUT.open("x") as f:
        json.dump({"scope": "exposed development numerical replay, not independent confirmation",
                   "source_sha256": {str(p): file_hash(p) for p in paths},
                   "model_sha256": plan["model_sha256"], "normalizer_sha256": normalizer,
                   "rows": rows}, f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
