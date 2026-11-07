"""Audit sealed candidate pools and replay a fixed subset of model scores."""
import json

import numpy as np
import torch

from experiments import pc_tworoom_block_cem_50dev_20261107_codex as exp
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

OUTPUT = exp.PREFIX.with_name(exp.PREFIX.name + "_audit.json")
REPLAY_ROOTS = (0, 9, 25, 49)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan = json.loads(exp.PLAN.read_text())
    physical = json.loads(exp.PHYSICAL.read_text())
    result = json.loads(exp.RESULT.read_text())
    old = json.loads(exp.block.PHYSICAL.read_text())
    for actual, expected in ((plan["pool_sha256"], file_hash(exp.POOL)),
                             (plan["runner_sha256"], file_hash(exp.__file__)),
                             (plan["protocol_sha256"], file_hash(exp.PROTOCOL)),
                             (physical["plan_sha256"], file_hash(exp.PLAN)),
                             (result["physical_sha256"], file_hash(exp.PHYSICAL))):
        if actual != expected:
            raise ValueError("artifact binding mismatch")
    model, scaler, transform, scaler_hash = exp.block.prior.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != plan["model_sha256"] or scaler_hash != plan["normalizer_sha256"]:
        raise ValueError("model or scaler drift")
    errors = []
    with (np.load(exp.POOL, allow_pickle=False) as pool,
          np.load(exp.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
          torch.inference_mode()):
        for root in range(50):
            p, q = plan["rows"][root], physical["rows"][root]
            if p["root"] != root or q["root"] != root or p["row"] != q["row"]:
                raise ValueError("row mismatch")
            for arm in exp.ARMS:
                actions = pool[f"r{root}_{arm}_actions"]
                scores = pool[f"r{root}_{arm}_scores"]
                trace = q["outcomes"][arm]["trace"]
                if (actions.shape != (300, 25, 2) or scores.shape != (300,)
                        or not np.isfinite(actions).all() or not np.isfinite(scores).all()
                        or np.max(np.abs(actions)) > 1 or len(trace) != 25
                        or int(np.argmin(scores)) != p["outcomes"][arm]["choice"]
                        or q["outcomes"][arm]["ever"] != any(t["terminated"] for t in trace)
                        or q["outcomes"][arm]["final"] != trace[-1]["terminated"]):
                    raise ValueError("candidate, selection or outcome mismatch")
                if arm.startswith("block_"):
                    for k in range(5):
                        np.testing.assert_array_equal(actions[:, k*5:k*5+5],
                                                      np.repeat(actions[:, k*5:k*5+1], 5, axis=1))
                if root < 10:
                    prior = old["rows"][root]["selected" if arm == "block_random" else "baseline"]
                    if arm != "block_cem" and trace != prior["trace"]:
                        raise ValueError("old selected physical trace mismatch")
                if root in REPLAY_ROOTS:
                    rng = np.random.default_rng(p["outcomes"][arm]["seed"])
                    shape = (25, 2) if arm == "full_cem" else (5, 2)
                    mean = np.broadcast_to(scaler.transform(np.zeros((1, 2)))[0], shape).copy()
                    std = np.ones(shape)
                    low = (-1 - scaler.mean_) / scaler.scale_
                    high = (1 - scaler.mean_) / scaler.scale_
                    normalized = []
                    for iteration in range(3):
                        if arm == "block_random":
                            original = exp.block.block_actions(p["outcomes"][arm]["seed"])
                            np.testing.assert_array_equal(original[iteration*100:(iteration+1)*100],
                                                          actions[iteration*100:(iteration+1)*100])
                            expanded = scaler.transform(actions[iteration*100:(iteration+1)*100]
                                                        .reshape(-1, 2)).reshape(100, 25, 2)
                        else:
                            samples = exp.normalized_cem_candidates(rng, mean, std, low, high,
                                                                    blockwise=arm == "block_cem")
                            expanded = np.repeat(samples, 5, axis=1) if arm == "block_cem" else samples
                            elite = samples[np.argsort(scores[iteration*100:(iteration+1)*100],
                                                       kind="stable")[:10]]
                            mean = elite.mean(axis=0)
                            std = np.maximum(elite.std(axis=0), 1e-5)
                        if arm != "block_random":
                            np.testing.assert_array_equal(
                                exp.block.prior.to_physical(expanded, scaler).astype(np.float32),
                                actions[iteration*100:(iteration+1)*100])
                        normalized.append(expanded)
                    norm = np.concatenate(normalized)
                    context = {"pixels": transform(inputs["current"][root])[None, None, None].cuda(),
                               "goal": transform(inputs["goal"][root])[None, None, None].cuda(),
                               "action": torch.empty((1, 1, 0, 10), device="cuda")}
                    replay = []
                    for offset in range(0, 300, 32):
                        chunk = torch.tensor(norm[offset:offset+32].reshape(1, -1, 5, 10),
                                             device="cuda", dtype=torch.float32)
                        replay.extend(model.get_cost(dict(context), chunk)[0].cpu().tolist())
                    error = float(np.max(np.abs(np.asarray(replay) - scores)))
                    if error > 1e-3 or int(np.argmin(replay)) != p["outcomes"][arm]["choice"]:
                        raise ValueError(f"GPU replay mismatch root={root} arm={arm} "
                                         f"max_error={error} replay_choice={int(np.argmin(replay))} "
                                         f"sealed_choice={p['outcomes'][arm]['choice']}")
                    errors.append({"root": root, "arm": arm, "max_score_error": error})
            print(f"audited {root + 1}/50", flush=True)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model mutated")
    with OUTPUT.open("x") as f:
        json.dump({"score_replays": errors, "roots_checked": 50,
                   "prior_trace_roots": 10, "plan_sha256": file_hash(exp.PLAN),
                   "physical_sha256": file_hash(exp.PHYSICAL), "result_sha256": file_hash(exp.RESULT),
                   "scope": "same implementation replay, not independent certification"}, f, indent=2)


if __name__ == "__main__":
    main()
