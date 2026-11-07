"""GPU audit: shared prefix can mask suffix differences in whole-time minima."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_retention_counterfactual_pool_20261107_codex as pool
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

run = pool.run
PROTOCOL = run.ROOT.parent / "docs/PC_TEMPORAL_PREFIX_MASK_AUDIT_PROTOCOL_20261107_CODEX.md"
OUT = run.ROOT / "results/pc_temporal_prefix_mask_audit_20261107_codex.json"


def analyze(time_cost, prefix_packets, *, tolerance=1e-3):
    """Input is prefix x tail x predicted packet, excluding initial state."""
    values = np.asarray(time_cost, dtype=np.float64)
    if (values.ndim != 3 or values.shape[:2] != (3, 3)
            or not 0 < prefix_packets < values.shape[2] or not np.isfinite(values).all()):
        raise ValueError("invalid time cost or prefix")
    full = values.min(axis=2)
    post = values[:, :, prefix_packets:].min(axis=2)
    prefix_min = values[:, :, :prefix_packets].min(axis=2)
    # A tie is counted as reaching the minimum by the prefix boundary.
    before_or_tied = prefix_min <= full + tolerance
    full_range = np.ptp(full, axis=1)
    post_range = np.ptp(post, axis=1)
    full_kept = np.argsort(full.min(axis=1), kind="stable")[:2].tolist()
    post_kept = np.argsort(post.min(axis=1), kind="stable")[:2].tolist()
    return {"all_min_by_prefix": before_or_tied.tolist(),
            "full_range": full_range.tolist(), "post_range": post_range.tolist(),
            "collapsed": (full_range <= tolerance).tolist(),
            "post_distinguishable": (post_range > tolerance).tolist(),
            "full_kept": full_kept, "post_kept": post_kept,
            "set_changed": set(full_kept) != set(post_kept)}


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    original = json.loads(run.OUTPUT.read_text())
    for key, path in (("runner_sha256", run.__file__), ("protocol_sha256", run.PROTOCOL),
                      ("input_sha256", run.INPUT)):
        if original[key] != file_hash(path):
            raise ValueError(f"source mismatch: {key}")
    model, scaler, transform, normalizer = run.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != original["model_sha256"] or normalizer != original["normalizer_sha256"]:
        raise ValueError("model or normalizer mismatch")
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, run.OUTPUT, run.INPUT,
                        run.common.CHECKPOINT / "weights.pt", run.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="five environment steps per predicted packet")
    boundary = RGBLeWM(model, audit)
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    packet_count = 0
    try:
        with (np.load(run.INPUT, allow_pickle=False) as inputs,
              h5py.File(run.common.DATA, "r") as data, torch.inference_mode()):
            for idx, row in enumerate(original["rows"]):
                if row["row"] != int(inputs["rows"][idx]) or row["episode"] != int(inputs["episodes"][idx]):
                    raise ValueError("episode mismatch")
                goal = transform(inputs["goal"][idx])[None, None, None].cuda()
                for look, effect, history, old_tail in pool.matched_decisions(row):
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row["row"]])
                    actor._set_goal_state(data["proprio"][row["row"] + 25])
                    for k, action in enumerate(history):
                        _, _, term, trunc, info = actor.step(np.asarray(action, dtype=np.float32))
                        expected = row["arms"]["lookahead"]["trace"][k]
                        if (bool(term) != expected["terminated"] or bool(trunc) != expected["truncated"]
                                or abs(float(info["distance_to_target"]) - expected["distance"]) > 1e-5):
                            raise ValueError("history replay mismatch")
                    remaining = look["remaining"]
                    prefix = min(10, remaining - 5)
                    parents, probes, _ = run.program.base.timed.build(look["candidate_seed"], old_tail, prefix)
                    parents, probes = parents[:, :remaining], probes[:, :remaining-prefix]
                    tails = np.concatenate((np.zeros_like(probes[:1]), probes))
                    chains = np.stack([np.concatenate((parents[i, :prefix], tails[j]))
                                       for i in range(3) for j in range(3)])
                    normal = scaler.transform(chains.reshape(-1, 2)).reshape(1, 9, remaining // 5, 10)
                    context = {"pixels": transform(actor.render().copy())[None, None, None].cuda(),
                               "goal": goal, "action": torch.empty((1, 1, 0, 10), device="cuda")}
                    endpoint, context = boundary.score(context, torch.as_tensor(normal, dtype=torch.float32, device="cuda"))
                    pred, target = context["predicted_emb"], context["goal_emb"][:, -1]
                    if pred.shape[2] != remaining // 5 + 1:
                        raise ValueError("prediction clock mismatch")
                    costs = (pred[:, :, 1:] - target[:, None, None]).square().sum(-1)[0].cpu().numpy().reshape(3, 3, -1)
                    for decision in (look, effect):
                        np.testing.assert_allclose(costs.min(axis=2).ravel(), decision["temporal_scores"][1],
                                                   atol=1e-3, rtol=0)
                        np.testing.assert_allclose(endpoint[0].cpu().numpy().reshape(3, 3), decision["probe_cost"],
                                                   atol=1e-3, rtol=0)
                    np.testing.assert_allclose(costs[:, :, -1], endpoint[0].cpu().numpy().reshape(3, 3),
                                               atol=1e-3, rtol=0)
                    diagnostic = analyze(costs, prefix // 5)
                    rows.append({"episode": row["episode"], "iteration": look["iteration"],
                                 "remaining": remaining, "prefix": prefix, "time_cost": costs.tolist(),
                                 **diagnostic})
                    packet_count += 9 * (remaining // 5)
                print(f"episode={row['episode']} decisions={len(rows)}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint or len(rows) != 119:
        raise ValueError("model drift or incomplete panel")
    summary = {"decisions": len(rows), "episodes": len({r["episode"] for r in rows}),
               "prefixes": 3 * len(rows),
               "all_three_min_by_prefix": sum(all(group) for r in rows for group in r["all_min_by_prefix"]),
               "collapsed_full": sum(sum(r["collapsed"]) for r in rows),
               "collapsed_but_post_distinguishable": sum(
                   sum(a and b for a, b in zip(r["collapsed"], r["post_distinguishable"], strict=True)) for r in rows),
               "decisions_changed_set": sum(r["set_changed"] for r in rows),
               "model_chain_queries": len(rows) * 9, "model_packet_steps": packet_count}
    with OUT.open("x") as stream:
        json.dump({"scope": "exposed development; no planner benefit or physical selection",
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "source_sha256": file_hash(run.OUTPUT), "input_sha256": file_hash(run.INPUT),
                   "model_sha256": fingerprint, "normalizer_sha256": normalizer,
                   "summary": summary, "rows": rows}, stream, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
