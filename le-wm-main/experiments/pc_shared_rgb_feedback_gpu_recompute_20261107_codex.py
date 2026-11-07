"""Recompute all saved feedback candidate costs from replayed RGB on GPU."""
import json
import time

import h5py
import numpy as np
import torch

from experiments import pc_shared_rgb_feedback_ledger_20261107_codex as saved
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit, tensor_receipt
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = saved.prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_SHARED_RGB_FEEDBACK_GPU_RECOMPUTE_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_shared_rgb_feedback_gpu_recompute_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    ledger = json.loads(saved.OUT.read_text())
    for key, path in (("protocol_sha256", saved.PROTOCOL), ("runner_sha256", saved.__file__),
                      ("prior_sha256", saved.prior.OUT), ("shared_sha256", saved.prior.shared.__file__),
                      ("adapter_sha256", saved.prior.adapter.__file__)):
        if ledger[key] != file_hash(path):
            raise ValueError(f"ledger source mismatch: {key}")
    model, scaler, transform, normalizer = saved.prior.source.block.prior.common.model_setup()
    plan = json.loads(saved.prior.source.PLAN.read_text())
    fingerprint = model_fingerprint(model)
    if fingerprint != ledger["model_sha256"] or normalizer != plan["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    deps = [__file__, PROTOCOL, saved.OUT, saved.prior.shared.__file__, saved.prior.adapter.__file__,
            saved.prior.source.block.prior.inputs.INPUT,
            saved.prior.source.block.prior.common.CHECKPOINT / "weights.pt"]
    audit = freeze_audit(RGB, "lewm", deps, preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="one observed frame; five packets of five environment steps")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(saved.prior.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
              h5py.File(saved.prior.source.block.prior.common.DATA, "r") as data, torch.inference_mode()):
            for row in ledger["rows"]:
                root, task = row["root"], row["row"]
                if task != int(inputs["rows"][root]):
                    raise ValueError("task mismatch")
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(data["proprio"][task])
                actor._set_goal_state(data["proprio"][task + 25])
                goal = transform(inputs["goal"][root])[None, None, None].cuda()
                tail, step = None, 0
                for iteration, item in enumerate(row["rounds"]):
                    pixels = transform(actor.render().copy())[None, None, None].cuda()
                    if item["receipts"][0]["inputs"]["pixels"] != tensor_receipt(pixels):
                        raise ValueError("replayed observation drift")
                    ctx = {"pixels": pixels, "goal": goal, "action": torch.empty((1, 1, 0, 10), device="cuda")}
                    parents, probes, tails = saved.prior.build_candidates(item["candidate_seed"], tail)
                    boundary = RGBLeWM(model, audit)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    current = saved.prior.adapter.compile_rgb(
                        boundary, ctx, parents, probes, tails, scaler=scaler, prefix_steps=10,
                        retain_count=1, strategy="lookahead", budget=11, seed=item["strategy_seed"])
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    max_error = 0.
                    for key in ("parent_cost", "probe_cost", "tail_cost"):
                        original = np.asarray(item[key])
                        np.testing.assert_allclose(current[key], original, atol=.001, rtol=1e-6)
                        max_error = max(max_error, float(np.max(np.abs(current[key] - original))))
                    if (current["retained_prefixes"] != item["retained"]
                            or current["choice"].source != item["choice"]["source"]
                            or current["choice"].index != item["choice"]["index"]):
                        raise ValueError("recomputed selection changed")
                    np.testing.assert_array_equal(current["actions"], np.asarray(item["actions"], dtype=np.float32))
                    dispatched = np.asarray(item["dispatched"], dtype=np.float32)
                    for action in dispatched:
                        _, _, term, trunc, info = actor.step(action.copy())
                        actual = row["trace"][step]
                        if (bool(term), bool(trunc)) != (actual["terminated"], actual["truncated"]):
                            raise ValueError("termination mismatch")
                        if abs(float(info["distance_to_target"]) - actual["distance"]) > 1e-6:
                            raise ValueError("physical replay mismatch")
                        step += 1
                    tail = current["actions"][len(dispatched):].copy()
                    rows.append({"root": root, "iteration": iteration, "max_score_error": max_error,
                                 "queries": current["model_queries"], "seconds": elapsed})
                if step != len(row["trace"]):
                    raise ValueError("incomplete physical trace")
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("weights changed")
    result = {"track": RGB, "ledger_sha256": file_hash(saved.OUT), "protocol_sha256": file_hash(PROTOCOL),
              "runner_sha256": file_hash(__file__), "model_sha256": fingerprint, "rows": rows,
              "scope": "same model and environment, exposed tasks; independent recomputation only"}
    with OUT.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({"rounds": len(rows), "queries": sum(r["queries"] for r in rows),
                      "max_score_error": max(r["max_score_error"] for r in rows),
                      "seconds": sum(r["seconds"] for r in rows)}, indent=2))


if __name__ == "__main__":
    main()
