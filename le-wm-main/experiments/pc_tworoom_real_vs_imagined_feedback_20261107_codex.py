"""Matched search, real RGB versus model-imagined state, selected dev cases."""
import json
import time
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_fixed_prefix_feedback_20261107_codex as prior
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

PREFIX = prior.source.b.q.ROOT / "results/pc_tworoom_real_vs_imagined_feedback_20261107_codex"
OUTPUT = Path(str(PREFIX) + "_result.json")
PROTOCOL = prior.source.b.q.ROOT.parent / "docs/PC_TWOROOM_REAL_VS_IMAGINED_FEEDBACK_PROTOCOL_20261107_CODEX.md"


def latent_rollout(model, initial, actions):
    """Use native H=1 rollout with a scoped, already-encoded initial state."""
    if initial.ndim != 1 or actions.ndim != 3 or actions.shape[-1] != 10 or not 1 <= actions.shape[1] <= 5:
        raise ValueError("invalid latent or action packet")
    def encoded(info):
        return {**info, "emb": initial[None, None]}
    dummy = torch.zeros((1, 1, 1, 1), device=initial.device)
    with patch.object(model, "encode", side_effect=encoded):
        return model.rollout({"pixels": dummy}, actions[None])["predicted_emb"][0]


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = prior.source
    source.verify()
    pre = json.loads(source.RESULT.read_text())
    if pre["features_sha256"] != file_hash(source.FEATURES) or pre["frame_sha256"] != file_hash(source.FRAMES):
        raise ValueError("temporal artifact drift")
    plan, model, scaler, transform = source.setup()
    common = source.b.q.old.base.source.block.prior.common
    inputs_path = source.b.q.old.base.source.block.prior.inputs.INPUT
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, prior.planner.__file__, source.b.q.PLAN,
                         source.b.q.POOL, inputs_path, common.CHECKPOINT / "weights.pt"],
                         preprocessing="ImageNet 224 and TwoRoom scaler",
                         clock="five environment steps per packet, remaining deadline")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with np.load(source.b.q.POOL, allow_pickle=False) as pools, \
                np.load(source.b.q.old.POOL, allow_pickle=False) as bank, \
                np.load(source.PRED, allow_pickle=False) as predictions, \
                np.load(inputs_path, allow_pickle=False) as inputs, \
                h5py.File(common.DATA, "r") as data, torch.inference_mode():
            for root in prior.ROOTS:
                task = next(row for row in plan["rows"] if row["root"] == root)
                chain = pools[f"r{root}_actions"][0]
                if prior.digest(chain) != task["decisions"]["lookahead"]["action_sha256"]:
                    raise ValueError("selected chain drift")
                goal_image = transform(inputs["goal"][root])[None, None].cuda()
                initial_pixels = transform(inputs["current"][root])[None, None, None].cuda()
                raw = bank[f"r{root}_b"].reshape(64, 25, 2)[:32]
                normal = scaler.transform(raw.reshape(-1, 2)).reshape(32, 5, 10)
                test = torch.as_tensor(normal, device="cuda", dtype=torch.float32)
                ctx = {"pixels": initial_pixels, "goal": goal_image[:, None],
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
                official_cost, out = RGBLeWM(model, audit).score(ctx, test[None])
                initial = out["predicted_emb"][0, 0, 0]
                goal = out["goal_emb"][0, 0]
                local = latent_rollout(model, initial, test)
                feat_error = float((local - out["predicted_emb"][0]).abs().max())
                costs = (local[:, -1] - goal).square().sum(-1)
                cost_error = float((costs - official_cost[0]).abs().max())
                if max(feat_error, cost_error) > 0.001:
                    raise ValueError(f"latent rollout mismatch {feat_error=} {cost_error=}")
                first_action = scaler.transform(chain[:5]).reshape(1, 1, 10)
                imagined_initial = latent_rollout(model, initial, torch.as_tensor(first_action, dtype=torch.float32,
                                                                                   device="cuda"))[0, 1]
                first_error = float((imagined_initial - torch.as_tensor(predictions[f"r{root}_pred"][64, 1],
                                                                         device="cuda")).abs().max())
                if first_error > 1e-5:
                    raise ValueError("sealed initial imagined state drift")
                arms = {}
                for arm in ("real_rgb", "imagined"):
                    started = time.perf_counter()
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][task["row"]])
                    actor._set_goal_state(data["proprio"][task["row"] + 25])
                    image_hash = prior.digest(actor.render().copy())
                    trace, decisions = [], []
                    for action in chain[:5]:
                        _, _, term, trunc, info = actor.step(action.copy())
                        trace.append({"action": action.tolist(), "distance": float(info["distance_to_target"]),
                                      "terminated": bool(term), "truncated": bool(trunc)})
                        if term or trunc:
                            break
                    prior.check_same_prefix(chain, trace)
                    if term or trunc:
                        raise ValueError("unexpected initial termination")
                    latent, old_tail = imagined_initial.clone(), chain[5:].copy()
                    for iteration in range(1, 5):
                        remaining = 25 - len(trace)
                        if arm == "real_rgb":
                            image = actor.render().copy()
                            latent = model.encode({"pixels": transform(image)[None, None].cuda()})["emb"][0, 0]
                        calls = 0

                        def score(actions):
                            nonlocal calls
                            normal = scaler.transform(actions.reshape(-1, 2)).reshape(-1, remaining // 5, 10)
                            outputs = []
                            for start in range(0, len(normal), 32):
                                chunk = torch.as_tensor(normal[start:start + 32], dtype=torch.float32, device="cuda")
                                pred = latent_rollout(model, latent, chunk)
                                outputs.extend((pred[:, -1] - goal).square().sum(-1).cpu().tolist())
                                calls += 1
                            return np.asarray(outputs)

                        torch.cuda.synchronize()
                        begin = time.perf_counter()
                        result = prior.planner.search(score, remaining=remaining, old_tail=old_tail,
                                                      seed=27121000 + root * 10 + iteration, arm="baseline")
                        chosen = result["actions"]
                        packet = scaler.transform(chosen[:5]).reshape(1, 1, 10)
                        next_latent = latent_rollout(model, latent, torch.as_tensor(packet, dtype=torch.float32,
                                                                                   device="cuda"))[0, 1]
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - begin
                        decisions.append({"iteration": iteration, "remaining": remaining, "winner": result["winner"],
                                          "score": result["cost"], "actions": chosen.tolist(),
                                          "actions_sha256": result["actions_sha256"], "old_tail_sha256": prior.digest(old_tail),
                                          "all_scores": result["all_scores"].tolist(), "queries": result["model_queries"],
                                          "packet_steps": result["model_packet_steps"], "forward_batches": calls,
                                          "propagation_rollouts": 1, "planning_seconds": elapsed})
                        dispatched = 0
                        for action in chosen[:5]:
                            _, _, term, trunc, info = actor.step(action.copy())
                            trace.append({"action": action.tolist(), "distance": float(info["distance_to_target"]),
                                          "terminated": bool(term), "truncated": bool(trunc)})
                            dispatched += 1
                            if term or trunc:
                                break
                        latent = next_latent
                        old_tail = chosen[dispatched:].copy()
                        if term or trunc:
                            break
                    arms[arm] = {"trace": trace, "decisions": decisions, "initial_rgb_sha256": image_hash,
                                 "ever_success": any(x["terminated"] for x in trace),
                                 "queries": sum(x["queries"] for x in decisions),
                                 "packet_steps": sum(x["packet_steps"] for x in decisions),
                                 "planning_seconds": sum(x["planning_seconds"] for x in decisions),
                                 "wall_seconds": time.perf_counter() - started}
                if arms["real_rgb"]["initial_rgb_sha256"] != arms["imagined"]["initial_rgb_sha256"]:
                    raise ValueError("reset image mismatch")
                rows.append({"root": root, "row": task["row"], "latent_feature_parity_max": feat_error,
                             "latent_cost_parity_max": cost_error, "first_successor_parity_max": first_error,
                             "arms": arms})
                print(f"root={root} actual={arms['real_rgb']['ever_success']} imagined={arms['imagined']['ever_success']}",
                      flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("weights changed")
    paths = [Path(__file__), PROTOCOL, Path(prior.__file__), Path(prior.planner.__file__),
             source.RESULT, source.PRED, source.META, source.b.q.PLAN, source.b.q.POOL,
             source.b.q.old.POOL, inputs_path, common.DATA, common.CHECKPOINT / "weights.pt"]
    with OUTPUT.open("x") as f:
        json.dump({"track": RGB, "rows": rows, "model_sha256": plan["model_sha256"],
                   "source_sha256": {str(path): file_hash(path) for path in paths},
                   "scope": "two retrospectively selected failures, one-latent replanning"},
                  f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
