"""One exposed PushT episode: verify timed feedback wiring against a sealed run."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments import pc_rgb_true_history_alignment_20261107_codex as source
from experiments.pc_geometric_contact_equivalence_dev_20261106_codex import IMAGES, PHYSICAL
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit, tensor_receipt
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot
from experiments.pc_rgb_history_feedback_20261107_codex import ObservedHistory, RGBFeedbackPlanner
from experiments.pc_rgb_single_frame_contract_20261107_codex import model_digest
from experiments.pc_rgb_timed_feedback_bridge_20261107_codex import TimedFeedbackBridge
from experiments.pc_rgb_timed_history_contract_20261107_codex import TimedHistoryRGB


OUT = source.ROOT / "results/pc_rgb_timed_feedback_replay_20261107_codex.json"
PRIOR = source.ROOT / "results/pc_rgb_history_feedback_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    physical = json.loads(PHYSICAL.read_text())
    if physical["images_sha256"] != file_hash(IMAGES):
        raise ValueError("source image mismatch")
    model, scaler, transform, normalizer_hash = source.model_and_scaler()
    original_fingerprint = model_fingerprint(model)
    digest = model_digest(model)
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PHYSICAL, IMAGES,
                         Path(__file__).with_name("pc_rgb_timed_feedback_bridge_20261107_codex.py"),
                         Path(__file__).with_name("pc_rgb_timed_history_contract_20261107_codex.py")],
                         preprocessing="PushT ImageNet transform and expert action scaler",
                         clock="actual dispatch count; five actions per observed packet")
    timed = TimedHistoryRGB(model, audit, expected_model_digest=digest,
                            action_mean=scaler.mean_, action_scale=scaler.scale_,
                            packet_steps=5, physical_low=-1., physical_high=1., max_frames=3)
    bridge = TimedFeedbackBridge(timed)
    env = gym.make("swm/PushT-v1", resolution=224, relative=True, render_action=False, with_target=True)
    actor = env.unwrapped
    plans, warm, clock = [], None, 0
    try:
        row = physical["rows"][0]
        condition = next(c for c in row["conditions"] if c["side"] == 1 and c["appearance"] == "white")
        with np.load(IMAGES, allow_pickle=False) as images:
            env.reset(seed=881001, options={"state": np.asarray(condition["requested_state"]),
                                          "goal_state": np.asarray(row["goal"])})
            actor.variation_space["background"]["color"].set_value(np.array([255, 255, 255], np.uint8))
            np.testing.assert_allclose(snapshot(actor), condition["actions"]["plus"]["initial_snapshot"],
                                       atol=1e-9, rtol=0)
            first = actor.render().copy()
            np.testing.assert_array_equal(first, images["r0_white_1_plus_initial"])
            history = ObservedHistory(first)
            frame_steps = [0]
            goal = images["r0_white_goal"].copy()
            planner = RGBFeedbackPlanner(bridge, scaler, transform, seed=4200)
            for iteration in range(3):
                frames, packets = history.view("history")
                normalized = (scaler.transform(np.asarray(packets).reshape(-1, 2))
                              .reshape(1, 1, len(packets), 10) if packets
                              else np.empty((1, 1, 0, 10), np.float32))
                bound = torch.tensor(normalized, device="cuda", dtype=torch.float32)
                bridge.bind(frame_steps=frame_steps, current_step=clock, history=bound)
                plan, warm, meta = planner.plan(frames, goal, packets, warm)
                meta["frame_steps"] = list(frame_steps)
                meta["history_receipt"] = tensor_receipt(bound)
                meta["frame_sha256"] = [hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest()
                                        for f in frames]
                steps = []
                for action in plan[0]:
                    _, reward, terminated, truncated, _ = actor.step(action.copy())
                    clock += 1
                    steps.append({"action": action.tolist(), "reward": float(reward),
                                  "terminated": bool(terminated), "truncated": bool(truncated),
                                  "snapshot": snapshot(actor).tolist()})
                    if terminated or truncated:
                        break
                if len(steps) == 5:
                    history.append(np.asarray([step["action"] for step in steps], dtype=np.float32),
                                   actor.render().copy())
                    frame_steps = (frame_steps + [clock])[-history.capacity:]
                meta.update(iteration=iteration, executed_steps=steps)
                plans.append(meta)
                print(f"iteration={iteration} clock={clock}", flush=True)
                if terminated or truncated:
                    break
    finally:
        env.close()
    timed.verify_unchanged()
    if model_fingerprint(model) != original_fingerprint:
        raise ValueError("model changed")
    # Old outcomes are read only after the new episode has completed.
    prior = json.loads(PRIOR.read_text())
    if prior["model_sha256"] != original_fingerprint or prior["normalizer_source_sha256"] != normalizer_hash:
        raise ValueError("sealed model or normalizer differs")
    old = next(r for r in prior["rows"] if r["root"] == 0 and r["mode"] == "history")
    comparisons = []
    for new, prev in zip(plans, old["plans"]):
        a = np.asarray(new["normalized_selected_chain"])
        b = np.asarray(prev["normalized_selected_chain"])
        x = np.asarray([s["action"] for s in new["executed_steps"]])
        y = np.asarray([s["action"] for s in prev["executed_steps"]])
        comparisons.append({"iteration": new["iteration"], "normalized_action_max_abs": float(np.max(np.abs(a-b))),
                            "dispatched_action_max_abs": float(np.max(np.abs(x-y))) if x.shape == y.shape else None,
                            "option_cost_max_abs": float(np.max(np.abs(np.asarray(new["option_costs"]) -
                                                                     np.asarray(prev["option_costs"])))),
                            "receipt_history_match": all(r["inputs"]["action"] == new["history_receipt"]
                                for r in old["receipts"][3*new["iteration"]:3*new["iteration"]+3])})
    output = {"track": RGB, "root": 0, "plans": plans, "comparisons": comparisons,
              "old_plan_count": len(old["plans"]), "model_sha256": original_fingerprint,
              "normalizer_source_sha256": normalizer_hash, "prior_sha256": file_hash(PRIOR),
              "runner_sha256": file_hash(__file__),
              "scope": "exposed one-root wiring replay; no independent performance or gate claim"}
    with OUT.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == "__main__":
    main()
