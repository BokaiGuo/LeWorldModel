"""Causal RGB feedback history and bounded-CEM development runner."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_history_contract_20261107_codex import score_future
from experiments import pc_rgb_true_history_alignment_20261107_codex as source


class ObservedHistory:
    """Append an observed successor only after the matching dispatched packet."""
    def __init__(self, initial_frame, block=5, capacity=3):
        self.block = block
        self.capacity = capacity
        self.frames = [np.asarray(initial_frame).copy()]
        self.packets = []

    def append(self, dispatched, successor):
        packet = np.asarray(dispatched, dtype=np.float32)
        if packet.shape != (self.block, 2) or not np.isfinite(packet).all():
            raise ValueError("history needs one finite, complete dispatched packet")
        if np.any(np.abs(packet) > 1):
            raise ValueError("physical action outside [-1,1]")
        frame = np.asarray(successor)
        if frame.shape != self.frames[-1].shape or frame.dtype != self.frames[-1].dtype:
            raise ValueError("successor frame mismatch")
        self.frames.append(frame.copy())
        self.packets.append(packet.copy())
        if len(self.frames) > self.capacity:
            self.frames.pop(0)
            self.packets.pop(0)

    def view(self, mode):
        if mode == "single":
            return self.frames[-1:], []
        if mode != "history":
            raise ValueError("unknown history mode")
        if len(self.packets) != len(self.frames) - 1:
            raise ValueError("observation/action clock mismatch")
        return self.frames, self.packets


class RGBFeedbackPlanner:
    """Accepts RGB and action history only; no simulator handle or truth."""
    def __init__(self, wrapper, scaler, transform, seed, samples=32, steps=2, elite=8):
        self.wrapper, self.scaler, self.transform = wrapper, scaler, transform
        self.samples, self.steps, self.elite = samples, steps, elite
        self.generator = torch.Generator(device="cuda").manual_seed(seed)
        self.low = torch.tensor(np.tile((-1 + 1e-6 - scaler.mean_) / scaler.scale_, 5),
                                device="cuda", dtype=torch.float32)
        self.high = torch.tensor(np.tile((1 - 1e-6 - scaler.mean_) / scaler.scale_, 5),
                                 device="cuda", dtype=torch.float32)
        self.zero = torch.tensor(np.tile(-scaler.mean_ / scaler.scale_, 5),
                                 device="cuda", dtype=torch.float32)

    @torch.inference_mode()
    def plan(self, frames, goal, packets, warm):
        if len(packets) != len(frames) - 1 or not 1 <= len(frames) <= 3:
            raise ValueError("history is not synchronized")
        pixels = torch.stack([self.transform(frame) for frame in frames]).cuda()[None, None]
        target = self.transform(goal).cuda()[None, None, None]
        if packets:
            past = self.scaler.transform(np.asarray(packets).reshape(-1, 2))
            history = torch.tensor(past.reshape(1, 1, len(packets), 10), device="cuda", dtype=torch.float32)
        else:
            history = torch.empty((1, 1, 0, 10), device="cuda")
        center = self.zero.repeat(3, 1)
        if warm is not None:
            if warm.shape != (2, 10):
                raise ValueError("invalid warm tail")
            center[:2] = torch.as_tensor(warm, device="cuda")
        center = center.clamp(self.low, self.high)
        warm_completed = center.clone()
        std = torch.ones_like(center)
        calls = 0

        def cost(actions):
            nonlocal calls
            calls += actions.shape[1]
            result, _ = score_future(self.wrapper, pixels=pixels, goal=target, history=history, future=actions)
            if result.shape != actions.shape[:2] or not torch.isfinite(result).all():
                raise ValueError("invalid costs")
            return result[0]

        start = time.perf_counter()
        for _ in range(self.steps):
            noise = torch.randn((self.samples, 3, 10), generator=self.generator, device="cuda")
            candidates = (center[None] + noise * std[None]).clamp(self.low, self.high)
            candidates[0] = center
            values = cost(candidates[None])
            selected = candidates[torch.topk(values, self.elite, largest=False).indices]
            center, std = selected.mean(0), selected.std(0)
        options = torch.stack((center, warm_completed))
        option_costs = cost(options[None])
        chosen = int(option_costs.argmin())
        torch.cuda.synchronize()
        chosen_action = options[chosen].cpu().numpy()
        raw = self.scaler.inverse_transform(chosen_action.reshape(-1, 2)).reshape(3, 5, 2)
        if np.any(np.abs(raw) > 1) or not np.isfinite(raw).all():
            raise ValueError("invalid physical action")
        return raw, chosen_action[1:], {
            "history_frames": len(frames), "history_packets": len(packets),
            "scored_candidates": calls, "option_costs": option_costs.cpu().tolist(),
            "selected_option": chosen, "planning_seconds": time.perf_counter() - start,
            "normalized_selected_chain": chosen_action.tolist(),
        }


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    from experiments.pc_geometric_contact_equivalence_dev_20261106_codex import PHYSICAL, IMAGES
    from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot

    out = source.ROOT / "results/pc_rgb_history_feedback_20261107_codex.json"
    protocol = source.ROOT.parent / "docs/PC_RGB_HISTORY_FEEDBACK_PROTOCOL_20261107_CODEX.md"
    if out.exists():
        raise FileExistsError(out)
    physical = json.loads(PHYSICAL.read_text())
    if physical["images_sha256"] != file_hash(IMAGES):
        raise ValueError("source image mismatch")
    model, scaler, transform, action_hash = source.model_and_scaler()
    fingerprint = model_fingerprint(model)
    audit = freeze_audit(RGB, "lewm", [Path(__file__), protocol,
                         Path(__file__).with_name("pc_rgb_history_contract_20261107_codex.py")],
                         preprocessing="PushT ImageNet transform and expert action scaler",
                         clock="5 executed actions per RGB observation")
    env = gym.make("swm/PushT-v1", resolution=224, relative=True, render_action=False, with_target=True)
    actor = env.unwrapped
    rows = []
    print("RGB-only development: paired single-frame versus actual multi-frame history.", flush=True)
    try:
        with np.load(IMAGES, allow_pickle=False) as images:
            for root in (0, 8):
                row = physical["rows"][root]
                condition = next(c for c in row["conditions"] if c["side"] == 1 and c["appearance"] == "white")
                for mode in ("single", "history"):
                    env.reset(seed=881001 + root, options={"state": np.asarray(condition["requested_state"]),
                                                           "goal_state": np.asarray(row["goal"])})
                    actor.variation_space["background"]["color"].set_value(np.array([255, 255, 255], np.uint8))
                    np.testing.assert_allclose(snapshot(actor), condition["actions"]["plus"]["initial_snapshot"],
                                               atol=1e-9, rtol=0)
                    first = actor.render().copy()
                    np.testing.assert_array_equal(first, images[f"r{root}_white_1_plus_initial"])
                    history = ObservedHistory(first)
                    goal = images[f"r{root}_white_goal"].copy()
                    wrapper = RGBLeWM(model, audit)
                    planner = RGBFeedbackPlanner(wrapper, scaler, transform, seed=4200 + root)
                    trace, warm = [], None
                    for iteration in range(3):
                        frames, packets = history.view(mode)
                        plan, warm, meta = planner.plan(frames, goal, packets, warm)
                        steps = []
                        for action in plan[0]:
                            obs, reward, terminated, truncated, info = actor.step(action.copy())
                            steps.append({"action": action.tolist(), "reward": float(reward),
                                          "terminated": bool(terminated), "truncated": bool(truncated),
                                          "snapshot": snapshot(actor).tolist()})
                            if terminated or truncated:
                                break
                        if len(steps) == 5:
                            history.append(np.asarray([x["action"] for x in steps], dtype=np.float32),
                                           actor.render().copy())
                        meta.update(iteration=iteration, executed_steps=steps)
                        trace.append(meta)
                        if terminated or truncated:
                            break
                    rows.append({"root": root, "mode": mode, "plans": trace, "receipts": wrapper.receipts})
                    print(f"root={root} mode={mode} plans={len(trace)}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model weights changed")
    output = {"track": RGB, "rows": rows, "model_sha256": fingerprint,
              "normalizer_source_sha256": action_hash, "source_physical_sha256": file_hash(PHYSICAL),
              "source_images_sha256": file_hash(IMAGES), "protocol_sha256": file_hash(protocol),
              "runner_sha256": file_hash(__file__),
              "scope": "exposed two-root development, not PC4 or efficacy confirmation"}
    with out.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
