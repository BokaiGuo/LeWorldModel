"""Render-only pusher intervention on reconstructed witnessed goal frames."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as base
from . import p1_backtrack_physical_runner_20261030 as physical
from . import p1_constructive_moving_history_source_20261027_v1 as moving

NAME = "codex_p1_pusher_render_intervention_20261030_main"
OFFSETS = ((0, 0), (-10, 0), (10, 0), (0, -10), (0, 10),
           (-20, 0), (20, 0), (0, -20), (0, 20),
           (-40, 0), (40, 0), (0, -40), (0, 40))


def images():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    prepared, data = physical.material()
    plan = data["source_plan"]
    output = []
    for root in data["roots"]:
        slot = root["slot"]
        if slot != plan["slots"][slot["index"]]:
            raise ValueError("root slot mismatch")
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)
        try:
            _, info = env.reset(seed=slot["reset_seed"])
            goal = moving.source.vector(info["goal_state"])
            requested = moving.source.sample_state(goal, list(env.unwrapped.block.shapes),
                                                   dict(root_seed=slot["root_seed"]), 0, 0, plan)
            env.reset(seed=0, options={"state": requested.copy(), "goal_state": goal.copy()})
            for action in moving.warmup_commands():
                env.step(action.copy())
            body = env.unwrapped
            _, actions = moving.witness(body._get_obs(), slot["sign"], plan["packet_logit_cap"])
            for action in actions:
                env.step(action.copy())
            if not np.array_equal(body._get_obs(), root["goal"]):
                raise ValueError("witness goal state mismatch")
            original_agent = tuple(body.agent.position)
            original_block = (tuple(body.block.position), float(body.block.angle))
            for condition in ("off", "on"):
                body.with_target = condition == "on"
                expected = root["packets"][condition]["goal_rgb"]
                if not np.array_equal(env.render(), expected):
                    raise ValueError("witness goal RGB mismatch")
                for offset in OFFSETS:
                    body.agent.position = (original_agent[0] + offset[0], original_agent[1] + offset[1])
                    body.space.reindex_shapes_for_body(body.agent)
                    image = np.asarray(env.render()).copy()
                    if (tuple(body.block.position), float(body.block.angle)) != original_block:
                        raise ValueError("object moved during rendering")
                    if offset == (0, 0) and not np.array_equal(image, expected):
                        raise ValueError("baseline image mismatch")
                    output.append({"root_slot": slot["index"], "condition": condition, "offset": list(offset),
                                   "image": image, "goal": np.asarray(expected).copy(),
                                   "object_pose": [*original_block[0], original_block[1]],
                                   "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest()})
                body.agent.position = original_agent
                body.space.reindex_shapes_for_body(body.agent)
        finally:
            env.close()
    return prepared, output


def run(out):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    if (not out.is_absolute() or out.parent != base.ROOT / "results" or
            not out.name.startswith(NAME + "_") or out.exists() or out.is_symlink()):
        raise ValueError("exclusive output")
    prepared, panel = images()
    if len(panel) != 52:
        raise ValueError("fixed denominator")
    bindings = base.bindings()
    bindings.update(prepared["bindings"])
    bindings[str(Path(__file__).resolve())] = base.sha(__file__)
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("model mismatch")
        transform = p0.image_transform()
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            reference = None
            for item in panel:
                latent = model.encode({"pixels": p0._image_tensor(item["image"], transform=transform, device=device)})["emb"]
                if latent.shape != (1, 1, 192) or not bool(torch.isfinite(latent).all()):
                    raise ValueError("latent ABI")
                latent = latent[0, 0]
                if item["offset"] == [0, 0]:
                    reference = latent.clone()
                delta = latent - reference
                rows.append({k: v for k, v in item.items() if k not in ("image", "goal")} |
                            {"latent_mse_to_goal": float(delta.square().mean()),
                             "changed_pixels": int(np.any(item["image"] != item["goal"], axis=2).sum())})
        torch.cuda.synchronize(device)
        if before != base.model_hash(model) or any(base.sha(Path(p)) != h for p, h in bindings.items()):
            raise ValueError("model or source drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "sources": bindings, "host": host,
                  "model_before": before, "model_after": before, "encode_calls": 52,
                  "diagnostic_only": True, "P1_admission": False, "independent_confirmation": False,
                  "limitations": "Two known goals, render-only positional intervention; potential overlap or "
                                 "clipping, no physical action reachability, no full dependency closure."}
        digest = base.write(out, result)
    print(json.dumps({"status": "complete", "rows": len(rows), "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print("Reconstruct fixed witnessed goals and test pusher-only renderer changes on host CUDA.", flush=True)
    run(Path(parser.parse_args().output))
