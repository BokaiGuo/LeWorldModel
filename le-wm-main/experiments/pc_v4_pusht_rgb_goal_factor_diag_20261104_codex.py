"""Post-hoc privileged static RGB factor diagnosis; never used for planning."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments import pc_v4_pusht_rgb_feedback_dev_20261104_codex as run
from experiments import pc_v4_pusht_executed_packet_diag_20261104_codex as prior

ROOT = run.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V4_PUSHT_RGB_GOAL_FACTOR_DIAG_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v4_pusht_rgb_goal_factor_diag_20261104_codex"
OUT = PREFIX.with_name(PREFIX.name + "_result.json")
IMAGES = PREFIX.with_name(PREFIX.name + "_images.npz")
FAILURE = PREFIX.with_name(PREFIX.name + "_failure.json")


def main():
    if any(p.exists() for p in (OUT, IMAGES, FAILURE)):
        raise FileExistsError("existing or partial factor diagnostic output")
    host, sha = run.HOST, run.HOST.bridge.sha_file
    if os.environ.get("CUDA_VISIBLE_DEVICES") != host.UUID:
        raise ValueError("cooperative GPU UUID required")
    source = json.loads(prior.OUT.read_text())
    run.seal.verify(source["dependencies"])
    source_measure = {(row["arm"], row["step"]): row for row in source["measurements"]}
    deps = run.seal.capture([Path(__file__), PROTOCOL, prior.OUT])
    images, rows = {}, []
    print("Rendering fixed goal factors at all eight saved PushT boundaries.", flush=True)
    try:
        with host.lock.gpu_lock(host.LOCK):
            from experiments.gpu_preflight import require_cuda_allocation
            allocation = require_cuda_allocation("cuda:0")
            if host.canonical_gpu_uuid(torch.cuda.get_device_properties(0).uuid) != host.canonical_gpu_uuid(host.UUID):
                raise ValueError("unexpected GPU")
            device = torch.device("cuda:0")
            model = run.saved.load_compatible_pretrained(run.saved.POLICY, device="cuda").eval().requires_grad_(False)
            model.interpolate_pos_encoding = True
            model_hash = run.saved.checked.tensor_state_hash(model)
            if model_hash != source["model_hash"]:
                raise ValueError("model mismatch")
            transform = run.saved.p0.image_transform()
            env = run.phys.official_env_factory()
            try:
                with torch.no_grad():
                    for arm in run.ARMS:
                        execution = json.loads(run.path(arm, "execution.json").read_text())
                        for index in range(4):
                            with np.load(run.path(arm, f"context{index}.npz"), allow_pickle=False) as ctx:
                                original = ctx["frames"][-1].copy()
                                goal_rgb = ctx["goal"].copy()
                            state = np.asarray(execution["outcomes"][index]["state"], np.float64)
                            goal = run.phys.DIAGNOSTIC_GOAL
                            states = {}
                            for name, agent_goal, object_goal in (
                                    ("base", False, False), ("agent", True, False),
                                    ("object", False, True), ("both", True, True)):
                                value = state.copy()
                                if agent_goal:
                                    value[:2] = goal[:2]
                                if object_goal:
                                    value[2:5] = goal[2:5]
                                states[name] = value
                            goal_latent = run.saved.p0._goal_latent(model, goal_rgb, transform=transform, device=device)
                            scores, hashes = {}, {}
                            for name, state_in in states.items():
                                _, info = env.reset(seed=0, options={"state": state_in.copy(), "goal_state": goal.copy()})
                                np.testing.assert_allclose(run.packets.state_of(env), state_in, atol=1e-10, rtol=0)
                                np.testing.assert_array_equal(info["goal"], goal_rgb)
                                rgb = run.phys.render_without_advancing(env)
                                if name == "base":
                                    np.testing.assert_array_equal(rgb, original)
                                key = f"{arm}_{index}_{name}"
                                images[key] = np.array(rgb, copy=True)
                                hashes[name] = run.phys._array_hash(rgb)
                                embedding = run.saved.p0._goal_latent(model, rgb, transform=transform, device=device)
                                scores[name] = float((embedding-goal_latent).square().mean().cpu())
                            reference = (source_measure[(arm, index)]["goal_mse_current"] if index < 3 else
                                         source_measure[(arm, 2)]["goal_mse_actual_next"])
                            if abs(scores["base"] - reference) > 2e-6:
                                raise ValueError("base MSE differs from previous RGB encoding")
                            rows.append({"arm": arm, "boundary": index,
                                "states": {k: v.tolist() for k, v in states.items()}, "rgb_sha256": hashes,
                                "goal_mse": scores, "agent_effect": scores["agent"]-scores["base"],
                                "object_effect": scores["object"]-scores["base"],
                                "interaction": scores["both"]-scores["agent"]-scores["object"]+scores["base"]})
                            print(f"{arm} boundary={index} baseline RGB matched", flush=True)
            finally:
                env.close()
            if run.saved.checked.tensor_state_hash(model) != model_hash:
                raise ValueError("model drift")
        run.seal.verify(source["dependencies"])
        run.seal.verify(deps)
        run.save_npz(IMAGES, **images)
        host.write_new(OUT, {"dependencies": deps, "source_sha256": sha(prior.OUT),
            "images_sha256": sha(IMAGES), "model_hash": model_hash, "cuda_allocation": allocation,
            "rows": rows, "scope": "post-hoc privileged static render; not planner or PC gate"})
    except Exception as exc:
        host.write_new(FAILURE, {"exception": type(exc).__name__, "message": str(exc),
            "completed_boundaries": len(rows), "dependencies": deps,
            "scope": "partial diagnostic; do not interpret as scientific result"})
        raise


if __name__ == "__main__":
    main()
