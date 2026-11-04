"""Render a fixed post-hoc factorial set without advancing physics."""
import importlib.util
import json
import os
from pathlib import Path
import traceback

import numpy as np

from experiments import pc_jepa_native_feedback_dev_20261104_codex as prior

ROOT = prior.ROOT
SOURCE = prior.OUT
OUT = ROOT / "results/pc_jepa_native_static_factor_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_JEPA_NATIVE_STATIC_FACTOR_PROTOCOL_20261104_CODEX.md"


def main():
    overlay = (ROOT / ".jepa-physics-site").resolve()
    if os.environ.get("PYTHONPATH") != str(overlay):
        raise ValueError("physics overlay required")
    import gym
    import pymunk
    import skimage
    if (gym.__version__, pymunk.version, skimage.__version__) != ("0.23.1", "6.8.0", "0.25.2"):
        raise ValueError("unexpected physics versions")
    if any(not Path(m.__file__).resolve().is_relative_to(overlay) for m in (gym, pymunk, skimage)):
        raise ValueError("unexpected physics module origin")
    audit = json.loads((SOURCE / "physical_replay.json").read_text())
    if audit["status"] != "exact" or audit["summary_sha256"] != prior.sha(SOURCE / "summary.json"):
        raise ValueError("source replay not verified")
    base = prior.prior.helpers(ROOT)
    runtime, _ = base.load_runtime(ROOT)
    spec = importlib.util.spec_from_file_location("factor_native_env", ROOT / runtime.ENV_REL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    OUT.mkdir()
    result = {"status": "running", "rows": [], "bindings": {}, "scope": "post-hoc static rendering only"}
    error = None
    try:
        paths = [PROTOCOL, Path(__file__), SOURCE / "physical_replay.json", SOURCE / "summary.json",
                 SOURCE / "goal_state_evaluation_only.npy", SOURCE / "goal_rgb.npy", ROOT / runtime.ENV_REL]
        goal = np.load(SOURCE / "goal_state_evaluation_only.npy", allow_pickle=False)
        goal_rgb = np.load(SOURCE / "goal_rgb.npy", allow_pickle=False)
        env = module.PushTEnv(**base.CTOR)
        env.seed(prior.prior.SEED)
        env.reset()

        def render(agent, block, angle):
            env.agent.position = tuple(agent)
            env.block.angle = float(angle)
            env.block.position = tuple(block)
            env.space.reindex_shapes_for_body(env.agent)
            env.space.reindex_shapes_for_body(env.block)
            return env.render("rgb_array")

        goal_reconstructed = render(goal[:2], goal[2:4], goal[4])
        np.testing.assert_array_equal(goal_reconstructed, goal_rgb)
        images = {"goal": goal_rgb}
        for arm in ("response", "cem"):
            for frame in (10, 15, 20, 25):
                directory = SOURCE / f"{arm}_physical/frame_{frame:02d}"
                state_path, rgb_path = directory / "full_public_state.json", directory / "rgb.npy"
                paths.extend((state_path, rgb_path))
                state = json.loads(state_path.read_text())["bodies"]
                agent = state["agent"]["position"]
                block = state["block"]["position"]
                angle = state["block"]["angle"]
                for name, a, b, t in (
                    ("base", agent, block, angle), ("agent", goal[:2], block, angle),
                    ("object", agent, goal[2:4], goal[4]), ("both", goal[:2], goal[2:4], goal[4]),
                ):
                    image = render(a, b, t)
                    if name == "base":
                        np.testing.assert_array_equal(image, np.load(rgb_path, allow_pickle=False))
                    if name == "both":
                        np.testing.assert_array_equal(image, goal_rgb)
                    images[f"{arm}_{frame}_{name}"] = image.copy()
                result["rows"].append({"arm": arm, "frame": frame, "source_rgb_sha256": prior.sha(rgb_path)})
                print(f"{arm} frame={frame}: source pixels exact", flush=True)
        env.close()
        result["bindings"] = {str(p): prior.sha(p) for p in paths}
        with (OUT / "images.npz").open("xb") as f:
            np.savez_compressed(f, **images)
        result["images_sha256"] = prior.sha(OUT / "images.npz")
        result["status"] = "render_complete"
    except Exception:
        error = traceback.format_exc()
        result.update(status="failed", error=error)
    with (OUT / "render.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    if error:
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
