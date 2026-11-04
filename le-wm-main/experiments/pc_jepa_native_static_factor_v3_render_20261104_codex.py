"""Replay full physical histories before controlled static rendering."""
import importlib.util
import json
import os
from pathlib import Path
import traceback

import numpy as np
import pymunk

from experiments import pc_jepa_native_feedback_dev_20261104_codex as prior

ROOT, SOURCE = prior.ROOT, prior.OUT
OUT = ROOT / "results/pc_jepa_native_static_factor_v3_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_JEPA_NATIVE_STATIC_FACTOR_V3_PROTOCOL_20261104_CODEX.md"


def main():
    overlay = (ROOT / ".jepa-physics-site").resolve()
    if os.environ.get("PYTHONPATH") != str(overlay):
        raise ValueError("physics overlay required")
    import gym
    import skimage
    if (gym.__version__, pymunk.version, skimage.__version__) != ("0.23.1", "6.8.0", "0.25.2"):
        raise ValueError("unexpected physics versions")
    if any(not Path(m.__file__).resolve().is_relative_to(overlay) for m in (gym, pymunk, skimage)):
        raise ValueError("unexpected physics module origin")
    replay = json.loads((SOURCE / "physical_replay.json").read_text())
    if replay["status"] != "exact" or replay["summary_sha256"] != prior.sha(SOURCE / "summary.json"):
        raise ValueError("unverified replay")
    base = prior.prior.helpers(ROOT)
    runtime, _ = base.load_runtime(ROOT)
    spec = importlib.util.spec_from_file_location("factor_v3_native_env", ROOT / runtime.ENV_REL)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    OUT.mkdir()
    record = {"status": "running", "rows": [], "scope": "post-hoc static images; no planner inputs"}
    error = None
    try:
        manifest_path = SOURCE / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        paths = [PROTOCOL, Path(__file__), ROOT / runtime.ENV_REL, manifest_path, SOURCE / "summary.json",
                 SOURCE / "physical_replay.json", SOURCE / "goal_rgb.npy",
                 SOURCE / "goal_state_evaluation_only.npy",
                 ROOT / "results/pc_jepa_native_contact_overlay_probe_20261104_codex.json"]

        def render(env, no_contacts):
            cls, init = native.DrawOptions, native.DrawOptions.__init__
            if no_contacts:
                def omit(self, surface):
                    init(self, surface)
                    self.flags &= ~pymunk.SpaceDebugDrawOptions.DRAW_COLLISION_POINTS
                cls.__init__ = omit
            try:
                return env.render("rgb_array").copy()
            finally:
                cls.__init__ = init

        goal_env = native.PushTEnv(**base.CTOR)
        goal_env.seed(prior.prior.SEED)
        goal_env.reset_to_state = prior.fixed_goal(prior.prior.SEED)
        goal_obs, goal_state = goal_env.reset()
        np.testing.assert_array_equal(goal_obs["visual"], np.load(SOURCE / "goal_rgb.npy"))
        np.testing.assert_array_equal(goal_state, np.load(SOURCE / "goal_state_evaluation_only.npy"))
        goal_agent, goal_block, goal_angle = (tuple(goal_env.agent.position), tuple(goal_env.block.position),
                                               goal_env.block.angle)
        images = {"goal_native": render(goal_env, False), "goal_no_contacts": render(goal_env, True)}
        np.testing.assert_array_equal(images["goal_native"], goal_obs["visual"])
        goal_env.close()
        for arm in ("response", "cem"):
            path = SOURCE / f"{arm}_physical"
            for boundary in (10, 15, 20, 25):
                env = native.PushTEnv(**base.CTOR)
                env.seed(prior.prior.SEED)
                obs, state = env.reset()
                for frame in range(boundary + 1):
                    directory = path / f"frame_{frame:02d}"
                    for key, observed in (("rgb", obs["visual"]), ("state", state), ("proprio", obs["proprio"])):
                        source = directory / f"{key}.npy"
                        np.testing.assert_array_equal(observed, np.load(source, allow_pickle=False))
                        paths.append(source)
                    if frame < boundary:
                        next_frame = frame + 1
                        if next_frame <= 10:
                            action = manifest["warmup"][next_frame - 1]
                        else:
                            command = path / f"command_{next_frame:02d}.json"
                            paths.append(command)
                            action = json.loads(command.read_text())["action"]
                        obs, _, done, info = env.step(np.asarray(action, np.float64))
                        if done:
                            raise ValueError("unexpected done")
                        state = info["state"]
                agent, block, angle = tuple(env.agent.position), tuple(env.block.position), env.block.angle
                for name, a, b, t in (("base", agent, block, angle), ("agent", goal_agent, block, angle),
                                       ("object", agent, goal_block, goal_angle),
                                       ("both", goal_agent, goal_block, goal_angle)):
                    env.agent.position = a
                    env.block.angle = t
                    env.block.position = b
                    env.space.reindex_shapes_for_body(env.agent)
                    env.space.reindex_shapes_for_body(env.block)
                    for mode, no_contacts in (("native", False), ("no_contacts", True)):
                        images[f"{arm}_{boundary}_{name}_{mode}"] = render(env, no_contacts)
                    if name == "base":
                        np.testing.assert_array_equal(images[f"{arm}_{boundary}_{name}_native"], obs["visual"])
                    if name == "both":
                        np.testing.assert_array_equal(images[f"{arm}_{boundary}_{name}_no_contacts"],
                                                      images["goal_no_contacts"])
                env.agent.position = agent
                env.block.angle = angle
                env.block.position = block
                env.space.reindex_shapes_for_body(env.agent)
                env.space.reindex_shapes_for_body(env.block)
                np.testing.assert_array_equal(render(env, False), obs["visual"])
                env.close()
                record["rows"].append({"arm": arm, "frame": boundary})
                print(f"{arm} frame={boundary}: replay, target geometry and restore exact", flush=True)
        record["bindings"] = {str(p): prior.sha(p) for p in paths}
        with (OUT / "images.npz").open("xb") as f:
            np.savez_compressed(f, **images)
        record.update(status="render_complete", images_sha256=prior.sha(OUT / "images.npz"))
    except Exception:
        error = traceback.format_exc()
        record.update(status="failed", error=error)
    with (OUT / "render.json").open("x") as f:
        json.dump(record, f, indent=2, allow_nan=False)
        f.write("\n")
    if error:
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
