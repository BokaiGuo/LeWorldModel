"""Compare native debug-draw callbacks for a replayed target and geometry copy."""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pygame

from experiments import pc_jepa_native_feedback_dev_20261104_codex as prior

ROOT = prior.ROOT
OUT = ROOT / "results/pc_jepa_native_render_callback_probe_20261104_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if os.environ.get("PYTHONPATH") != str((ROOT / ".jepa-physics-site").resolve()):
        raise ValueError("physics overlay required")
    base = prior.prior.helpers(ROOT)
    runtime, _ = base.load_runtime(ROOT)
    spec = importlib.util.spec_from_file_location("callback_probe_native", ROOT / runtime.ENV_REL)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    goal = native.PushTEnv(**base.CTOR)
    goal.seed(prior.prior.SEED)
    goal.reset_to_state = prior.fixed_goal(prior.prior.SEED)
    obs, _ = goal.reset()
    other = native.PushTEnv(**base.CTOR)
    other.seed(prior.prior.SEED)
    other.reset()
    other.agent.position = tuple(goal.agent.position)
    other.block.angle = goal.block.angle
    other.block.position = tuple(goal.block.position)
    other.space.reindex_shapes_for_body(other.agent)
    other.space.reindex_shapes_for_body(other.block)

    class Capture(native.DrawOptions):
        def __init__(self, surface):
            super().__init__(surface)
            self.events = []

        def draw_circle(self, pos, angle, radius, outline_color, fill_color):
            self.events.append({"type": "circle", "pos": list(pos), "angle": angle,
                                "radius": radius, "outline": list(outline_color),
                                "fill": list(fill_color)})
            return super().draw_circle(pos, angle, radius, outline_color, fill_color)

        def draw_polygon(self, verts, radius, outline_color, fill_color):
            self.events.append({"type": "polygon", "verts": [list(v) for v in verts],
                                "radius": radius, "outline": list(outline_color),
                                "fill": list(fill_color)})
            return super().draw_polygon(verts, radius, outline_color, fill_color)

    events = {}
    for name, env in (("goal", goal), ("other", other)):
        canvas = pygame.Surface((512, 512))
        capture = Capture(canvas)
        env.space.debug_draw(capture)
        events[name] = capture.events
    image = other.render("rgb_array")
    mismatch = np.argwhere(image != obs["visual"])
    result = {"scope": "debugging replay only; no planning or factor scores",
              "source_sha256": prior.sha(Path(__file__)),
              "native_sha256": prior.sha(ROOT / runtime.ENV_REL),
              "goal_image_matches_saved": bool(np.array_equal(obs["visual"], np.load(prior.OUT / "goal_rgb.npy"))),
              "mismatch_channels": int(len(mismatch)),
              "mismatch_bbox_yx": ([mismatch[:, :2].min(axis=0).tolist(),
                                     mismatch[:, :2].max(axis=0).tolist()] if len(mismatch) else None),
              "events": events}
    with OUT.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    goal.close()
    other.close()
    print(f"pixel differences: {len(mismatch)}; callback trace saved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
