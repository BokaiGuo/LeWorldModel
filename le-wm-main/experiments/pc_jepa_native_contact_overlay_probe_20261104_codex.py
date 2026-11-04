"""Test whether cached collision debug drawing explains target reconstruction."""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pymunk

from experiments import pc_jepa_native_feedback_dev_20261104_codex as prior

ROOT = prior.ROOT
OUT = ROOT / "results/pc_jepa_native_contact_overlay_probe_20261104_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if os.environ.get("PYTHONPATH") != str((ROOT / ".jepa-physics-site").resolve()):
        raise ValueError("physics overlay required")
    base = prior.prior.helpers(ROOT)
    runtime, _ = base.load_runtime(ROOT)
    spec = importlib.util.spec_from_file_location("contact_overlay_native", ROOT / runtime.ENV_REL)
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
    original = native.DrawOptions
    before_goal = goal.render("rgb_array")
    before_other = other.render("rgb_array")

    original_init = original.__init__

    def no_contacts_init(self, surface):
        original_init(self, surface)
        self.flags &= ~pymunk.SpaceDebugDrawOptions.DRAW_COLLISION_POINTS

    original.__init__ = no_contacts_init
    try:
        after_goal = goal.render("rgb_array")
        after_other = other.render("rgb_array")
    finally:
        original.__init__ = original_init
        goal.close()
        other.close()
    result = {"scope": "debug render flags only; not a modified planning observation",
              "source_sha256": prior.sha(Path(__file__)), "native_sha256": prior.sha(ROOT / runtime.ENV_REL),
              "recorded_goal_matches_replayed": bool(np.array_equal(obs["visual"], np.load(prior.OUT / "goal_rgb.npy"))),
              "default_between_environments_channels": int(np.count_nonzero(before_goal != before_other)),
              "without_collision_points_between_environments_channels": int(np.count_nonzero(after_goal != after_other)),
              "disabling_collision_points_changed_goal_channels": int(np.count_nonzero(before_goal != after_goal)),
              "disabling_collision_points_changed_other_channels": int(np.count_nonzero(before_other != after_other))}
    with OUT.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
