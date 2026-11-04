"""Capture pygame primitives drawn during both full native renders."""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pygame

from experiments import pc_jepa_native_feedback_dev_20261104_codex as prior

ROOT = prior.ROOT
OUT = ROOT / "results/pc_jepa_native_render_primitive_probe_20261104_codex.json"


def normalize(value):
    if isinstance(value, (tuple, list, pygame.Color)):
        return [normalize(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pygame.Surface):
        return {"surface_size": list(value.get_size())}
    return value


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if os.environ.get("PYTHONPATH") != str((ROOT / ".jepa-physics-site").resolve()):
        raise ValueError("physics overlay required")
    base = prior.prior.helpers(ROOT)
    runtime, _ = base.load_runtime(ROOT)
    spec = importlib.util.spec_from_file_location("primitive_probe_native", ROOT / runtime.ENV_REL)
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
    calls = []
    originals = {}
    for method in ("polygon", "circle", "lines", "aalines"):
        original = getattr(pygame.draw, method)
        originals[method] = original

        def wrapped(*args, _name=method, _fn=original, **kwargs):
            calls.append({"method": _name, "args": normalize(args), "kwargs": normalize(kwargs)})
            return _fn(*args, **kwargs)

        setattr(pygame.draw, method, wrapped)
    try:
        a = goal.render("rgb_array")
        first = calls[:]
        calls.clear()
        b = other.render("rgb_array")
        second = calls[:]
    finally:
        for method, original in originals.items():
            setattr(pygame.draw, method, original)
        goal.close()
        other.close()
    diffs = [{"index": i, "goal": x, "other": y} for i, (x, y) in
             enumerate(zip(first, second)) if x != y]
    result = {"source_sha256": prior.sha(Path(__file__)), "native_sha256": prior.sha(ROOT / runtime.ENV_REL),
              "replay_matches_saved_goal": bool(np.array_equal(obs["visual"], np.load(prior.OUT / "goal_rgb.npy"))),
              "replay_repeat_equal": bool(np.array_equal(obs["visual"], a)),
              "different_channels": int(np.count_nonzero(a != b)),
              "goal_calls": len(first), "other_calls": len(second), "different_calls": diffs,
              "calls_goal": first, "calls_other": second}
    with OUT.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"calls {len(first)}/{len(second)}, differing calls {len(diffs)}, channels {result['different_channels']}")


if __name__ == "__main__":
    main()
