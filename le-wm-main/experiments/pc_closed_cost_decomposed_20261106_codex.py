"""Repeat closed-loop costs with nested physical, rendering, and GPU-event meters."""
import json
import os
import sys
import time
from functools import wraps
from pathlib import Path

import pymunk
import torch
from stable_worldmodel.envs.pusht.env import PushT

import eval as native_eval
from experiments import pc_closed_loop_cost_probe_20261106_codex as base


def timed_gpu(owner, name, label, events):
    original = getattr(owner, name)

    @wraps(original)
    def wrapped(*args, **kwargs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            return original(*args, **kwargs)
        finally:
            end.record()
            events.setdefault(label, []).append((start, end))

    setattr(owner, name, wrapped)


def main():
    event_file = Path(os.environ["PC_GPU_EVENTS_PATH"])
    if event_file.exists():
        raise FileExistsError(event_file)
    print("Measure repeated closed-loop execution, rendering, physics and GPU stream events.", flush=True)
    base.instrument(PushT, "step", "pusht_step_including_observation_reward")
    base.instrument(PushT, "_render_frame", "pusht_render_frame")
    base.instrument(pymunk.Space, "step", "pymunk_physics_step")
    events = {}
    original_load = native_eval.load_compatible_pretrained

    @wraps(original_load)
    def load(*args, **kwargs):
        model = original_load(*args, **kwargs)
        timed_gpu(model, "encode", "encode", events)
        timed_gpu(model, "get_cost", "get_cost", events)
        return model

    native_eval.load_compatible_pretrained = load
    started = time.perf_counter()
    base.main()
    torch.cuda.synchronize()
    result = {key: {"calls": len(items),
                    "gpu_stream_ms": sum(a.elapsed_time(b) for a, b in items)}
              for key, items in events.items()}
    with event_file.open("x", encoding="utf-8") as stream:
        json.dump({"events": result, "full_instrumented_wall_s": time.perf_counter() - started,
                   "limitations": ["Nested stream-event intervals are not additive.",
                       "GPU event intervals include stream idle time and host scheduling gaps.",
                       "Physics step is nested inside PushT.step; rendering also occurs at reset.",
                       "Instrumented wall times include measurement, trace and video overhead."]},
                  stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
