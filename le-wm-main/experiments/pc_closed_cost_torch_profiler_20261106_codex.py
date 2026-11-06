"""Single development evaluation with CPU/CUDA activity attribution."""
import json
import os
from collections import defaultdict
from functools import wraps
from pathlib import Path

import torch
from stable_worldmodel.world.world import World

import runpy


def main():
    destination = Path(os.environ["PC_TORCH_PROFILE_PATH"])
    if destination.exists():
        raise FileExistsError(destination)
    original = World.evaluate

    @wraps(original)
    def profile_evaluation(*args, **kwargs):
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            profile_memory=True,
        ) as profiler:
            value = original(*args, **kwargs)
            torch.cuda.synchronize()
        events = defaultdict(lambda: {"calls": 0, "duration_us": 0.0})
        for event in profiler.events():
            if event.device_type == torch.autograd.DeviceType.CUDA:
                record = events[event.name]
                record["calls"] += 1
                record["duration_us"] += event.time_range.elapsed_us()
        ordered = sorted(({"name": key, **value} for key, value in events.items()),
                         key=lambda row: row["duration_us"], reverse=True)
        operators = sorted(({"name": event.key, "calls": event.count,
                             "cpu_self_us": event.self_cpu_time_total,
                             "device_self_us": event.self_device_time_total}
                            for event in profiler.key_averages()),
                           key=lambda row: row["device_self_us"], reverse=True)
        with destination.open("x") as stream:
            json.dump({"cuda_activity_count": sum(r["calls"] for r in ordered),
                       "cuda_activity_duration_sum_us": sum(r["duration_us"] for r in ordered),
                       "cuda_activity_by_name": ordered,
                       "top_device_operators": operators[:100],
                       "limitations": [
                           "Profiler changes timing and memory behavior; this is a diagnostic run.",
                           "CUDA activity duration sums can overlap and are not end-to-end wall time.",
                           "CUDA activities include kernels and device transfer/memory activity.",
                           "Video output remains enabled; setup outside evaluate is excluded.",
                       ]}, stream, indent=2, allow_nan=False)
        return value

    World.evaluate = profile_evaluation
    print("Profile CPU and CUDA activities during a complete development evaluation.", flush=True)
    runpy.run_module("experiments.pc_eval_seeded_reset_20261105_codex", run_name="__main__")


if __name__ == "__main__":
    main()
