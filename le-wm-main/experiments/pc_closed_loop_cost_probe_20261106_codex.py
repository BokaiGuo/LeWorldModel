"""Development-only nested wall-clock probe of the seeded closed-loop evaluator.

Timings are deliberately nested and not additive. CUDA synchronization around
each solve makes planning wall time meaningful but perturbs throughput.
"""
import json
import os
import sys
import time
from functools import wraps
from pathlib import Path

import torch
from stable_worldmodel.policy import BasePolicy, WorldModelPolicy
from stable_worldmodel.world.env_pool import EnvPool
from stable_worldmodel.world.world import World

import eval as native_eval
from experiments import pc_eval_seeded_trace_20261105_codex as seeded
from experiments.pc_explicit_prefix_program_20261106_codex import ExplicitPrefixProgram
from experiments.pc_feedback_cem_20261106_codex import FeedbackCEM


records = {}


def instrument(owner, method, label, synchronize=False):
    original = getattr(owner, method)

    @wraps(original)
    def measured(*args, **kwargs):
        if synchronize and torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            if synchronize and torch.cuda.is_available():
                torch.cuda.synchronize()
            entry = records.setdefault(label, {"calls": 0, "wall_s": 0.0})
            entry["calls"] += 1
            entry["wall_s"] += time.perf_counter() - started

    setattr(owner, method, measured)


def main():
    destination = Path(os.environ["PC_COST_PROFILE_PATH"])
    if destination.exists():
        raise FileExistsError(destination)
    instrument(BasePolicy, "_prepare_info", "observation_preprocessing")
    instrument(WorldModelPolicy, "get_action", "policy_get_action")
    instrument(EnvPool, "step", "env_pool_step_including_render")
    instrument(World, "evaluate", "whole_evaluation")
    instrument(FeedbackCEM, "solve", "solver", synchronize=True)
    instrument(ExplicitPrefixProgram, "solve", "solver", synchronize=True)

    original_load = native_eval.load_compatible_pretrained

    @wraps(original_load)
    def load(*args, **kwargs):
        start = time.perf_counter()
        model = original_load(*args, **kwargs)
        records["model_load"] = {"calls": 1, "wall_s": time.perf_counter() - start}
        instrument(model, "encode", "model_encode_dispatch")
        instrument(model, "get_cost", "model_cost_dispatch")
        return model

    native_eval.load_compatible_pretrained = load
    start = time.perf_counter()
    seeded.main()
    records["runner_including_setup_and_output"] = {
        "calls": 1, "wall_s": time.perf_counter() - start
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump({
            "records": records,
            "scope": "one vectorized development evaluation; nested timings are not additive",
            "limitations": [
                "model dispatch timings are asynchronous CPU wall time, not GPU kernel time",
                "solver wall time is CUDA-synchronized, changing throughput",
                "environment pool step includes rendering, wrappers and scheduling",
                "traced evaluator writes per-step physical traces and videos",
                "model loading is included in runner, not whole_evaluation",
            ],
        }, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
