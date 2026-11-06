"""Synchronized GPU/encoding cost accounting on saved real planning contexts."""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from gymnasium.spaces import Box
from sklearn.preprocessing import StandardScaler

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_explicit_prefix_program_20261106_codex import ExplicitPrefixProgram
from experiments.pc_feedback_cem_20261106_codex import FeedbackCEM
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint


class TimedModel:
    def __init__(self, model):
        self.model = model
        self.enc = []
        self.calls = []
        self._original_encode = model.encode
        model.encode = self._encode

    def parameters(self):
        return self.model.parameters()

    def _encode(self, *args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        ev_start, ev_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ev_start.record()
        result = self._original_encode(*args, **kwargs)
        ev_end.record()
        torch.cuda.synchronize()
        self.enc.append({"wall_s": time.perf_counter() - start, "gpu_ms": ev_start.elapsed_time(ev_end)})
        return result

    def get_cost(self, info, actions):
        torch.cuda.synchronize()
        start = time.perf_counter()
        ev_start, ev_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ev_start.record()
        output = self.model.get_cost(info, actions)
        ev_end.record()
        torch.cuda.synchronize()
        self.calls.append({"wall_s": time.perf_counter() - start, "gpu_ms": ev_start.elapsed_time(ev_end),
                           "candidates": int(actions.shape[0] * actions.shape[1])})
        return output

    def close(self):
        self.model.encode = self._original_encode


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "results/pc_gpu_cost_profile_dev_20261106_codex.json"
    if output.exists():
        raise FileExistsError(output)
    print("Measure synchronized planning, get_cost, encoding, GPU events and memory on saved contexts.", flush=True)
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        actions = np.asarray(data["action"][:])
    scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    fingerprint = model_fingerprint(model)
    space = Box(-1, 1, shape=(1, 2), dtype=np.float32)
    config = SimpleNamespace(action_block=5, horizon=5)
    results = []
    configs = [("cem_105x6", 105, 6, 10), ("cem_300x10", 300, 10, 30),
               ("cem_300x30", 300, 30, 30), ("prefix_632", 100, 3, 10)]

    def run(task, name, samples, steps, topk, seed):
        capture = root / f"results/pc_branch_second_replan_lookahead_seed42_20261106_codex.capture/task_{task:02d}_replan_call_00.pt"
        saved = torch.load(capture, map_location="cpu", weights_only=False)
        info = {k: saved["context"][k][:, 0] for k in ("pixels", "goal")}
        # get_cost removes this key from the goal branch and rollout overwrites
        # it with the scored candidate; it is not an additional observation.
        info["action"] = torch.zeros(1, 1, 10)
        wrapped = TimedModel(model)
        try:
            kwargs = dict(model=wrapped, action_mean=scaler.mean_, action_scale=scaler.scale_,
                          batch_size=1, num_samples=samples, n_steps=steps, topk=topk,
                          seed=seed, device="cuda")
            solver = (ExplicitPrefixProgram(prefix_mode="lookahead", **kwargs) if name == "prefix_632"
                      else FeedbackCEM(selection="mean", **kwargs))
            solver.configure(action_space=space, n_envs=1, config=config)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            ev_start, ev_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            ev_start.record()
            result = solver.solve(info)
            ev_end.record()
            torch.cuda.synchronize()
            wall = time.perf_counter() - started
            calls, enc = wrapped.calls, wrapped.enc
            scored = sum(c["candidates"] for c in calls)
            expected = 632 if name == "prefix_632" else samples * steps + 2
            if scored != expected:
                raise ValueError(f"score count mismatch {scored} != {expected}")
            return {"task": task, "method": name, "seed": seed, "scored": scored,
                    "model_calls": len(calls), "encode_calls": len(enc), "plan_wall_s": wall,
                    "plan_gpu_ms": ev_start.elapsed_time(ev_end),
                    "model_wall_s": sum(c["wall_s"] for c in calls),
                    "model_gpu_ms": sum(c["gpu_ms"] for c in calls),
                    "encode_wall_s": sum(c["wall_s"] for c in enc),
                    "encode_gpu_ms": sum(c["gpu_ms"] for c in enc),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "output_model_cost": result["costs"][0]}
        finally:
            wrapped.close()

    run(15, "cem_105x6", 105, 6, 10, 9)
    for task in (15, 25, 35):
        for name, samples, steps, topk in configs:
            for seed in (42, 1042, 2042):
                results.append(run(task, name, samples, steps, topk, seed))
                print(task, name, seed, flush=True)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model mutated")
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"model_sha256": fingerprint, "device": torch.cuda.get_device_name(),
                   "cost_scope": "synchronized fixed context planning; nested timings",
                   "records": results}, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
