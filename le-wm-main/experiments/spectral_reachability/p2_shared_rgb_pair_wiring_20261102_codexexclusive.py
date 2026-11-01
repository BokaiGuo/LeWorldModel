"""Encode common CP15 RGB and execute the existing matched P2 planner pair.

The CLI consumes an externally SHA-pinned capture bundle. It checks internal
consistency, not whether its producer really observed a physical environment.
No action is dispatched here; a later trusted physical consumer is required.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from . import p2_matched_program_kernel_20261101_parallel as kernel
from . import p2_matched_program_saved_20261101_parallel as saved
from . import p2_matched_program_physical_reuse_20261101_parallel as reuse
from . import p1_dev301_blind_plan_physical_20261101_codex as store_api
from . import p1_constructive_blind_encode_cuda_20261027_v1 as checkpoint

NAME = "p2_shared_rgb_pair_wiring_20261102_codexexclusive"
ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "results/e3_p2_host_cuda_cooperative.lock"
need = kernel.old.require


def pixel_sha(rgb):
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def run_pair(frames, goal_rgb, raw_actions, mean, std, encode, adapter_factory,
             model_scope, emit, *, base, clock=time.monotonic):
    """One shared encoding, then two independent budgeted native kernel calls."""
    frames = list(frames)
    actions = np.asarray(raw_actions, dtype="<f4")
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    need(type(base) is int and base in range(4) and len(frames) == 3
         and all(isinstance(x, np.ndarray) and x.shape == (224, 224, 3) and x.dtype == np.uint8
                 for x in [*frames, goal_rgb]), "three actual RGB frames and goal")
    need(actions.shape == (15, 2) and np.isfinite(actions).all() and (np.abs(actions) <= 1).all()
         and mean.shape == std.shape == (2,) and np.isfinite(mean).all()
         and np.isfinite(std).all() and (std > 0).all(), "actual actions/normalization")
    need(all(callable(f) for f in (encode, adapter_factory, model_scope, emit)), "callbacks")
    latents = []
    shared = {"encode_attempted": 0, "encode_returned": 0, "encode_failed": 0}
    for role, pixels in zip(("control5", "control10", "control15", "goal"),
                            [*frames, goal_rgb], strict=True):
        digest = pixel_sha(pixels)
        emit({"kind": "shared_encode_attempt", "role": role, "pixel_sha256": digest})
        shared["encode_attempted"] += 1
        started = time.monotonic()
        try:
            value = np.asarray(encode(pixels.copy()), dtype=np.float32)
            need(value.shape == (192,) and np.isfinite(value).all(), "encoded latent")
        except BaseException as exc:
            shared["encode_failed"] += 1
            emit({"kind": "shared_encode_failure", "role": role, "error_type": type(exc).__name__})
            raise
        shared["encode_returned"] += 1
        latents.append(value.copy())
        emit({"kind": "shared_encode_return", "role": role, "pixel_sha256": digest,
              "latent_float32_sha256": hashlib.sha256(value.astype("<f4").tobytes()).hexdigest(),
              "wall_seconds": time.monotonic() - started})
    action_history = ((actions[-10:] - mean) / std).reshape(2, 10)
    context = {"layout": "states_with_between_state_actions_v1",
               "latent_history": np.stack(latents[:3]).tolist(),
               "action_history": action_history.tolist()}
    case = {"base": base, "context": context, "goal": latents[3].tolist()}
    emit({"kind": "shared_context", "context_sha256": kernel.old.digest(context),
          "goal_sha256": kernel.old.digest(case["goal"]),
          "raw_history_float32_sha256": hashlib.sha256(actions[-10:].tobytes()).hexdigest(),
          "shared_setup": dict(shared)})
    branches = []
    for arm in kernel.ARMS:
        events = []
        def event(item):
            events.append(deepcopy(item))
            emit({"kind": "branch", "record": item})
        try:
            branch = kernel.run_branch(deepcopy(case), arm, adapter_factory, event, model_scope,
                                       clock=clock)
            saved.validate_branch(case, branch, events)
        except BaseException as exc:
            emit({"kind": "pair_failure", "arm": arm, "error_type": type(exc).__name__})
            raise
        branches.append(branch)
    need(all(row["resource_counts"]["attempted"] == row["resource_counts"]["returned"] == kernel.CAPS
             for row in branches), "matched measured resource axes")
    emit({"kind": "pair_complete", "context_sha256": kernel.old.digest(context),
          "winner_action_sha256": {
              row["arm"]: hashlib.sha256(bytes.fromhex(reuse.winner_action_hex(row))).hexdigest()
              for row in branches}})
    return {"schema": NAME + ".pair.v1", "context": context, "goal": case["goal"],
            "branches": branches, "shared_setup": shared,
            "resource_axes": dict(kernel.CAPS), "equal_total_compute": False,
            "source_authentication": "capture_bundle_producer_not_authenticated",
            "independent_confirmation": False, "P1_admission": False, "P2_admission": False}


def load_capture(path, expected):
    """Validate a local bundle and return the supplied data, not provenance."""
    path = Path(path).absolute()
    need(path.is_file() and not path.is_symlink() and checkpoint.sha(path) == expected, "bundle SHA")
    path = path.resolve(strict=True)
    bundle = json.loads(path.read_bytes())
    need(bundle["schema"] == NAME + ".capture.v1"
         and [row["step"] for row in bundle["frames"]] == [5, 10, 15]
         and len(bundle["actions"]) == 15, "capture schedule")
    arrays = []
    for row in [*bundle["frames"], bundle["goal"]]:
        name = row["file"]
        need(isinstance(name, str) and Path(name).name == name, "flat image path")
        image_path = path.parent / name
        need(not image_path.is_symlink() and checkpoint.sha(image_path) == row["file_sha256"],
             "capture image SHA")
        pixels = store_api.image(image_path)
        need(pixel_sha(pixels) == row["pixel_sha256"], "capture pixel SHA")
        arrays.append(pixels)
    actions = []
    for step, row in enumerate(bundle["actions"], 1):
        command64 = np.asarray(row["dispatch"], dtype=np.float64)
        actual64 = np.asarray(row["returned"], dtype=np.float64)
        command = command64.astype("<f4")
        actual = actual64.astype("<f4")
        need(row["step"] == step and command.shape == actual.shape == (2,)
             and np.isfinite(command).all() and (np.abs(command) <= 1).all()
             and np.array_equal(command64, actual64)
             and np.array_equal(command64, command.astype(np.float64))
             and row["terminated"] is False and row["truncated"] is False,
             "actual action receipt")
        actions.append(actual)
    need(checkpoint.sha(path) == expected, "capture drift")
    return bundle, arrays[:3], arrays[3], np.stack(actions)


def execute_cuda(bundle_path, expected, output):
    """Host CUDA execution only; caller must acquire the cooperative lock."""
    bundle, frames, goal, actions = load_capture(bundle_path, expected)
    metadata = checkpoint.native_metadata()
    store = store_api.Store(output)
    store.emit("run_started", bundle_sha256=expected,
               source_authentication="capture_bundle_producer_not_authenticated")
    result = None
    try:
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.checkpoint_compat import load_compatible_pretrained
        from . import preflight_pusht_d2p_p0 as p0
        from . import codex_p2_multistart_joint_cuda_20261030_v1 as adapter
        from . import codex_p2_program_model_scope_20261030_v1 as scope

        device = torch.device("cuda:0")
        preflight_started = time.monotonic()
        gpu = require_cuda_allocation(device)
        store.emit("cuda_preflight", metadata=gpu, wall_seconds=time.monotonic() - preflight_started)
        store.emit("model_load_attempt")
        load_started = time.monotonic()
        model = load_compatible_pretrained(str(checkpoint.POLICY), device=device).eval().requires_grad_(False)
        torch.cuda.synchronize(device)
        store.emit("model_load_return", wall_seconds=time.monotonic() - load_started)
        checkpoint.frozen(model)
        before = checkpoint.tensor_hash(model)
        need(before == checkpoint.MODEL_TENSOR_SHA, "pinned model tensor hash")
        transform = p0.image_transform()
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        def encode(pixels):
            tensor = p0._image_tensor(pixels, transform=transform, device=device)
            with torch.inference_mode():
                latent = model.encode({"pixels": tensor})["emb"][0, 0]
                torch.cuda.synchronize(device)
                need(latent.device.type == "cuda" and latent.dtype == torch.float32, "CUDA encoding")
                return latent.detach().cpu().numpy().copy()
        def emit(event):
            store.check()
            store.emit("pair_event", event=event)
        def clock():
            torch.cuda.synchronize(device)
            need(torch.cuda.memory_allocated(device) <= 4 * 1024**3
                 and torch.cuda.memory_reserved(device) <= 6 * 1024**3, "CUDA memory cap")
            return time.monotonic()
        with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
            result = run_pair(frames, goal, actions, metadata["mean"], metadata["std"], encode,
                              lambda meter: adapter.MultistartCudaAdapter(model, device, meter),
                              lambda meter: scope.model_scope(model, meter), emit,
                              base=bundle["base"], clock=clock)
        checkpoint.frozen(model)
        after = checkpoint.tensor_hash(model)
        need(before == after, "model drift")
        load_capture(bundle_path, expected)
        data_sha = store.json("pair.json", result)
        store.emit("run_finished", status="completed", pair_sha256=data_sha)
        report = {"schema": NAME + ".report.v1", "status": "completed", "bundle_sha256": expected,
                  "pair_sha256": data_sha, "journal_sha256": checkpoint.sha(store.path / "journal.jsonl"),
                  "journal_tail": store.events[-1]["sha256"], "model_tensor_sha256": before,
                  "source_authentication": "capture_bundle_producer_not_authenticated",
                  "equal_total_compute": False, "P1_admission": False, "P2_admission": False,
                  "physical_execution": False, "independent_confirmation": False}
    except BaseException as exc:
        store.emit("run_finished", status="partial_or_failed", error_type=type(exc).__name__,
                   message=str(exc)[:1000])
        report = {"schema": NAME + ".report.v1", "status": "partial_or_failed",
                  "bundle_sha256": expected, "journal_sha256": checkpoint.sha(store.path / "journal.jsonl"),
                  "journal_tail": store.events[-1]["sha256"], "error_type": type(exc).__name__,
                  "source_authentication": "capture_bundle_producer_not_authenticated",
                  "P1_admission": False, "P2_admission": False}
    digest = store.json("report.json", report)
    return report, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--capture-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-host-cuda", action="store_true")
    args = parser.parse_args()
    need(args.authorize_host_cuda, "explicit host CUDA authorization")
    output = Path(args.output).absolute()
    need(output.parent == ROOT / "results" and output.name.startswith(NAME + "_RUN_")
         and not os.path.lexists(output), "exclusive output")
    with store_api.locking.gpu_lock(LOCK):
        report, digest = execute_cuda(args.capture, args.capture_sha256, output)
    print(json.dumps({"status": report["status"], "report_sha256": digest, "output": str(output)}))
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
