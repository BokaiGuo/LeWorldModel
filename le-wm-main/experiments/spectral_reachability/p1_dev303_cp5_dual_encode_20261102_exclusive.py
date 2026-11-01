"""CUDA-only two-arm cp5 encoding; no planning or physical execution."""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np

from . import p1_dev303_cp5_equal_v13_replan_20261102_exclusive as parent

NAME = "p1_dev303_cp5_dual_encode_20261102_exclusive"
ROOT = parent.ROOT


def protocol():
    source = parent.protocol()
    return {
        "schema": NAME + ".protocol.v1",
        "roles": source["roles"],
        "task": source["task"],
        "policy": source["policy"],
        "model_sha256": source["model_sha256"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "source_archive_sha256": source["source_archive_sha256"],
        "source_task_sha256": source["task_sha256"],
        "operation": "encode two actual cp5 RGB frames only",
        "context": "sealed warmup latents 5/10 plus cp5 encoding; actual last two action packets",
        "rgb_encode_calls": 2,
        "planning": False,
        "physical_execution": False,
        "P1_admission": False,
        "independent_confirmation": False,
        "bindings": {**source["bindings"], str(Path(__file__).resolve()):
                     parent.export.sha(parent.export.read_regular(__file__))},
    }


def contexts(task, prefix, latent):
    """Build three-observation/two-packet model context from actual actions."""
    mean = np.asarray(task["action_mean"], np.float32)
    std = np.asarray(task["action_std"], np.float32)
    actual = np.asarray(prefix, np.float32)
    latent = np.asarray(latent, np.float32)
    parent.require(actual.shape == (5, 2) and latent.shape == (task["latent_dim"],)
                   and np.isfinite(actual).all() and np.isfinite(latent).all(),
                   "actual prefix/cp5 latent")
    states = np.vstack([np.asarray(task["root_context"]["latent_history"][-2:], np.float32),
                        latent])
    actions = np.vstack([np.asarray(task["root_context"]["action_history"][-1], np.float32),
                         ((actual - mean) / std).reshape(10)])
    parent.require(states.shape == (3, task["latent_dim"]) and actions.shape == (2, 10)
                   and np.isfinite(states).all() and np.isfinite(actions).all(),
                   "finite model context")
    return states, actions


def run_cuda(prepared, expected_sha, output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    prepared, output = Path(prepared), Path(output)
    contract = protocol()
    parent.require(parent.export.sha(parent.export.read_regular(prepared)) == expected_sha
                   and parent.export.decode(parent.export.read_regular(prepared)) == contract,
                   "prepared contract")
    parent.require(not os.path.lexists(output), "exclusive output")
    output.mkdir()
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    device = torch.device("cuda:0")
    host = require_cuda_allocation("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = load_compatible_pretrained(contract["policy"], device=device).eval().requires_grad_(False)
    before = parent.engine.base.blind.host.signal.model_hash(model)
    parent.require(before["original_state_dict"] == contract["model_sha256"], "model SHA")
    _, arrays = parent.input_arrays()
    output_arrays = {}
    task = contract["task"]
    with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
        for role in contract["roles"]:
            arm = role["role"]
            pixels = arrays[arm + "_rgb_history"][-1]
            image = parent.engine.base.p0._image_tensor(
                pixels, transform=parent.engine.base.p0.image_transform(), device=device)
            latent = model.encode({"pixels": image})["emb"][0, 0]
            torch.cuda.synchronize(device)
            parent.require(latent.device.type == "cuda", "CUDA cp5 encoding")
            states, actions = contexts(task, role["actual_prefix"], latent.cpu().numpy())
            output_arrays[arm + "_latent_history"] = states
            output_arrays[arm + "_normalized_action_history"] = actions
    after = parent.engine.base.blind.host.signal.model_hash(model)
    parent.require(before == after, "model unchanged")
    parent.require(all(parent.export.sha(parent.export.read_regular(path)) == digest
                       for path, digest in contract["bindings"].items()), "source closure")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **output_arrays)
    raw = buffer.getvalue()
    with (output / "encoded_contexts.npz").open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    report = {"schema": NAME + ".report.v1", "status": "encoded_model_only",
              "protocol_sha256": expected_sha, "encoded_contexts_sha256": parent.export.sha(raw),
              "model_before": before, "model_after": after, "host": host,
              "rgb_encode_calls": 2, "planning": False, "physical_execution": False,
              "P1_admission": False, "independent_confirmation": False}
    return parent.engine.write_new(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--prepare-sha256")
    parser.add_argument("--authorize-model-cuda", action="store_true")
    args = parser.parse_args()
    parent.require(args.tag.isascii() and args.tag.isalnum(), "ASCII alphanumeric tag")
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    if args.command == "prepare":
        parent.require(not args.prepare_sha256 and not args.authorize_model_cuda
                       and not os.path.lexists(output), "prepare only")
        digest = parent.engine.write_new(prepared, protocol())
        print(json.dumps({"prepared": str(prepared), "sha256": digest}, sort_keys=True))
    else:
        parent.require(args.prepare_sha256 and args.authorize_model_cuda, "CUDA authorization and SHA")
        with parent.engine.base.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            digest = run_cuda(prepared, args.prepare_sha256, output)
        print(json.dumps({"output": str(output), "report_sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
