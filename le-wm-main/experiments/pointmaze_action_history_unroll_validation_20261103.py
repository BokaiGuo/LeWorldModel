"""CUDA parity checks for explicit-action-history model unroll."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from experiments.a0_pointmaze_proprio_action_response_20261103 import encode_with_state
from experiments.a1_pointmaze_rgb_position_history_diagnostic_20261103 import (
    ARRAYS as PREDS, OUTPUT as SOURCE, PREFIX_SOURCE, SOURCE as RGB_SOURCE,
)
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

SOURCE_SHA = "e14033e10cc0c17a1b7362e723c953f67fe71a833272086df365fece26a1388e"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_ACTION_HISTORY_UNROLL_VALIDATION_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pointmaze_action_history_unroll_validation_20261103_DEV01.json"


def main():
    if OUTPUT.exists() or sha256_file(SOURCE) != SOURCE_SHA:
        raise RuntimeError("output exists or source changed")
    source, rgb_source, prefix_source = (json.loads(p.read_text()) for p in (SOURCE, RGB_SOURCE, PREFIX_SOURCE))
    for path, expected in ((PREDS, source["arrays_sha256"]), (RGB_SOURCE, source["source_sha256"]),
                           (RGB_SOURCE.with_suffix(".npz"), rgb_source["arrays_sha256"]),
                           (PREFIX_SOURCE, source["prefix_source_sha256"]),
                           (DEFAULT_CHECKPOINT, prefix_source["checkpoint_sha256"]),
                           (DEFAULT_DATASET / "states.pth", prefix_source["states_sha256"]),
                           (DEFAULT_DATASET / "actions.pth", prefix_source["actions_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"changed source: {path}")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    rows = []
    print("CUDA: verify observed-action unroll against direct history and single-frame wrapper.", flush=True)
    with np.load(PREDS, allow_pickle=False) as saved, np.load(RGB_SOURCE.with_suffix(".npz"), allow_pickle=False) as rgb, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != source["model_before"]:
            raise RuntimeError("checkpoint identity mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        for row, rgbrow in zip(source["rows"], rgb_source["rows"], strict=True):
            root = row["root"]
            if root != rgbrow["root"]:
                raise RuntimeError("root mismatch")
            results = []
            for i, branch in enumerate(row["branches"]):
                method = branch["initial_method"]
                group = rgbrow["branches"][3*i:3*i+3]
                frames = rgb[f"r{root}_{method}_{group[0]['policy']}_history"]
                first = encode_with_state(model, frames[0], branch["input0"], device)
                second = encode_with_state(model, frames[-1], branch["input5"], device)
                context = {k: torch.cat((first[k], second[k]), dim=1) for k in ("visual", "proprio")}
                history = ((torch.as_tensor(branch["prefix_action"], device=device) - mean) / std).repeat(5)
                tails = ((torch.as_tensor(branch["tail_actions"], device=device) - mean) / std).repeat(1, 5)
                with torch.no_grad():
                    out = unroll_with_action_history(model, context, history[None, None].expand(1, 3, 10),
                                                     tails[None], 2)["visual"][-1]
                    expected = torch.as_tensor(saved[f"r{root}_{method}_history"], device=device)
                    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)
                    future = tails[:2, None, :]
                    ours = unroll_with_action_history(model, second, future[:0], future, 1)
                    reference = model.unroll(TensorDict(dict(second), batch_size=[1, 1], device=device), future)
                    torch.testing.assert_close(ours["visual"], reference["visual"], rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(ours["proprio"], reference["proprio"], rtol=1e-5, atol=1e-6)
                results.append({"method": method,
                                "history_visual_max_abs": float((out - expected).abs().max().cpu()),
                                "single_visual_max_abs": float((ours["visual"] - reference["visual"]).abs().max().cpu()),
                                "single_proprio_max_abs": float((ours["proprio"] - reference["proprio"]).abs().max().cpu())})
            rows.append({"root": root, "branches": results})
            print(f"root {root}: history and single-frame parity verified", flush=True)
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model changed")
    report = {"schema": "pointmaze_action_history_unroll_validation.v1", "source_sha256": SOURCE_SHA,
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "unroll_sha256": sha256_file(ROOT / "experiments/pointmaze_action_history_unroll_20261103.py"),
              "model_before": before, "model_after": after, "device": torch.cuda.get_device_name(device),
              "rtol": 1e-5, "atol": 1e-6, "rows": rows}
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
