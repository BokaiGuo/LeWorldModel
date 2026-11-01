"""Re-encode all original warmup RGB frames and verify P1 history tokens."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_uphill_adam_candidate_replay_20261031 as planner
from . import p1_uphill_cp5_bundle_20261030 as bundle_builder
from . import p1_backtrack_physical_runner_20261030 as physical
from . import p1_cp5_eval_transform_cuda_20261031 as comparison
from .p1_cp5_eval_transform_cuda_20261031 import compare_arrays

NAME = "p1_history_rgb_binding_cuda_20261031"
PREP_SHA = "890e38ecf7758ae739c25e9d355533e4d3b31be91a8c063c9625d1886c77d56d"


def normalized_actions(raw, mean, std):
    raw, mean, std = (np.asarray(x, dtype=np.float32) for x in (raw, mean, std))
    if raw.shape != (10, 2) or mean.shape != (2,) or std.shape != (2,) \
            or not all(np.isfinite(x).all() for x in (raw, mean, std)) or not (std > 0).all():
        raise ValueError("warmup or normalizer")
    return ((raw - mean) / std).reshape(2, 10)


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    base = planner.planner.adam.prior.base
    output = Path(output)
    if (not output.is_absolute() or output.parent != base.ROOT / "results"
            or not output.name.startswith(NAME + "_") or output.exists() or output.is_symlink()):
        raise ValueError("exclusive output")
    bundle, _ = planner.planner.adam.prior.gap.read(bundle_builder.OUT, planner.planner.BUNDLE_SHA)
    if bundle_builder.build(PREP_SHA) != bundle:
        raise ValueError("bundle source reconstruction")
    planner.planner.configure()
    prior = planner.planner.adam.prior
    prep, _ = prior.gap.read(bundle_builder.PREP, PREP_SHA)
    physical_prepare, data = physical.material()
    if len(bundle["rows"]) != 24 or len(prep["anchors"]) != 24:
        raise ValueError("fixed roster")
    paths = (Path(__file__).resolve(), Path(bundle_builder.__file__).resolve(),
             Path(comparison.__file__).resolve(),
             Path(physical.__file__).resolve(),
             base.ROOT.parent / "docs/D2P_P1_HISTORY_RGB_BINDING_PROTOCOL_20261031_MAIN.md")
    sources = base.bindings()
    sources.update(physical_prepare["bindings"])
    sources.update({str(path): base.sha(path) for path in paths})
    sources.update({str(base.BUNDLE): base.BUNDLE_SHA, str(bundle_builder.PREP): PREP_SHA})
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("checkpoint")
        transform = p0.image_transform()
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for i, (row, anchor) in enumerate(zip(bundle["rows"], prep["anchors"], strict=True)):
                base.bridge.cp5.blind.validate_row(row)
                if (row["index"] != i or anchor["index"] != i
                        or row["logical_pair"] != anchor["logical_pair"] or row["role"] != anchor["role"]):
                    raise ValueError("row identity")
                root = data["roots"][anchor["root_slot"]]
                if root["root_sha256"] != anchor["physical_root_sha256"]:
                    raise ValueError("root identity")
                packet = root["packets"][anchor["condition"]]
                actions = compare_arrays(normalized_actions(packet["raw_action_history"], row["action_mean"],
                                                             row["action_std"]),
                                         np.asarray(row["root_action_history"], np.float32))
                history = np.asarray(packet["rgb_history"])
                if history.shape != (3, 224, 224, 3) or history.dtype != np.uint8:
                    raise ValueError("RGB history ABI")
                frames = []
                for step, rgb in enumerate(history):
                    encoded = model.encode({"pixels": p0._image_tensor(rgb, transform=transform, device=device)})["emb"]
                    if encoded.shape != (1, 1, 192):
                        raise ValueError("latent ABI")
                    actual = encoded[0, 0].cpu().numpy()
                    saved = np.asarray(row["root_latent_history"][step], np.float32)
                    frames.append({"history_index": step, "pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                                   "comparison": compare_arrays(actual, saved)})
                rows.append({"index": i, "logical_pair": row["logical_pair"], "role": row["role"],
                             "root_slot": anchor["root_slot"], "condition": anchor["condition"],
                             "physical_root_sha256": root["root_sha256"], "actions": actions, "frames": frames})
        torch.cuda.synchronize(device)
        if base.model_hash(model) != before or any(base.sha(path) != digest for path, digest in sources.items()):
            raise ValueError("model or source drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "sources": sources, "host": host,
                  "model_before": before, "model_after": before, "encode_calls": 72,
                  "independent_confirmation": False, "training_normalizer_certified": False,
                  "P1_admission": False}
        digest = base.write(output, result)
    print(json.dumps({"sha256": digest, "rows": len(rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print("Compare all fixed real RGB history frames and warmup tokens on host CUDA.", flush=True)
    run(parser.parse_args().output)
