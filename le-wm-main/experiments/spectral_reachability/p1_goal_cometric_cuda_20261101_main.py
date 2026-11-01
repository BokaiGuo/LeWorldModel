"""Measure the existing goal-only encoder co-metric for the fixed return task."""
import argparse
import base64
import json
from pathlib import Path

import numpy as np

from . import e3_cp10_target_cuda_20261031 as source
from . import reacher_d2p_goal_metric_runtime as metric_runtime
from . import reacher_d2p_goal_encoder_metric as metric_definition

NAME = "p1_goal_cometric_cuda_20261101_main"
TARGET = source.ROOT / "results/p1_return_target_cuda_20261101_main_MAIN06.json"
TARGET_SHA = "5bd5f5bbe82fe085cf26ceef12dc8ff6e0bb2fbc784e241ae4f5f8e4f693ed8a"
DOC = source.ROOT.parent / "docs/D2P_P1_GOAL_COMETRIC_PROTOCOL_20261101_MAIN.md"


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    out = Path(output)
    if not out.is_absolute() or out.parent != source.ROOT / "results" or out.exists():
        raise ValueError("exclusive output")
    if source.base.sha(TARGET) != TARGET_SHA:
        raise ValueError("fixed task receipt")
    target = json.loads(TARGET.read_bytes())["rows"][0]
    pixels = np.frombuffer(base64.b64decode(target["pixels_uint8_base64"], validate=True),
                           dtype=np.uint8).reshape(224, 224, 3).copy()
    sources = source.base.bindings()
    for path in (TARGET, DOC, Path(__file__).resolve(), Path(metric_runtime.__file__).resolve(),
                 Path(metric_definition.__file__).resolve()):
        sources[str(path)] = source.base.sha(path)
    with source.base.locking.gpu_lock(source.base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(source.base.MODEL), device=device).eval().requires_grad_(False)
        before = source.base.model_hash(model)
        if before != source.base.MODEL_HASH:
            raise ValueError("frozen checkpoint")
        image = p0._image_tensor(pixels, transform=p0.image_transform(), device=device)
        result = dict(schema=NAME, status="rejected", root=0, host=host, sources=sources,
                      target_sha256=TARGET_SHA, goal_latent=target["latent"], P1_admission=False,
                      physical_future_labels_read=False, training=False)
        try:
            with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
                metric, receipt = metric_runtime.measure_goal(
                    model, image, target["latent"], seed=20261101)
            result.update(status="measured_not_control_validated", measurement=receipt,
                          sqrt_matrix=metric.sqrt_matrix.detach().cpu().tolist(),
                          eigenvalues=metric.eigenvalues.detach().cpu().tolist())
        except Exception as exc:
            result["error"] = dict(type=type(exc).__name__, message=str(exc))
        torch.cuda.synchronize(device)
        after = source.base.model_hash(model)
        if after != before or any(source.base.sha(p) != h for p, h in sources.items()):
            raise ValueError("model/source drift")
        result.update(model_before=before, model_after=after)
        digest = source.base.write(out, result)
    print(json.dumps(dict(status=result["status"], output=str(out), sha256=digest)), flush=True)
    return result["status"] == "measured_not_control_validated"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-host-cuda", action="store_true", required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output) else 1)
