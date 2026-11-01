"""Compare P1 cp5 pixels against the repository's actual evaluation transform."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from . import p1_uphill_adam_candidate_replay_20261031 as source

NAME = "p1_cp5_eval_transform_cuda_20261031"


def compare_arrays(left, right):
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or a.dtype != b.dtype or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("shape, dtype, or finite input")
    return {"exact_bytes": a.tobytes() == b.tobytes(),
            "max_abs_difference": float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))),
            "left_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "right_sha256": hashlib.sha256(b.tobytes()).hexdigest(),
            "shape": list(a.shape), "dtype": str(a.dtype)}


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import eval as official_eval
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    source.planner.configure()
    prior, base = source.planner.adam.prior, source.planner.adam.prior.base
    output = Path(output)
    if (not output.is_absolute() or output.parent != base.ROOT / "results"
            or not output.name.startswith(NAME + "_") or output.exists() or output.is_symlink()):
        raise ValueError("exclusive output")
    bundle, _ = prior.gap.read(base.BUNDLE, base.BUNDLE_SHA)
    if len(bundle["rows"]) != 24:
        raise ValueError("fixed roster")
    protocol = base.ROOT.parent / "docs/D2P_P1_CP5_EVAL_TRANSFORM_PROTOCOL_20261031_MAIN.md"
    paths = (Path(__file__).resolve(), Path(official_eval.__file__).resolve(), protocol.resolve())
    sources = base.bindings()
    sources.update({str(path): base.sha(path) for path in paths})
    sources[str(base.BUNDLE)] = base.BUNDLE_SHA
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("checkpoint")
        local = p0.image_transform()
        official = official_eval.img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for i, row in enumerate(bundle["rows"]):
                base.bridge.cp5.blind.validate_row(row)
                if row["index"] != i:
                    raise ValueError("roster order")
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    pixels = np.asarray(image).copy()
                raw = torch.from_numpy(pixels).permute(2, 0, 1)
                left = p0._image_tensor(pixels, transform=local, device=device)
                right = official(raw).to(device)[None, None]
                inputs = compare_arrays(left.cpu().numpy(), right.cpu().numpy())
                a = model.encode({"pixels": left})["emb"]
                b = model.encode({"pixels": right})["emb"]
                embeddings = compare_arrays(a.cpu().numpy(), b.cpu().numpy())
                rows.append({"index": i, "logical_pair": row["logical_pair"], "role": row["role"],
                             "cp5_pixel_sha256": row["cp5_pixel_sha256"],
                             "input_comparison": inputs, "embedding_comparison": embeddings})
        torch.cuda.synchronize(device)
        if base.model_hash(model) != before or any(base.sha(p) != h for p, h in sources.items()):
            raise ValueError("model or source drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "sources": sources,
                  "host": host, "model_before": before, "model_after": before,
                  "model_calls": {"encode": 48}, "P1_admission": False,
                  "independent_confirmation": False, "training_preprocessing_certified": False,
                  "limitations": "Known cp5 images only; current evaluation transform, not training provenance."}
        digest = base.write(output, result)
    print(json.dumps({"output": str(output), "sha256": digest, "rows": len(rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print("Compare fixed cp5 input and frozen-model encoding with official evaluation transform on CUDA.", flush=True)
    run(parser.parse_args().output)
