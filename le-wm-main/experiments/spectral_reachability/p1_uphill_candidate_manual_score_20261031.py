"""Score all reproduced candidates using explicit two-packet context construction."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_uphill_adam_candidate_replay_20261031 as source

NAME = "p1_uphill_candidate_manual_score_20261031"
REPLAY = source.planner.bundle.ROOT / "results/p1_uphill_adam_candidate_replay_20261031_RUN01.json"
REPLAY_SHA = "929732e4a5ed25146de3899a2ff4e1f326e813fe01260fce00a17a2adca0742a"


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    source.planner.configure()
    prior = source.planner.adam.prior
    base = prior.base
    output = Path(output)
    if (not output.is_absolute() or output.parent != base.ROOT / "results"
            or not output.name.startswith(NAME + "_") or output.exists() or output.is_symlink()):
        raise ValueError("exclusive output")
    replay, _ = prior.gap.read(REPLAY, REPLAY_SHA)
    bundle, _ = prior.gap.read(base.BUNDLE, base.BUNDLE_SHA)
    if replay["status"] != "complete" or len(replay["rows"]) != 24 or len(bundle["rows"]) != 24:
        raise ValueError("complete roster")
    sources = base.bindings()
    sources.update({str(Path(__file__).resolve()): base.sha(Path(__file__).resolve()),
                    str(REPLAY): REPLAY_SHA, str(base.BUNDLE): base.BUNDLE_SHA})
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
            for i, (row, saved) in enumerate(zip(bundle["rows"], replay["rows"], strict=True)):
                base.bridge.cp5.blind.validate_row(row)
                if row["index"] != i or saved["index"] != i or any(
                        row[k] != saved[k] for k in ("logical_pair", "role")):
                    raise ValueError("row identity")
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    pixels = np.asarray(image).copy()
                cp5 = model.encode({"pixels": p0._image_tensor(
                    pixels, transform=transform, device=device)})["emb"][0, 0]
                states, history, goal, mean, std = base.bridge.cp5.context(row, cp5, device=device)
                scores = []
                for action, digest in zip(saved["candidate_actions"], saved["candidate_sha256"], strict=True):
                    array = np.asarray(action, dtype="<f4")
                    if array.shape != (2, 5, 2) or not np.isfinite(array).all() \
                            or hashlib.sha256(array.tobytes()).hexdigest() != digest:
                        raise ValueError("candidate action")
                    raw = torch.as_tensor(array[None], dtype=torch.float32, device=device)
                    ctx1 = torch.cat((history, ((raw[:, 0] - mean) / std).reshape(1, 1, 10)), dim=1)
                    pred5 = model.predict(states, model.action_encoder(ctx1))[:, -1:]
                    ctx2 = torch.cat((ctx1[:, -2:], ((raw[:, 1] - mean) / std).reshape(1, 1, 10)), dim=1)
                    terminal = model.predict(torch.cat((states, pred5), dim=1)[:, -3:],
                                             model.action_encoder(ctx2))[0, -1]
                    scores.append(float((terminal - goal[0]).square().mean()))
                if len(scores) != 4 or not np.isfinite(scores).all():
                    raise ValueError("finite candidate scores")
                delta = np.asarray(scores) - np.asarray(saved["scores"])
                rows.append({"index": i, "logical_pair": row["logical_pair"], "role": row["role"],
                             "manual_scores": scores, "delta": delta.tolist(),
                             "manual_winner": prior.pick(scores), "saved_winner": saved["winner"]})
        torch.cuda.synchronize(device)
        if base.model_hash(model) != before or any(base.sha(p) != h for p, h in sources.items()):
            raise ValueError("model or source drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "sources": sources, "host": host,
                  "model_before": before, "model_after": before,
                  "calls_per_row": {"encoder": 1, "action_encoder": 8, "predict": 8},
                  "P1_admission": False, "independent_confirmation": False,
                  "limitations": "Explicit packet construction, but shared cp5 context, preprocessing, "
                                 "normalizer and model; reproduced candidates are not original nonwinner artifacts."}
        digest = base.write(output, result)
    print(json.dumps({"output": str(output), "sha256": digest, "rows": len(rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print("Recompute 96 fixed candidate scores with explicit packet construction on CUDA.", flush=True)
    run(parser.parse_args().output)
