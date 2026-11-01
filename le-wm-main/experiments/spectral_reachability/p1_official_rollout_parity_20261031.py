"""Compare the P1 two-packet rollout to the checkpoint's native rollout."""
import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_uphill_all_candidates_midpoint_20261031 as panel
from . import p1_candidate_short_horizon_alignment_20261031 as previous

base = panel.base
ROOT = base.ROOT
NAME = "p1_official_rollout_parity_20261031"
PROTOCOL = ROOT.parent / "docs/D2P_P1_OFFICIAL_ROLLOUT_PARITY_PROTOCOL_20261031.md"


class EncodedContext:
    """Provide pre-encoded context while delegating dynamics to the real model."""

    def __init__(self, model, states):
        self.model = model
        self.states = states
        self.action_encoder = model.action_encoder
        self.predict = model.predict

    def encode(self, info):
        return {"emb": self.states.clone()}


def compare(pred5, pred10, rollout):
    import torch
    if tuple(rollout.shape) != (1, 1, 5, 192):
        raise ValueError("native rollout shape")
    if not bool(torch.isfinite(rollout).all()):
        raise ValueError("nonfinite rollout")
    return (float((pred5 - rollout[:, 0, 3:4]).abs().max()),
            float((pred10 - rollout[:, 0, 4:5]).abs().max()))


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    out = Path(output)
    if not out.is_absolute() or out.parent != ROOT / "results" or \
            not out.name.startswith(NAME + "_") or out.exists() or out.is_symlink():
        raise ValueError("exclusive output")
    entries, sources = panel.inputs()
    old = panel.read(previous.MID, previous.MID_SHA)
    oldrows = {(r["index"], r["kind"]): r for r in old["rows"]}
    if old["status"] != "complete" or len(oldrows) != 96:
        raise ValueError("previous complete predictions")
    sources.update(base.bindings())
    sources.update({str(previous.MID): previous.MID_SHA, str(PROTOCOL): base.sha(PROTOCOL),
                    str(Path(__file__).resolve()): base.sha(__file__)})
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        model_source = Path(__import__(type(model).__module__, fromlist=["x"]).__file__).resolve()
        sources[str(model_source)] = base.sha(model_source)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("checkpoint")
        transform = p0.image_transform()
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for entry in entries:
                row = entry["blind"]
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    pixels = np.asarray(image).copy()
                cp = model.encode({"pixels": p0._image_tensor(pixels, transform=transform, device=device)})["emb"][0, 0]
                states, history, goal, mean, std = base.bridge.cp5.context(row, cp, device=device)
                raw = torch.as_tensor(entry["action"][None], dtype=torch.float32, device=device)
                tokens = ((raw - mean) / std).reshape(1, 2, 10)
                a1 = torch.cat((history, tokens[:, :1]), dim=1)
                pred5 = model.predict(states, model.action_encoder(a1))[:, -1:]
                a2 = torch.cat((a1[:, -2:], tokens[:, 1:]), dim=1)
                pred10 = model.predict(torch.cat((states, pred5), dim=1)[:, -3:],
                                       model.action_encoder(a2))[:, -1:]
                actions = torch.cat((history, tokens), dim=1)[:, None]
                info = {"pixels": torch.zeros((1, 1, 3, 1), dtype=torch.float32, device=device)}
                native = type(model).rollout(EncodedContext(model, states), info, actions)["predicted_emb"]
                d5, d10 = compare(pred5, pred10, native)
                saved = oldrows[(entry["index"], entry["kind"])]
                scores = [float((pred5[0, 0] - goal[0]).square().mean()),
                          float((pred10[0, 0] - goal[0]).square().mean())]
                score_errors = [abs(scores[0] - saved["predicted_midpoint_goal_mse"]),
                                abs(scores[1] - saved["open_terminal_goal_mse"])]
                if entry["candidate"] != saved["candidate"] or max(score_errors) > 1e-5:
                    raise ValueError("saved prediction binding")
                rows.append(dict(index=entry["index"], candidate=entry["candidate"], kind=entry["kind"],
                                 max_latent_abs_t5=d5, max_latent_abs_t10=d10,
                                 saved_score_abs_t5=score_errors[0], saved_score_abs_t10=score_errors[1]))
        torch.cuda.synchronize(device)
        after = base.model_hash(model)
        if before != after or any(base.sha(path) != digest for path, digest in sources.items()):
            raise ValueError("model or source drift")
        digest = base.write(out, dict(schema=NAME, status="complete", host=host, rows=rows, sources=sources,
                                      model_before=before, model_after=after, encoder_calls=96,
                                      predictor_calls=384, action_encoder_calls=384,
                                      physical_execution=False, scientific_admission=False,
                                      independent_confirmation=False))
    print(json.dumps({"sha256": digest, "rows": len(rows),
                      "max_t5": max(r["max_latent_abs_t5"] for r in rows),
                      "max_t10": max(r["max_latent_abs_t10"] for r in rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print("Compare all fixed P1 candidates with native rollout on host CUDA.", flush=True)
    run(parser.parse_args().output)
