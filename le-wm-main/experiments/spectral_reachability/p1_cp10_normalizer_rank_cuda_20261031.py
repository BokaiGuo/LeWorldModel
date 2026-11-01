"""Fixed cp10 action-ranking sensitivity to two action normalizers."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_cp10_paired_prediction_cuda_20261031 as source

ROOT, need, sha, base = source.ROOT, source.need, source.sha, source.base
NAME = "p1_cp10_normalizer_rank_cuda_20261031"
PROTOCOL = ROOT.parent / "docs/D2P_P1_CP10_NORMALIZER_RANK_PROTOCOL_20261031.md"
STATS = ROOT / "results/pusht_d2p_archive_action_statistics_v1_20261026.json"
STATS_SHA = "c816c535e08a946a6f7464c659b23e98022b0375359bf5a7add4edd6e018cdda"
REFERENCE = ROOT / "results/p1_cp10_paired_prediction_cuda_20261031_RUN01.json"
REFERENCE_SHA = "6dd3ee6f86aad55a963c065982f65d898f647ea287bebe6c659f3d198fe0a3cf"


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    output = Path(output)
    need(output.is_absolute() and output.parent == ROOT / "results" and
         output.name.startswith(NAME + "_") and not output.exists(), "exclusive output")
    need(sha(STATS) == STATS_SHA and sha(REFERENCE) == REFERENCE_SHA, "pinned inputs")
    stats, reference = source.feedback_plan.source.physical.h.read(STATS), \
        source.feedback_plan.source.physical.h.read(REFERENCE)
    need(stats["status"] == "completed" and stats["checkpoint_training_statistics_certified"] is False
         and reference["status"] == "complete" and len(reference["rows"]) == 2, "source contract")
    items, sources = source.inputs()
    for path in (STATS, REFERENCE, PROTOCOL, Path(__file__).resolve(), Path(source.__file__).resolve()):
        sources[str(path)] = sha(path)
    alt_mean = np.asarray(stats["statistics"]["mean"], dtype="<f4")
    alt_std = np.asarray(stats["statistics"]["sample_std"], dtype="<f4")
    need(alt_mean.shape == alt_std.shape == (2,) and np.isfinite(alt_mean).all()
         and np.isfinite(alt_std).all() and (alt_std > 0).all(), "archive statistics")
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        need(before == base.MODEL_HASH, "frozen model")
        transform, rows = p0.image_transform(), []
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
            for item, saved in zip(items, reference["rows"], strict=True):
                blind = item["blind"]
                need(saved["root_slot"] == item["anchor"]["root_slot"], "root identity")
                with Image.open(io.BytesIO(base64.b64decode(blind["cp5_png_base64"], validate=True))) as img:
                    cp5_rgb = np.asarray(img).copy()
                mid_path, mid_receipt = item["frames"]["open"][0]
                cp10_rgb = source.load_image(mid_path, mid_receipt["pixel_sha256"])
                def encode(image):
                    return model.encode({"pixels": p0._image_tensor(image, transform=transform,
                                                                      device=device)})["emb"][0, 0]
                cp5, cp10 = encode(cp5_rgb), encode(cp10_rgb)
                states, root_history, goal, mean, std = base.bridge.cp5.context(blind, cp5, device=device)
                states = torch.cat((states, cp10[None, None]), 1)[:, -3:]
                prefix = np.frombuffer(bytes.fromhex(blind["prefix_raw_f32_hex"]), "<f4").copy().reshape(5, 2)
                first = item["open_raw"][0]
                need(first.tobytes() == item["feedback_raw"][0].tobytes(), "same actual history")
                raw_history = np.stack((prefix, first))
                baseline_token = torch.cat((root_history[:, -1:],
                    ((torch.tensor(first, device=device) - mean) / std).reshape(1, 1, 10)), 1)
                variants = (("deployed", mean, std),
                            ("archive", torch.tensor(alt_mean, device=device), torch.tensor(alt_std, device=device)))
                arms = {}
                for label in ("open", "feedback"):
                    raw = item[label + "_raw"][1]
                    need(hashlib.sha256(raw.tobytes()).hexdigest() == saved["arms"][label]["action_sha256"],
                         "action hash")
                    actual = torch.tensor(saved["arms"][label]["actual"], dtype=torch.float32, device=device)
                    scores = {}
                    for name, m, s in variants:
                        raw_tensor = torch.tensor(np.concatenate((raw_history, raw[None])), device=device)
                        tokens = ((raw_tensor - m) / s).reshape(1, 3, 10)
                        if name == "deployed":
                            need(torch.equal(tokens[:, :2], baseline_token), "history token binding")
                        predicted = model.predict(states, model.action_encoder(tokens))[0, -1]
                        if name == "deployed":
                            previous = torch.tensor(saved["arms"][label]["prediction"], device=device)
                            need(torch.allclose(predicted, previous, rtol=1e-5, atol=1e-6),
                                 "reference prediction binding")
                        scores[name] = {"predicted_goal_mse": float((predicted-goal[0]).square().mean()),
                                        "prediction_actual_mse": float((predicted-actual).square().mean()),
                                        "prediction": predicted.cpu().tolist(), "tokens": tokens[0].cpu().tolist()}
                    arms[label] = {"action_sha256": saved["arms"][label]["action_sha256"], "variants": scores}
                rows.append({"root_slot": saved["root_slot"], "goal": goal[0].cpu().tolist(),
                             "actual": {a: saved["arms"][a]["actual"] for a in ("open", "feedback")},
                             "arms": arms})
        torch.cuda.synchronize(device)
        need(base.model_hash(model) == before and all(sha(p) == d for p, d in sources.items()), "source drift")
        result = {"schema": NAME, "status": "complete", "host": host, "model_before": before,
                  "model_after": before, "rows": rows, "sources": sources,
                  "normalizers": {"archive_mean": alt_mean.tolist(), "archive_std": alt_std.tolist()},
                  "calls": {"encoder": 4, "action_encoder": 8, "predict": 8},
                  "P1_admission": False, "independent_confirmation": False,
                  "limitations": "Fixed known actions; archive normalizer not certified as training normalizer."}
        digest = base.write(output, result)
    print(json.dumps({"status": "complete", "rows": len(rows), "sha256": digest}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
