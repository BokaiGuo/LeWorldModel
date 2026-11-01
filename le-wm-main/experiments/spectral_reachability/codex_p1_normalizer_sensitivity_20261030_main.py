"""Replay fixed executed actions with an alternative archive-statistics normalizer."""
import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_adam_rgb_gap_20261030_main as source

base = source.base
NAME = "codex_p1_normalizer_sensitivity_20261030_main"
STATS = base.ROOT / "results/pusht_d2p_archive_action_statistics_v1_20261026.json"
STATS_SHA = "c816c535e08a946a6f7464c659b23e98022b0375359bf5a7add4edd6e018cdda"


def run(out):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    stats, _ = source.prior.gap.read(STATS, STATS_SHA)
    entries = source.inputs()
    old_result, _ = source.prior.gap.read(
        base.ROOT / "results/codex_p1_adam_rgb_gap_20261030_main_RUN01.json",
        "264d1aaf155896b7ead155624d1ffb074d829be6cee13da80e116500700c48e8")
    if len(entries) != len(old_result["rows"]) or stats["statistics"]["valid_rows"] != 2336736:
        raise ValueError("fixed denominator")
    results = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("checkpoint")
        mean_new = torch.tensor(stats["statistics"]["mean"], dtype=torch.float32, device=device)
        std_new = torch.tensor(stats["statistics"]["sample_std"], dtype=torch.float32, device=device)
        tx = p0.image_transform()
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for entry, old in zip(entries, old_result["rows"], strict=True):
                i, row, physical, plan, raw, rgb, frame = entry
                if i != old["index"] or physical["raw10_sha256"] != old["action_sha256"]:
                    raise ValueError("source identity")
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    cp5_rgb = np.asarray(image).copy()
                cp5 = model.encode({"pixels": p0._image_tensor(cp5_rgb, transform=tx, device=device)})["emb"][0, 0]
                states, history, goal, mean_old, std_old = source.prior.fixed.base.bridge.cp5.context(row, cp5, device=device)
                # Reconstruct physical actions before switching normalization, including history.
                history_raw = history.reshape(1, 2, 5, 2) * std_old + mean_old
                history_new = ((history_raw - mean_new) / std_new).reshape(1, 2, 10)
                for packet in torch.as_tensor(raw, device=device):
                    token = ((packet - mean_new) / std_new).reshape(1, 1, 10)
                    context = torch.cat((history_new, token), dim=1)
                    prediction = model.predict(states, model.action_encoder(context))
                    states = torch.cat((states, prediction[:, -1:]), dim=1)[:, -3:]
                    history_new = context[:, -2:]
                pred = states[0, -1]
                actual = torch.tensor(old["actual_latent"], dtype=torch.float32, device=device)
                mse = lambda a, b: float((a-b).square().mean().item())
                results.append({"index": i, "root_slot": physical["root_slot"], "condition": physical["condition"],
                                "role": physical["role"], "logical_pair": physical["logical_pair"],
                                "action_sha256": physical["raw10_sha256"],
                                "current_prediction_error_mse": old["prediction_error_mse"],
                                "alternate_prediction_error_mse": mse(pred, actual),
                                "current_goal_mse": old["predicted_goal_mse"],
                                "alternate_goal_mse": mse(pred, goal[0]),
                                "alternate_predicted_latent": pred.cpu().tolist()})
        torch.cuda.synchronize(device)
        if before != base.model_hash(model):
            raise ValueError("model mutation")
    result = {"schema": NAME, "status": "complete", "rows": results, "host": host,
              "model_before": before, "model_after": before, "archive_stats_sha256": STATS_SHA,
              "code_sha256": base.sha(Path(__file__).resolve()), "censored_omitted": 5,
              "P1_admission": False, "training_normalizer_certified": False,
              "limitations": "alternative derived from local archive, not proven checkpoint training normalizer; "
                             "fixed known-root actions, no new physical experiment, post-outcome sensitivity only"}
    print(json.dumps({"rows": len(results), "sha256": base.write(out, result)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True)
    out = Path(ap.parse_args().output)
    if not out.is_absolute() or out.parent != base.ROOT / "results" or not out.name.startswith(NAME + "_") or out.exists() or out.is_symlink():
        raise ValueError("exclusive output")
    print("Test archive action-statistics sensitivity on the fixed physical actions, host CUDA only.", flush=True)
    run(out)
