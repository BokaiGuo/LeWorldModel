"""Compare fixed trust/best suffix predictions against their real RGB outcomes."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_adam_best_iterate_compare_20261030_main as cmp
from . import codex_p1_adam_feedback_rank_20261030_main as rank
from . import codex_p1_midpoint_stop_guard_20261030_v1 as guard

base = cmp.compare.base
NAME = "codex_p1_best_iterate_rank_20261030_main"
TRUST_DIR = base.ROOT / "results/codex_p1_adam_trust_physical_20261030_main_RUN01"
TRUST_REPORT = "cc8c0d35abc287d97abb141fb4fb12a70e41c72659cf9b6a96b5111536c4601b"
TRUST_SAVED = base.ROOT / "results/codex_p1_adam_trust_physical_20261030_main_SAVED_RUN01.json"
TRUST_SAVED_SHA = "60f3874081d873c9ff7bd8d7f18d6726261db22b12d6d88d1552bd3ac6e5405c"
NEW_DIR = base.ROOT / f"results/{cmp.physical.NAME}_RUN01"
NEW_SAVED = base.ROOT / f"results/{cmp.physical.NAME}_SAVED_RUN01.json"


def inputs():
    rows = cmp.build()["rows"]
    fixed = guard.verified_saved_inputs()
    read = cmp.compare.plan.gap.prior.gap.read
    plans = [read(cmp.TRUST_PLAN, cmp.TRUST_PLAN_SHA)[0], read(cmp.physical.PLAN, cmp.physical.SHA)[0]]
    reports = [read(TRUST_DIR / "run_report.json", TRUST_REPORT)[0],
               read(NEW_DIR / "run_report.json", cmp.REPORT_SHA)[0]]
    saved = [read(TRUST_SAVED, TRUST_SAVED_SHA)[0], read(NEW_SAVED, cmp.SAVED_SHA)[0]]
    if any(len(x) != 24 for x in (rows, fixed, *(p["rows"] for p in plans),
                                   *(r["shards"] for r in reports), *(s["rows"] for s in saved))):
        raise ValueError("fixed roster")
    entries = []
    for i, (row, entry) in enumerate(zip(rows, fixed, strict=True)):
        if row["index"] != i:
            raise ValueError("identity")
        if row["new_minus_trust"] is None:
            continue
        bundle, physical, raw, midpoint, midframe = entry
        terminals, packets = [], []
        for plan, report, save, directory in zip(plans, reports, saved, (TRUST_DIR, NEW_DIR), strict=True):
            packet = np.frombuffer(bytes.fromhex(plan["rows"][i]["second_packet_hex"]), dtype="<f4").copy().reshape(5, 2)
            folder = directory / f"row_{i:02d}"
            shard = report["shards"][i]
            transitions = cmp.compare.trace.transitions(folder, shard["report_sha256"])
            if len(transitions) != 10 or any(transitions[j+5]["actual_bits"] != packet[j].tobytes().hex()
                                             for j in range(5)):
                raise ValueError("executed packet mismatch")
            terminals.append(rank.frame(folder, shard, physical["condition"], 10, save["rows"][i]["completed_endpoint"]))
            packets.append(packet)
        entries.append((bundle, physical, raw[0], midpoint, midframe, packets, terminals, row))
    if len(entries) != 22:
        raise ValueError("uncensored denominator")
    return entries


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    entries, rows = inputs(), []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("checkpoint")
        tx = p0.image_transform()
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for bundle, physical, first, midpoint, midframe, packets, terminals, row in entries:
                with Image.open(io.BytesIO(base64.b64decode(bundle["cp5_png_base64"], validate=True))) as image:
                    cp5_rgb = np.asarray(image).copy()
                def encode(x):
                    return model.encode({"pixels": p0._image_tensor(x, transform=tx, device=device)})["emb"][0, 0]
                cp5, mid = encode(cp5_rgb), encode(midpoint)
                states, history, goal, mean, std = cmp.compare.plan.feedback.second_context(bundle, cp5, mid, first)
                candidates = []
                for packet, (rgb, pixel_hash) in zip(packets, terminals, strict=True):
                    token = ((torch.as_tensor(packet, device=device)-mean)/std).reshape(1, 1, 10)
                    pred = model.predict(states, model.action_encoder(torch.cat((history, token), dim=1)))[0, -1]
                    actual = encode(rgb)
                    def mse(a, b):
                        return float((a-b).square().mean().item())
                    candidates.append({"predicted_goal_mse": mse(pred, goal[0]), "actual_goal_mse": mse(actual, goal[0]),
                                       "prediction_error_mse": mse(pred, actual), "pixel_sha256": pixel_hash,
                                       "predicted_latent": pred.cpu().tolist(), "actual_latent": actual.cpu().tolist()})
                rows.append({"index": row["index"], "root_slot": row["root_slot"], "condition": row["condition"],
                             "role": row["role"], "same_suffix": row["same_suffix_as_trust"],
                             "midpoint_pixel_sha256": midframe["pixel_sha256"], "trust": candidates[0],
                             "best": candidates[1], "physical_delta": row["new_minus_trust"],
                             "regularized_objective_delta": row["objective_delta_vs_trust"]})
        torch.cuda.synchronize(device)
        if before != base.model_hash(model):
            raise ValueError("model mutation")
    result = {"schema": NAME, "status": "complete", "rows": rows, "fixed_roster": 24, "censored": 2,
              "host": host, "model_before": before, "model_after": before,
              "code_sha256": base.sha(Path(__file__).resolve()), "plan_sha256": cmp.physical.SHA,
              "trust_plan_sha256": cmp.TRUST_PLAN_SHA, "physical_report_sha256": cmp.REPORT_SHA,
              "trust_report_sha256": TRUST_REPORT, "P1_admission": False, "independent_confirmation": False,
              "limitations": "post-outcome known roots and duplicated trajectories; censored endpoints excluded"}
    print(json.dumps({"rows": len(rows), "sha256": base.write(output, result)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True)
    output = Path(ap.parse_args().output)
    if not output.is_absolute() or output.parent != base.ROOT / "results" or not output.name.startswith(NAME+"_") or output.exists() or output.is_symlink():
        raise ValueError("exclusive output")
    print("Replay both executed suffixes from the same real RGB midpoint on host CUDA.", flush=True)
    run(output)
