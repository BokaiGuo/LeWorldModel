"""Host-CUDA replay of fixed original tails and independently optimized tails."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as base
from . import codex_p1_cp5_terminal_rgb_gap_20261030_main as gap
from . import codex_p1_cp5_vector_replay_20261030_main as replay
from . import codex_p1_h10_independent_join_20261030_v1 as source

NAME = "codex_p1_original_tail_score_20261030_main"
VIEW = base.ROOT / "results/codex_p1_complete_paired_view_20261030_main_RUN01.json"
VIEW_SHA = "096210929f78f2d8477a3705713dc4c0498e4bf826d89ae580b8652765d0b850"


def inputs():
    bundle, entries = replay.inputs()
    prep, _ = gap.read(source.H10_PREP, source.H10_SHA)
    view, _ = gap.read(VIEW, VIEW_SHA)
    if len(view["rows"]) != 24 or len(prep["original"]["probe_banks"]) != 8:
        raise ValueError("incomplete roster")
    rows = []
    for i, saved in enumerate(view["rows"]):
        identity = saved["identity"]
        entry = entries[2*i+1]
        if (identity["index"] != i or entry["label"] != "twenty_step"
                or any(identity[k] != entry["identity"][k]
                       for k in ("root_slot", "condition", "arm", "role", "logical_pair"))):
            raise ValueError("identity mismatch")
        probe = prep["original"]["probe_banks"][identity["logical_pair"]][41]
        common = saved["common_suffixes"][41]
        raw = bytes.fromhex(probe["raw10_float32_hex"])
        if (probe["name"] != "original_selected_tail" or len(raw) != 80
                or hashlib.sha256(raw).hexdigest() != probe["raw10_sha256"]
                or probe["raw10_sha256"] != common["probe_sha256"]):
            raise ValueError("original tail mismatch")
        rows.append((bundle["rows"][i], entry, common, raw))
    return rows


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    output = Path(output)
    if (not output.is_absolute() or output.parent != base.ROOT / "results"
            or not output.name.startswith(NAME + "_") or output.exists() or output.is_symlink()):
        raise ValueError("exclusive output")
    inputs_ = inputs()
    sources = base.bindings()
    for path in (Path(__file__).resolve(), Path(replay.__file__).resolve(),
                 Path(gap.__file__).resolve(), VIEW, source.H10_PREP, base.BUNDLE, replay.RGB, gap.PAIRS):
        sources[str(path)] = base.sha(path)
    for _, entry, _, _ in inputs_:
        sources[entry["event_path"]] = entry["event_sha256"]
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("model mismatch")
        transform = p0.image_transform()
        rows = []
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            for row, entry, common, original_bytes in inputs_:
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    pixels = np.asarray(image).copy()
                cp5 = model.encode({"pixels": p0._image_tensor(pixels, transform=transform, device=device)})["emb"][0, 0]
                event, _ = gap.read(Path(entry["event_path"]), entry["event_sha256"])
                original = np.frombuffer(original_bytes, dtype="<f4").copy().reshape(1, 2, 5, 2)
                optimized = np.asarray(event["result"]["actions"], dtype="<f4")
                scores = {}
                for label, actions in (("original", original), ("optimized", optimized)):
                    if actions.shape != (1, 2, 5, 2) or not np.isfinite(actions).all() or (np.abs(actions) > 1).any():
                        raise ValueError("action ABI")
                    pred, goal = replay.replay(model, row, cp5, torch.as_tensor(actions, device=device))
                    scores[label] = float((pred-goal).square().mean())
                if abs(scores["optimized"]-entry["predicted_goal_mse"]) > 1e-5:
                    raise ValueError("optimized score mismatch")
                rows.append({"index": row["index"], "identity": entry["identity"],
                             "original_action_sha256": common["probe_sha256"],
                             "original_physical_status": common["status"],
                             "original_observation": common["observation"],
                             "original_endpoint_metrics": common["common_suffix_endpoint_native_metrics"],
                             "optimized_physical_metrics": entry["physical_metrics"],
                             "model_goal_mse": scores,
                             "original_minus_optimized_score": scores["original"]-scores["optimized"]})
        torch.cuda.synchronize(device)
        if base.model_hash(model) != before or any(base.sha(Path(p)) != h for p, h in sources.items()):
            raise ValueError("source or model drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "sources": sources,
                  "host": host, "model_before": before, "model_after": before,
                  "calls": {"encode": 24, "action_encoder": 96, "predict": 96},
                  "independent_confirmation": False, "P1_admission": False,
                  "limitations": "Post-outcome original actions; known roots and duplicates. Original "
                                 "score versus optimized score is not a fair search or control experiment; "
                                 "native-stopped endpoint stays null under sealed contract."}
        digest = base.write(output, result)
    print(json.dumps({"status": "complete", "rows": len(rows), "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Compare fixed original and optimized suffix model scores, without selecting actions.", flush=True)
    run(args.output)
