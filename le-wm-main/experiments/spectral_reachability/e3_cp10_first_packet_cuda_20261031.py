"""Predict all fixed E3 first packets and compare with actual RGB successors."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as base
from . import e3_cp10_target_cuda_20261031 as targets

ROOT = base.ROOT
NAME = "e3_cp10_first_packet_cuda_20261031"
PROTOCOL = ROOT.parent / "docs/D2P_E3_CP10_FIRST_PACKET_PREDICTION_PROTOCOL_20261031.md"
TARGET = ROOT / "results/e3_cp10_target_cuda_20261031_RUN01.json"
TARGET_SHA = "a0a524ef256dc8737995ce2f0daac5c15ad7578e2234be42c19c19cfbc864aef"
REPORTS = (
    ("c533033e379eb670cf101e8ff14df655718c4a5367dc83e904a4d15f9083c38b",
     "7b14579fcfc3543eb7b3ce55f06621e400c897dcc5a726d120692afb843879a3"),
    ("d39658f27ae0838428bab3e1ea4529d98f31eacec8006bf46d79713486c5240e",
     "28a5daf2d7b191b33ab3f0885d72ef0f91cf2a2939f390f153eeaf658a688564"),
)
ARMS = ("structured", "native", "random", "immediate_progress")


def need(value, message):
    if not value:
        raise ValueError(message)


def collect():
    need(base.sha(TARGET) == TARGET_SHA, "target receipt")
    receipt = json.loads(TARGET.read_bytes())
    entries, sources = targets.collect()
    sources.update(base.bindings())
    sources.update({str(TARGET): TARGET_SHA, str(PROTOCOL): base.sha(PROTOCOL),
                    str(Path(__file__).resolve()): base.sha(Path(__file__).resolve())})
    rows = []
    for root, (report_sha, saved_sha) in enumerate(REPORTS):
        stem = f"e3_cp10_four_arm_root{root}_20261031"
        directory = ROOT / f"results/{stem}_RUN01"
        report, prepare = directory / "report.json", ROOT / f"results/{stem}_PREPARE01.json"
        saved = ROOT / f"results/e3_cp10_four_arm_root{root}_saved_20261031_RUN01.json"
        need(base.sha(report) == report_sha and base.sha(saved) == saved_sha, "sealed report")
        doc, seal, prep = (json.loads(p.read_bytes()) for p in (report, saved, prepare))
        need(doc["status"] == "recorded_known_root_not_admitted" and not doc["errors"]
             and seal["report_sha256"] == report_sha and doc["E3_admission"] is False
             and [r["arm"] for r in doc["rows"]] == list(ARMS), "report contract")
        target = receipt["rows"][root]
        need(target["root"] == root and target["pixel_sha256"] == entries[root]["pixel_sha256"], "target")
        task = prep["source"]["tasks"][root]
        for path in (report, saved, prepare, directory / "journal.jsonl"):
            sources[str(path)] = base.sha(path)
        need(sources[str(directory / "journal.jsonl")] == doc["journal_sha256"], "journal")
        for row in doc["rows"]:
            arm = row["arm"]
            checks = {c["step"]: c for c in row["checkpoints"]}
            need(row["control_returned"] >= 5 and 0 in checks and 5 in checks, "first packet")
            path = Path(checks[5]["path"])
            need(path.parent == directory and not path.is_symlink()
                 and base.sha(path) == checks[5]["file_sha256"], "actual image file")
            with Image.open(path) as im:
                pixels = np.asarray(im).copy()
            need(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8 and
                 hashlib.sha256(pixels.tobytes()).hexdigest() == checks[5]["pixel_sha256"], "actual pixels")
            encoding_path = directory / f"s{root}_{arm}_b00_encoding.json"
            seal_path = directory / f"s{root}_{arm}_b00_seal.json"
            encoding, node = (json.loads(p.read_bytes()) for p in (encoding_path, seal_path))
            need(encoding["actual_step"] == 0 and node["identity"]["boundary"] == 0 and
                 node["identity"]["observation"]["pixel_sha256"] == checks[0]["pixel_sha256"],
                 "initial context binding")
            action = np.asarray(row["executed_raw"][:5], dtype="<f4")
            commands = np.asarray(node["receipt"]["commands"], dtype="<f4")
            need(action.shape == (5, 2) and action.tobytes() == commands.tobytes(), "executed action")
            history = np.asarray(node["identity"]["raw10"], dtype="<f4")
            need(history.shape == (10, 2) and np.isfinite(history).all(), "history")
            latent_meta = encoding["response"]["latent"]
            latent_path = Path(latent_meta["path"])
            need(latent_path.parent == directory / f"s{root}_{arm}_b00_encode"
                 and not latent_path.is_symlink() and base.sha(latent_path) == latent_meta["file_sha256"],
                 "initial latent file")
            initial_latent = np.load(latent_path, allow_pickle=False)
            need(initial_latent.shape == (192,) and initial_latent.dtype == np.float32
                 and np.isfinite(initial_latent).all()
                 and hashlib.sha256(initial_latent.tobytes()).hexdigest() == latent_meta["array_sha256"],
                 "initial latent bytes")
            sources.update({str(path): checks[5]["file_sha256"], str(encoding_path): base.sha(encoding_path),
                            str(seal_path): base.sha(seal_path), str(latent_path): latent_meta["file_sha256"]})
            rows.append({"root": root, "arm": arm, "task": task, "goal": target["latent"],
                         "initial_latent": initial_latent, "history": history,
                         "action": action, "pixels": pixels, "actual_state": checks[5]["state"],
                         "goal_state": target["goal_state"], "pixel_sha256": checks[5]["pixel_sha256"]})
    need(all(base.sha(p) == d for p, d in sources.items()), "source drift")
    return rows, sources


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    out = Path(output)
    need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_")
         and out.suffix == ".json" and not out.exists() and not out.is_symlink(), "exclusive output")
    items, sources = collect()
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        need(before == base.MODEL_HASH, "checkpoint")
        rows = []
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
            transform = p0.image_transform()
            for item in items:
                tensor = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)
                task = item["task"]
                states = tensor(task["root_context"]["latent_history"]).clone()
                latent = tensor(item["initial_latent"])
                need(states.shape == (3, 192) and latent.shape == (192,), "latent ABI")
                states[-1] = latent
                mean, std = tensor(task["action_mean"]), tensor(task["action_std"])
                need(mean.shape == std.shape == (2,) and bool((std > 0).all()), "normalizer")
                history = ((tensor(item["history"]) - mean) / std).reshape(1, 2, 10)
                token = ((tensor(item["action"]) - mean) / std).reshape(1, 1, 10)
                pred = model.predict(states[None], model.action_encoder(torch.cat((history, token), 1)))[0, -1]
                actual = model.encode({"pixels": p0._image_tensor(item["pixels"], transform=transform,
                                                                   device=device)})["emb"][0, 0]
                goal = tensor(item["goal"])
                need(pred.shape == actual.shape == goal.shape == (192,) and bool(torch.isfinite(pred).all())
                     and bool(torch.isfinite(actual).all()), "output ABI")
                rows.append({"root": item["root"], "arm": item["arm"],
                             "action_sha256": hashlib.sha256(item["action"].tobytes()).hexdigest(),
                             "pixel_sha256": item["pixel_sha256"], "goal": goal.cpu().tolist(),
                             "prediction": pred.cpu().tolist(), "actual": actual.cpu().tolist(),
                             "prediction_goal_mse": float((pred - goal).square().mean()),
                             "actual_goal_mse": float((actual - goal).square().mean()),
                             "prediction_actual_mse": float((pred - actual).square().mean()),
                             "actual_state": item["actual_state"], "goal_state": item["goal_state"]})
        torch.cuda.synchronize(device)
        need(base.model_hash(model) == before and all(base.sha(p) == d for p, d in sources.items()), "drift")
        digest = base.write(out, {"schema": NAME, "status": "complete", "host": host,
                                  "rows": rows, "sources": sources, "model_before": before,
                                  "model_after": before, "calls": {"encoder": 8, "predict": 8,
                                                                    "action_encoder": 8},
                                  "E3_admission": False, "independent_confirmation": False})
    print(json.dumps({"status": "complete", "rows": len(rows), "sha256": digest}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
