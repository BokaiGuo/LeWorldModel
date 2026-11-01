"""Encode fixed historical cp10 task targets for two known E3 roots."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as base
from . import e3_initial_goal_qualification_20261031 as qualification

ROOT = base.ROOT
NAME = "e3_cp10_target_cuda_20261031"
PROTOCOL = ROOT.parent / "docs/D2P_E3_CP10_TARGET_PROTOCOL_20261031.md"
DIRECTORY = ROOT / "results/agent_p3_h3_witness_physical_20261029_v1_MAIN01"
REPORT = DIRECTORY / "report.json"
REPORT_SHA = "c8fa06be8eaaf6f7515e6e9550c900c71e04f2141d745699b33d1585eb21a2ce"


def collect():
    need = qualification.require
    need(base.sha(REPORT) == REPORT_SHA and base.sha(qualification.INPUT) == qualification.INPUT_SHA
         and base.sha(qualification.ENV_SOURCE) == qualification.ENV_SHA, "sealed inputs")
    doc = json.loads(REPORT.read_bytes())
    data = json.loads(qualification.INPUT.read_bytes())
    qualification.analyze(data)
    journal = DIRECTORY / "journal.jsonl"
    need(base.sha(journal) == doc["journal_sha256"], "journal SHA")
    selected = [row for row in doc["rows"] if row["arm"] == "endpoints"]
    need(len(selected) == 2 and [r["root_slot"] for r in selected] == [0, 1], "fixed roster")
    sources = {str(REPORT): REPORT_SHA, str(journal): doc["journal_sha256"],
               str(qualification.INPUT): qualification.INPUT_SHA,
               str(qualification.ENV_SOURCE): qualification.ENV_SHA,
               str(PROTOCOL): base.sha(PROTOCOL), str(Path(__file__).resolve()): base.sha(Path(__file__).resolve())}
    entries = []
    for root, row in enumerate(selected):
        ref = data["root_provenance"][root]["provenance"]["root_native_branches"][0]
        points = {c["step"]: c for c in row["checkpoints"]}
        need(row["status"] == "horizon_completed" and len(row["responses"]) == 20
             and set(points) == {0, 5, 10, 15, 20}
             and all(points[n]["state"] == ref[f"c{n}"]["state"]
                     and points[n]["pixel_sha256"] == ref[f"c{n}"]["pixel_sha256"]
                     for n in (0, 5)), "witness identity")
        target = points[10]
        need(row["responses"][9]["state"] == target["state"], "target step identity")
        initial = qualification.native(target["state"], row["root_state"])
        need(not initial["initial_success"], "initial native success")
        first = next((i for i, x in enumerate(row["responses"], 1)
                      if qualification.native(target["state"], x["state"])["initial_success"]), None)
        need(first is not None and first <= 10, "historical native witness")
        path = DIRECTORY / target["png"]
        need(path.name == target["png"] and not path.is_symlink()
             and base.sha(path) == target["file_sha256"], "target PNG")
        with Image.open(path) as image:
            pixels = np.asarray(image).copy()
        need(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8
             and hashlib.sha256(pixels.tobytes()).hexdigest() == target["pixel_sha256"], "target pixels")
        sources[str(path)] = target["file_sha256"]
        commands = np.asarray(row["plan"]["commands"], dtype="<f4")
        need(commands.shape == (20, 2) and np.isfinite(commands).all()
             and hashlib.sha256(commands.tobytes()).hexdigest() == row["plan"]["raw_float32_sha256"],
             "action witness")
        entries.append({"root": root, "pixels": pixels, "pixel_sha256": target["pixel_sha256"],
                        "goal_state": target["state"], "root_state": row["root_state"],
                        "historical_first_success_step": first,
                        "initial_native_distance": initial["native7_distance"],
                        "initial_joint_xy_error": initial["joint_xy_error"],
                        "witness10_float32_hex": commands[:10].tobytes().hex()})
    need(all(base.sha(p) == s for p, s in sources.items()), "source drift")
    return entries, sources


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0
    need = qualification.require
    out = Path(output)
    need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_")
         and out.suffix == ".json" and not out.exists() and not out.is_symlink(), "exclusive output")
    entries, sources = collect()
    sources.update(base.bindings())
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        need(before == base.MODEL_HASH, "frozen checkpoint")
        rows = []
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
            transform = p0.image_transform()
            for item in entries:
                latent = model.encode({"pixels": p0._image_tensor(item["pixels"], transform=transform,
                                                                   device=device)})["emb"][0, 0]
                need(latent.shape == (192,) and bool(torch.isfinite(latent).all()), "target latent")
                rows.append({**{k: v for k, v in item.items() if k != "pixels"},
                             "latent": latent.cpu().tolist()})
        torch.cuda.synchronize(device)
        need(base.model_hash(model) == before and all(base.sha(p) == d for p, d in sources.items()), "drift")
        digest = base.write(out, {"schema": NAME, "status": "complete", "host": host, "rows": rows,
                                  "model_before": before, "model_after": before, "sources": sources,
                                  "encoder_calls": 2, "independent_confirmation": False,
                                  "E3_admission": False, "physical_execution": False})
    print(json.dumps({"status": "complete", "sha256": digest, "rows": len(rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
