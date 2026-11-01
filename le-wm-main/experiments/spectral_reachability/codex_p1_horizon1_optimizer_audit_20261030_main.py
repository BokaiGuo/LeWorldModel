"""Check saved one-packet actions against the existing metered optimizer on CUDA."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from . import codex_p1_horizon1_midpoint_20261030_main as run
from . import codex_p1_horizon1_physical_20261030_main as physical
from . import codex_four_arm_raw_suffix_20261030_main as optimizer

NAME = "codex_p1_horizon1_optimizer_audit_20261030_main"
base, pair, source = run.base, run.pair, run.source


def audit(out):
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0
    from . import codex_p1_two_stage_metered_plan_20261030_v1 as stage

    source.feedback.configure()
    plan = pair.read(physical.PLAN, physical.PLAN_SHA)
    checked = source.feedback.saved.consume(
        str(source.FB_DIR), source.FB_REPORT_SHA, str(source.FB_PREP), source.FB_PREP_SHA)
    report = pair.read(source.FB_DIR / "run_report.json", source.FB_REPORT_SHA)
    bundle = pair.read(base.BUNDLE, base.BUNDLE_SHA)
    pair.need(all(len(x["rows"]) == 24 for x in (plan, checked, bundle)), "fixed denominator")
    sources = base.bindings()
    out.mkdir(mode=0o700)
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        pair.need(before == base.MODEL_HASH, "frozen model identity")
        transform = p0.image_transform()
        with sdpa_kernel(SDPBackend.MATH):
            for i, (p, item, row, shard) in enumerate(zip(
                    plan["rows"], checked["rows"], bundle["rows"], report["shards"], strict=True)):
                pair.need(p["index"] == item["index"] == row["index"] == shard["index"] == i,
                          "row identity")
                folder = source.FB_DIR / f"row_{i:02d}"
                physical_report = pair.read(folder / "physical/report.json", shard["report_sha256"])
                transitions = pair.transitions(folder / "physical", shard["report_sha256"])
                pair.need(len(transitions) == 10, "full source trajectory")
                packet = np.frombuffer(b"".join(bytes.fromhex(t["actual_bits"]) for t in transitions[:5]),
                                       dtype="<f4").copy().reshape(5, 2)
                pair.need(packet.tobytes().hex() == p["first_packet_hex"], "actual first packet")
                rgb, pixel_sha = run.cross.frame(folder, physical_report, item["condition"], 5)
                pair.need(pixel_sha == p["midpoint_pixel_sha256"], "actual midpoint")
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    cp5 = np.asarray(image).copy()
                with torch.no_grad():
                    encode = lambda x: model.encode(
                        {"pixels": p0._image_tensor(x, transform=transform, device=device)})["emb"][0, 0]
                    z5, z10 = encode(cp5), encode(rgb)
                    context = run.cross.pred.bridge.second_context(row, z5, z10, packet)
                png_path = folder / "physical" / next(
                    f["png"] for f in physical_report["row"]["trajectory"]
                    if f["phase"] == "probe" and f["step"] == 5 and f["condition"] == item["condition"]
                    and f["overlay_index"] == (1 if item["condition"] == "on" else 0))
                binding = {"rgb_sha256": base.sha(png_path),
                           "previous_latent_sha256": stage.tensor_hash(z5),
                           "executed_packet_sha256": hashlib.sha256(packet.tobytes()).hexdigest()}
                _, contract = stage.contract_for(row, sources, 2, binding)
                digest = base.spec.digest(contract)
                meter = base.durable.Driver(out / f"row_{i:02d}", contract, digest,
                                            "structured", "fixed", f"audit_{i:02d}")
                try:
                    leaf = base.leafmod.LeafAdapter(model, meter, contract, digest)
                    states, history, goal, mean, std = context
                    initial = torch.zeros((1, 1, 5, 2), device=device, dtype=states.dtype)
                    result = optimizer.optimize(leaf, states, history, initial, goal, mean, std,
                                                stage.bridge.CONFIG.copy())
                finally:
                    receipt = meter.finish()
                actual = result.detach().cpu().numpy().astype("<f4", copy=False).reshape(5, 2)
                expected = np.frombuffer(bytes.fromhex(p["second_packet_hex"]), dtype="<f4").reshape(5, 2)
                diff = float(np.max(np.abs(actual - expected)))
                rows.append({"index": i, "max_abs_action_difference": diff,
                             "bitwise_equal": actual.tobytes() == expected.tobytes(),
                             "planning_journal_sha256": receipt["sha256"]})
        torch.cuda.synchronize(device)
        pair.need(base.model_hash(model) == before, "model mutation")
    result = {"schema": NAME, "status": "complete", "rows": rows, "host": host,
              "model_before": before, "model_after": before,
              "source_plan_sha256": physical.PLAN_SHA, "code_sha256": base.sha(Path(__file__).resolve()),
              "limitations": "same model and CUDA implementation; this does not certify normalizer "
                             "provenance or physical correctness; counterfactual replay cost not performance",
              "P1_admission": False}
    print(json.dumps({"sha256": base.write(out / "report.json", result),
                      "max_diff": max(x["max_abs_action_difference"] for x in rows),
                      "bitwise_equal": sum(x["bitwise_equal"] for x in rows)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    pair.need(out.is_absolute() and out.parent == base.ROOT / "results"
              and out.name.startswith(NAME + "_") and not out.exists() and not out.is_symlink(),
              "exclusive output")
    audit(out)
