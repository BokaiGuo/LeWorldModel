"""Per-stage frozen-model planning and separately durable resource ledgers.

Not a launcher: its caller must authenticate live inputs, own the cooperative
GPU lock, bound wall/memory, and persist the physical sequence.
"""
import hashlib
import time
from pathlib import Path

import numpy as np

from . import codex_p1_two_stage_feedback_20261030_v1 as bridge
from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as base

WALL = 1800
CAPS = dict.fromkeys(base.spec.AXES, 0)
CAPS.update(action_encode_calls=42, predict_calls=42, predict_samples=42,
            backward_calls=20, vjp_rows=20, replay_calls=1, selection_evaluations=1)
ENC_CAPS = dict(base.ENC_CAPS)


def contract_for(row, source_hashes, stage, input_binding):
    bridge.common.blind.validate_row(row)
    bridge.need(stage in (1, 2) and type(stage) is int and type(input_binding) is dict
                and set(input_binding) == {"rgb_sha256", "previous_latent_sha256", "executed_packet_sha256"},
                "stage input binding")
    bridge.need(all(base.spec.is_hash(x) for x in input_binding.values()), "stage input hashes")
    policy, contract = base.make_contract(row, source_hashes)
    identity = contract["identity"]
    identity["optimizer_config"] = base.spec.digest(bridge.CONFIG)
    identity["history"] = base.spec.digest({"row": row["index"], "stage": stage,
                                            "bundle": base.BUNDLE_SHA, **input_binding})
    for item in contract["arms"].values():
        item["optimizer_config_sha256"] = identity["optimizer_config"]
    contract["levels"][0]["caps"] = dict(CAPS)
    contract["levels"][1]["caps"] = {k: v + 1 for k, v in CAPS.items()}
    base.spec.preflight(contract, base.spec.digest(contract))
    return policy, contract


def tensor_hash(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def run_stage(folder, row, stage, png, previous_latent, executed_packet,
              model, transform, device, source_hashes):
    """Run one phase; caller supplies only authenticated RGB and causal history."""
    bridge.common.blind.validate_row(row)
    bridge.need(stage in (1, 2) and type(stage) is int, "stage")
    pixels = bridge.rgb_pixels(png)
    if stage == 1:
        bridge.need(previous_latent is None and executed_packet is None
                    and hashlib.sha256(pixels.tobytes()).hexdigest() == row["cp5_pixel_sha256"],
                    "cp5 source")
        previous_sha = packet_sha = hashlib.sha256(b"").hexdigest()
    else:
        bridge.need(previous_latent is not None, "prior latent required")
        previous_sha = tensor_hash(previous_latent)
        packet_sha = hashlib.sha256(bridge.packet_bytes(executed_packet)).hexdigest()
    binding = {"rgb_sha256": hashlib.sha256(png).hexdigest(),
               "previous_latent_sha256": previous_sha, "executed_packet_sha256": packet_sha}
    policy, contract = contract_for(row, source_hashes, stage, binding)
    folder = Path(folder)
    bridge.need(folder.is_absolute() and folder.parent.is_dir() and not folder.exists()
                and not folder.is_symlink(), "exclusive stage directory")
    folder.mkdir(mode=0o700)
    started = time.monotonic()
    base.write(folder/"input.json", {"index": row["index"], "stage": stage, "binding": binding,
                                    "contract": contract, "policy": policy})
    error = None
    encoder_sha = None
    receipt = None
    result = None
    latent = None
    try:
        budget = base.encbudget.Budget(folder/"encoder.jsonl", ENC_CAPS, WALL)
        try:
            latent = bridge.encode_fresh(pixels, model, budget, transform, device)
        finally:
            encoder_sha = budget.close()
        if stage == 1:
            context = bridge.common.context(row, latent, device=device)
        else:
            context = bridge.second_context(row, previous_latent, latent, executed_packet)
        contract_sha = base.spec.digest(contract)
        run_id = f"row_{row['index']:02d}_stage_{stage}"
        meter = base.durable.Driver(folder/"planning", contract, contract_sha, "structured", "fixed", run_id)
        try:
            leaf = base.leafmod.LeafAdapter(model, meter, contract, contract_sha)
            raw, replay = bridge.plan_context(context, leaf, policy, contract)
        finally:
            receipt = meter.finish()
        account = base.consumer.consume(folder/"encoder.jsonl", encoder_sha, ENC_CAPS, WALL,
                  receipt["journal"], receipt["sha256"], contract_sha, "structured", "fixed", run_id)
        bridge.need(account["complete"] and account["planning"]["charged_upper_bounds"] == CAPS,
                    "complete stage leaf counts")
        array = raw.detach().cpu().numpy().astype("<f4", copy=False)
        bridge.need(array.shape == (1, 2, 5, 2), "planned horizon shape")
        bridge.packet_bytes(array[0, 0])
        result = {"raw_horizon_f32_hex": array.tobytes().hex(), "execute_packet": 0,
                  "latent_sha256": tensor_hash(latent), "score": replay["scores"][0], "account": account}
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)[:1000]}
    record = {"index": row["index"], "stage": stage, "binding": binding,
              "status": "complete" if error is None else "failed", "error": error,
              "result": result, "encoder_sha256": encoder_sha, "planning_receipt": receipt,
              "wall_seconds": time.monotonic()-started,
              "isolation_certified": False, "P1_admission": False}
    base.write(folder/"report.json", record)
    if error is not None:
        raise RuntimeError("stage failed; report retained: " + str(folder/"report.json"))
    return latent, result
