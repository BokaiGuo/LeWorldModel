"""Explicit-opt-in four-arm cp5 encoding and equal-initialization V13 CUDA planner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import p1_dev303_v3_four_arm_cp5_v13_inputs_20261102_exclusive as inputs

NAME = "p1_dev303_v3_four_arm_cp5_v13_model_20261102_exclusive"
ROOT, base, require = inputs.ROOT, inputs.base, inputs.require
PREP = ROOT / "results/p1_dev303_v3_four_arm_cp5_v13_inputs_20261102_exclusive_PREP_01.json"
PREP_SHA = "5076071e3b55a623b842a4d6058f9f9f0f26f5099891a7da59111cffc66668e9"
AUTHORITY = "EXPLICIT_DEV303_FOUR_ARM_CP5_V13_MODEL_ONLY"
engine, encode, planner = inputs.v13.engine, inputs.encode, inputs.planner


def protocol():
    require(base.sha(PREP) == PREP_SHA and base.read(PREP) == inputs.build(), "input contract/source drift")
    inputs.physical.cuda.verify_contract(inputs.physical.audit.CUDA_PREP,
                                         inputs.physical.audit.PINS["cuda_prepare"])
    return {"schema": NAME + ".protocol.v1", "input_sha256": PREP_SHA,
            "source_sha256": base.sha(__file__), "roles": list(inputs.ROLES),
            "seed": inputs.v13.SEED,
            "common_initialization": "V13 zero mean, zero seed tails, same random seed and std",
            "per_arm_target": {"rgb_encode_calls": 1, "action_encoder_calls": 10,
                               "predict_calls": 10, "predict_samples": 3200},
            "ledger_verified": False, "equal_full_budget_certified": False,
            "P1_admission": False, "independent_confirmation": False,
            "physical_actions_authorized": False}


def run_cuda(prepared, prepared_sha, output, authority):
    require(authority == AUTHORITY, "explicit model-only authority")
    prepared, output = Path(prepared), Path(output)
    require(base.sha(prepared) == prepared_sha and base.read(prepared) == protocol(), "prepared protocol drift")
    require(not os.path.lexists(output), "exclusive output")
    value = base.read(PREP)
    output.mkdir()
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    device = torch.device("cuda:0")
    host = require_cuda_allocation("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = load_compatible_pretrained(value["policy"], device=device).eval().requires_grad_(False)
    before = engine.base.blind.host.signal.model_hash(model)
    require(before["original_state_dict"] == value["model_sha256"], "model state SHA")
    task, rows = value["task"], []
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    mean, std, goal = tensor(task["action_mean"]), tensor(task["action_std"]), tensor(task["goal"])[None]
    with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
        for role in value["roles"]:
            pixels = inputs.rgb._frame(Path(role["cp5_png"]).parent,
                                       {"file": Path(role["cp5_png"]).name,
                                        "file_sha256": role["cp5_file_sha256"],
                                        "pixel_sha256": role["cp5_pixel_sha256"]})
            counts = {"rgb_encode_calls": 0, "action_encoder_calls": 0,
                      "predict_calls": 0, "predict_samples": 0}
            hooks = {"image_encoder": 0, "image_projector": 0, "action_encoder": 0, "predictor": 0}
            def hook(name):
                def record(_module, _args):
                    hooks[name] += 1
                return record
            handles = [module.register_forward_pre_hook(hook(name)) for name, module in (
                ("image_encoder", model.encoder), ("image_projector", model.projector),
                ("action_encoder", model.action_encoder), ("predictor", model.predictor))]
            try:
                image = engine.base.p0._image_tensor(pixels, transform=engine.base.p0.image_transform(), device=device)
                counts["rgb_encode_calls"] += 1
                latent = model.encode({"pixels": image})["emb"][0, 0]
                torch.cuda.synchronize(device)
                require(latent.device.type == "cuda", "cp5 encoding device")
                states, actions = encode.contexts(task, role["actual_prefix"], latent.cpu().numpy())
                z0, history = tensor(states)[None], tensor(actions)[None]

                def score(variants):
                    flat = tensor(variants.reshape(engine.SAMPLES * engine.VARIANTS, 2, 5, 2))
                    batch = engine.SAMPLES * engine.VARIANTS
                    z, past = z0.repeat(batch, 1, 1), history.repeat(batch, 1, 1)
                    for packet in range(engine.HORIZON):
                        context = torch.cat((past, ((flat[:, packet] - mean) / std).reshape(batch, 1, 10)), dim=1)
                        counts["action_encoder_calls"] += 1
                        encoded = model.action_encoder(context)
                        counts["predict_calls"] += 1
                        counts["predict_samples"] += batch
                        predicted = model.predict(z, encoded)
                        require(predicted.shape == z.shape and bool(torch.isfinite(predicted).all()),
                                "finite prediction")
                        z, past = torch.cat((z, predicted[:, -1:]), dim=1)[:, -3:], context[:, -2:]
                    return ((z[:, -1] - goal).square().mean(dim=1)
                            .reshape(engine.SAMPLES, engine.VARIANTS).cpu().numpy())

                result = planner.plan_one(role["rgb_contact_geometry"]["contact_direction_raw_xy"], score)
            finally:
                for handle in handles:
                    handle.remove()
            require(counts == base.read(prepared)["per_arm_target"]
                    and hooks == {"image_encoder": 1, "image_projector": 1,
                                  "action_encoder": 10, "predictor": 10}, "observed budget/hooks")
            selected = np.asarray(result["selected"]["actions"], dtype="<f4").reshape(10, 2)
            require(np.isfinite(selected).all() and (np.abs(selected) <= 1).all(), "selected bounded tail")
            rows.append({"role": role["role"], "cp5_pixel_sha256": role["cp5_pixel_sha256"],
                         "actual_prefix_sha256": role["actual_prefix_sha256"], "counts": counts,
                         "hooks": hooks, "tail_actions_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
                         **result})
    torch.cuda.synchronize(device)
    after = engine.base.blind.host.signal.model_hash(model)
    require(before == after and len({r["initial_bank_sha256"] for r in rows}) == 1,
            "model drift or initialization mismatch")
    require(base.read(prepared) == protocol(), "postrun source closure")
    report = {"schema": NAME + ".report.v1", "prepared_sha256": prepared_sha,
              "input_sha256": PREP_SHA, "status": "model_only_complete",
              "model_before": before, "model_after": after, "rows": rows, "host": host,
              "ledger_verified": False, "equal_full_budget_certified": False,
              "physical_actions_authorized": False, "P1_admission": False,
              "independent_confirmation": False}
    return inputs.physical.prior.write_new(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--authority")
    args = parser.parse_args()
    if args.command == "prepare":
        require(args.contract_sha256 is None and args.output is None and args.authority is None, "prepare only")
        print(json.dumps({"sha256": inputs.physical.prior.write_new(args.contract, protocol())}))
    else:
        require(args.contract_sha256 and args.output, "run arguments")
        with engine.base.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            print(json.dumps({"report_sha256": run_cuda(args.contract, args.contract_sha256,
                                                       args.output, args.authority)}))


if __name__ == "__main__":
    main()
