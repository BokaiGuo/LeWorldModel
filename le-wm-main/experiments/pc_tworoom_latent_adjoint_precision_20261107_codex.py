"""Precision-controlled diagnostic of a previously selected adjoint mismatch."""
import json
import types

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments import pc_tworoom_search_region_latent_jvp_20261107_codex as jvp
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_rgb_single_frame_contract_20261107_codex import SingleFrameRGB, model_digest

ROOT = jvp.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_LATENT_ADJOINT_PRECISION_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_tworoom_latent_adjoint_precision_20261107_codex.json"


def dtype_aware_embedder(module, x):
    x = x.to(dtype=module.patch_embed.weight.dtype).permute(0, 2, 1)
    return module.embed(module.patch_embed(x).permute(0, 2, 1))


def projected(rollout, x, direction, projection, epsilons):
    x = x.detach().requires_grad_(True)
    _, tangent = torch.autograd.functional.jvp(rollout, x, direction, strict=True)
    y = rollout(x)
    gradient, = torch.autograd.grad((y * projection).sum(), x)
    left, right = float((tangent * projection).sum()), float((gradient * direction).sum())
    with torch.no_grad():
        fd = {str(eps): float(((rollout(x + eps * direction) - rollout(x - eps * direction))
                               * projection).sum() / (2 * eps)) for eps in epsilons}
    return {"jvp": left, "vjp": right, "adjoint_error": abs(left - right), "projected_fd": fd}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    previous = json.loads(jvp.OUTPUT.read_text())
    if (previous["runner_sha256"] != file_hash(jvp.__file__)
            or previous["protocol_sha256"] != file_hash(jvp.PROTOCOL)
            or previous["parent_sha256"] != file_hash(jvp.previous.OUTPUT)):
        raise ValueError("source drift")
    parent = json.loads(jvp.previous.OUTPUT.read_text())
    row, = (r for r in parent["rows"] if (r["episode"], r["steps"], r["direction"]) == (9000, 25, 0))
    model, scaler, transform, normalizer = jvp.previous.source.common.model_setup()
    if model_digest(model) != previous["model_digest"] or normalizer != previous["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    checkpoint_digest = model_digest(model)
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, jvp.OUTPUT, jvp.previous.OUTPUT,
                         jvp.previous.source.INPUT, jvp.previous.source.common.CHECKPOINT / "weights.pt",
                         jvp.previous.source.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224; TwoRoom action scaler", clock="five steps per packet")
    with np.load(jvp.previous.source.INPUT, allow_pickle=False) as data:
        indexes = np.flatnonzero(data["episodes"] == 9000)
        if len(indexes) != 1:
            raise ValueError("ambiguous episode")
        i = int(indexes[0])
        pixels, goal = transform(data["current"][i]), transform(data["goal"][i])
    rng = np.random.default_rng(27126000 + 9000 * 100 + 25 * 2)
    projection = rng.choice([-1., 1.], size=192) / np.sqrt(192)
    with torch.no_grad():
        probe = torch.arange(20, device="cuda", dtype=torch.float32).reshape(1, 2, 10) / 20
        original = model.action_encoder(probe)
        adapted = dtype_aware_embedder(model.action_encoder, probe)
        adapter_error = float((original - adapted).abs().max())
        if adapter_error != 0:
            raise ValueError("float32 encoder adapter mismatch")
    modes = (("float32_original", torch.float32, True),
             ("float32_no_cudnn_tf32", torch.float32, False),
             ("float64_diagnostic", torch.float64, False))
    rows = []
    print("Selected RGB adjoint discrepancy: float32 backend settings vs float64 diagnostic.", flush=True)
    for name, dtype, cudnn_tf32 in modes:
        model.to(dtype=dtype)
        if dtype == torch.float64:
            model.action_encoder.forward = types.MethodType(dtype_aware_embedder, model.action_encoder)
        digest = model_digest(model)
        guard = SingleFrameRGB(model, audit, expected_model_digest=digest, action_mean=scaler.mean_,
                               action_scale=scaler.scale_, packet_steps=5, physical_low=-1, physical_high=1)
        context = {"pixels": pixels[None, None, None].to(device="cuda", dtype=dtype),
                   "goal": goal[None, None, None].to(device="cuda", dtype=dtype),
                   "action": torch.empty((1, 1, 0, 10), device="cuda", dtype=dtype)}
        raw = torch.tensor(row["physical_actions"], device="cuda", dtype=dtype)
        direction = torch.tensor(row["direction_vector"], device="cuda", dtype=dtype)
        u = torch.tensor(projection, device="cuda", dtype=dtype)
        def forward(x):
            cost, ctx = guard.score(context, jvp.previous.normalize_actions(x, scaler.mean_, scaler.scale_))
            return cost.sum(), ctx["predicted_emb"][0, 0, -1]
        def rollout(x):
            return forward(x)[1]
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        with sdpa_kernel(SDPBackend.MATH):
            with torch.no_grad():
                cost, embedding = forward(raw)
            reps = [projected(rollout, raw, direction, u, jvp.previous.EPSILONS) for _ in range(3)]
        guard.verify_unchanged()
        rows.append({"mode": name, "dtype": str(dtype), "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                     "cudnn_tf32": torch.backends.cudnn.allow_tf32, "model_digest": digest,
                     "cost": float(cost), "embedding": embedding.cpu().tolist(), "repeats": reps})
        print(f"{name} adjoint={reps[0]['adjoint_error']:.12g}", flush=True)
    with OUTPUT.open("x") as f:
        json.dump({"scope": "post-hoc selected RGB numerical diagnostic", "track": RGB,
                   "source_sha256": file_hash(jvp.OUTPUT), "parent_sha256": file_hash(jvp.previous.OUTPUT),
                   "runner_sha256": file_hash(__file__), "protocol_sha256": file_hash(PROTOCOL),
                   "checkpoint_model_digest": checkpoint_digest, "normalizer_sha256": normalizer,
                   "fp32_adapter_max_error": adapter_error, "rows": rows},
                  f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
