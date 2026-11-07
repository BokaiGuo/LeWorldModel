"""Frozen RGB latent JVP, finite-difference and VJP development check."""
import json

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments import pc_tworoom_search_region_multihorizon_derivative_20261107_codex as previous
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_rgb_single_frame_contract_20261107_codex import SingleFrameRGB, model_digest

ROOT = previous.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_SEARCH_REGION_LATENT_JVP_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_tworoom_search_region_latent_jvp_20261107_codex.json"


def compare(rollout, raw, direction, projection, epsilons):
    x = raw.detach().requires_grad_(True)
    y, jvp = torch.autograd.functional.jvp(rollout, x, direction, strict=True)
    if y.shape != projection.shape or not torch.isfinite(jvp).all():
        raise ValueError("invalid output or tangent")
    value = rollout(x)
    vjp, = torch.autograd.grad((value * projection).sum(), x)
    left = float((jvp * projection).sum())
    right = float((vjp * direction).sum())
    adj_error = abs(left - right)
    norm = float(torch.linalg.vector_norm(jvp))
    records = {}
    with torch.no_grad():
        for eps in epsilons:
            if eps <= 0 or ((x + eps * direction).abs() > 1).any() or (
                    (x - eps * direction).abs() > 1).any():
                raise ValueError("illegal perturbation")
            fd = (rollout(x + eps * direction) - rollout(x - eps * direction)) / (2 * eps)
            error = float(torch.linalg.vector_norm(jvp - fd))
            records[str(eps)] = {"l2_error": error, "fd_norm": float(torch.linalg.vector_norm(fd)),
                                 "pass": error <= 0.01 + 0.02 * norm}
    return {"latent_shape": list(y.shape), "jvp_norm": norm, "adjoint_left": left,
            "adjoint_right": right, "adjoint_error": adj_error,
            "adjoint_pass": adj_error <= 1e-4 + 1e-4 * abs(left), "finite_difference": records}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    parent = json.loads(previous.OUTPUT.read_text())
    if (parent["input_sha256"] != file_hash(previous.source.INPUT)
            or parent["runner_sha256"] != file_hash(previous.__file__)
            or parent["protocol_sha256"] != file_hash(previous.PROTOCOL)
            or parent["source_sha256"] != file_hash(previous.source.OUTPUT)):
        raise ValueError("parent provenance drift")
    model, scaler, transform, normalizer = previous.source.common.model_setup()
    digest = model_digest(model)
    if digest != parent["model_digest"] or normalizer != parent["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, previous.OUTPUT, previous.source.INPUT,
                         previous.source.common.CHECKPOINT / "weights.pt",
                         previous.source.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224; TwoRoom action scaler", clock="five steps per packet")
    guard = SingleFrameRGB(model, audit, expected_model_digest=digest, action_mean=scaler.mean_,
                           action_scale=scaler.scale_, packet_steps=5, physical_low=-1, physical_high=1)
    rows = []
    print("Frozen RGB latent JVP versus central differences and VJP adjoint, exposed tasks.", flush=True)
    with np.load(previous.source.INPUT, allow_pickle=False) as data:
        for row in parent["rows"]:
            episode, steps = row["episode"], row["steps"]
            matches = np.flatnonzero(data["episodes"] == episode)
            if len(matches) != 1:
                raise ValueError("ambiguous episode")
            i = int(matches[0])
            context = {"pixels": transform(data["current"][i])[None, None, None].cuda(),
                       "goal": transform(data["goal"][i])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            raw = torch.tensor(row["physical_actions"], dtype=torch.float32, device="cuda")
            vector = torch.tensor(row["direction_vector"], dtype=torch.float32, device="cuda")
            def forward(x):
                cost, ctx = guard.score(context, previous.normalize_actions(x, scaler.mean_, scaler.scale_))
                return cost.sum(), ctx["predicted_emb"][0, 0, -1]
            def rollout(x):
                return forward(x)[1]
            with torch.no_grad():
                default_cost, default_latent = forward(raw)
                with sdpa_kernel(SDPBackend.MATH):
                    math_cost, math_latent = forward(raw)
                latent_error = float((default_latent - math_latent).abs().max())
                cost_error = float((default_cost - math_cost).abs())
                if latent_error > 1e-4 or cost_error > 1e-3:
                    raise ValueError(f"attention backend parity failed: {latent_error}, {cost_error}")
                shape = tuple(math_latent.shape)
            rng = np.random.default_rng(27126000 + episode * 100 + steps * 2 + row["direction"])
            projection = rng.choice([-1., 1.], size=shape) / np.sqrt(np.prod(shape))
            projection = torch.tensor(projection, dtype=torch.float32, device="cuda")
            with sdpa_kernel(SDPBackend.MATH):
                result = compare(rollout, raw, vector, projection, previous.EPSILONS)
            rows.append({"episode": episode, "steps": steps, "direction": row["direction"],
                         "backend_latent_max_abs": latent_error, "backend_cost_abs": cost_error, **result})
            print(f"episode={episode} steps={steps} direction={row['direction']} complete", flush=True)
    guard.verify_unchanged()
    with OUTPUT.open("x") as f:
        json.dump({"scope": "exposed RGB development; model latent derivative only", "track": RGB,
                   "parent_sha256": file_hash(previous.OUTPUT), "input_sha256": file_hash(previous.source.INPUT),
                   "runner_sha256": file_hash(__file__), "protocol_sha256": file_hash(PROTOCOL),
                   "model_digest": digest, "normalizer_sha256": normalizer, "rows": rows},
                  f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
