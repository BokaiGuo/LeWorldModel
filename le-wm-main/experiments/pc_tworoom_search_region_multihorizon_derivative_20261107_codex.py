"""Frozen RGB score directional derivatives in physical action coordinates."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments import pc_tworoom_temporal_new_episodes_20261107_codex as source
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_single_frame_contract_20261107_codex import SingleFrameRGB, model_digest

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_SEARCH_REGION_MULTIHORIZON_DERIVATIVE_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_tworoom_search_region_multihorizon_derivative_20261107_codex.json"
EPISODES = (9000, 9013, 9022, 9032)
HORIZONS = (5, 15, 25)
EPSILONS = (0.001, 0.002, 0.01)


def normalize_actions(raw, mean, scale):
    mean = torch.as_tensor(mean, dtype=raw.dtype, device=raw.device)
    scale = torch.as_tensor(scale, dtype=raw.dtype, device=raw.device)
    if raw.ndim != 2 or raw.shape[1] != 2 or raw.shape[0] % 5 or (scale <= 0).any():
        raise ValueError("invalid action or normalizer")
    return ((raw - mean) / scale).reshape(1, 1, -1, 10)


def directions(episode, steps):
    rng = np.random.default_rng(27125000 + episode * 10 + steps)
    return rng.choice([-1.0, 1.0], size=(2, steps, 2)) / np.sqrt(steps * 2)


def measure(score, raw, direction, epsilon):
    if epsilon <= 0 or raw.shape != direction.shape:
        raise ValueError("invalid direction")
    if not torch.isfinite(raw).all() or not torch.isfinite(direction).all():
        raise ValueError("nonfinite input")
    if (raw.abs() > 1).any() or ((raw + epsilon * direction).abs() > 1).any() or (
            (raw - epsilon * direction).abs() > 1).any():
        raise ValueError("physical bounds exceeded")
    x = raw.detach().clone().requires_grad_(True)
    value = score(x)
    gradient, = torch.autograd.grad(value, x)
    ad = float((gradient * direction).sum().detach())
    with torch.no_grad():
        plus = float(score(raw + epsilon * direction))
        minus = float(score(raw - epsilon * direction))
    fd = (plus - minus) / (2 * epsilon)
    error = abs(fd - ad)
    return {"value": float(value.detach()), "ad": ad, "plus": plus, "minus": minus,
            "fd": fd, "abs_error": error, "relative_error": error / max(abs(ad), 1e-12),
            "pass": error <= 0.1 + 0.01 * abs(ad)}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    old = json.loads(source.OUTPUT.read_text())
    if (old["input_sha256"] != file_hash(source.INPUT)
            or old["protocol_sha256"] != file_hash(source.PROTOCOL)
            or old["runner_sha256"] != file_hash(source.__file__)):
        raise ValueError("source provenance drift")
    model, scaler, transform, normalizer = source.common.model_setup()
    if model_fingerprint(model) != old["model_sha256"] or normalizer != old["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    digest = model_digest(model)
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, source.OUTPUT, source.INPUT,
                         source.common.CHECKPOINT / "weights.pt", source.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224; TwoRoom action scaler",
                         clock="five environment steps per packet")
    guard = SingleFrameRGB(model, audit, expected_model_digest=digest, action_mean=scaler.mean_,
                           action_scale=scaler.scale_, packet_steps=5, physical_low=-1, physical_high=1)
    rows = []
    print("RGB frozen score: physical-action AD versus central differences, exposed development tasks.", flush=True)
    with np.load(source.INPUT, allow_pickle=False) as data:
        for episode in EPISODES:
            indices = np.flatnonzero(data["episodes"] == episode)
            matches = [r for r in old["rows"] if r["episode"] == episode]
            if len(indices) != 1 or len(matches) != 1:
                raise ValueError("ambiguous episode")
            idx = int(indices[0])
            plan = np.asarray(matches[0]["arms"]["cem16"]["decisions"][0]["actions"], dtype=np.float32)
            if plan.shape != (25, 2) or not np.isfinite(plan).all() or np.max(np.abs(plan)) > 1.000001:
                raise ValueError("invalid sealed actions")
            context = {"pixels": transform(data["current"][idx])[None, None, None].cuda(),
                       "goal": transform(data["goal"][idx])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            def score(raw):
                values, _ = guard.score(context, normalize_actions(raw, scaler.mean_, scaler.scale_))
                return values.sum()
            clipped = np.clip(plan, -0.7, 0.7)
            for steps in HORIZONS:
                raw = torch.tensor(clipped[:steps], dtype=torch.float32, device="cuda")
                for i, direction in enumerate(directions(episode, steps)):
                    vector = torch.tensor(direction, dtype=torch.float32, device="cuda")
                    results = {str(eps): measure(score, raw, vector, eps) for eps in EPSILONS}
                    rows.append({"episode": episode, "steps": steps, "direction": i,
                                 "clipped_coordinates": int(np.count_nonzero(plan[:steps] != clipped[:steps])),
                                 "physical_actions": clipped[:steps].tolist(),
                                 "direction_vector": direction.tolist(), "results": results})
                print(f"episode={episode} steps={steps} complete", flush=True)
    guard.verify_unchanged()
    payload = {"scope": "exposed RGB development, numerical scalar derivative only", "track": RGB,
               "source_sha256": file_hash(source.OUTPUT), "input_sha256": file_hash(source.INPUT),
               "runner_sha256": file_hash(__file__), "protocol_sha256": file_hash(PROTOCOL),
               "model_digest": digest, "normalizer_sha256": normalizer, "rows": rows}
    with OUTPUT.open("x") as f:
        json.dump(payload, f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
