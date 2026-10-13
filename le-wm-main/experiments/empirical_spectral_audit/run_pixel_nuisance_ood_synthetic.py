"""Learned pixel-world benchmark with controlled visual nuisance OOD.

The object follows exact torus translations while an independent smooth
background nuisance is held fixed across each transition.  The experiment
measures whether action-conditioned latent prediction and operator geometry
remain stable when nuisance amplitude increases, and reports nuisance/action
tangent ratios as a direct bridge to the frozen visual-encoder audit.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    from module import SIGReg
    from experiments.empirical_spectral_audit.regularizers import (
        operator_alignment_diagnostic,
        operator_alignment_loss,
    )
    from experiments.empirical_spectral_audit.run_pixel_world_model_synthetic import (
        ACTION_VECTORS,
        PixelWorldModel,
    )
except ModuleNotFoundError:  # direct invocation from the experiment package
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from module import SIGReg
    from experiments.empirical_spectral_audit.regularizers import (
        operator_alignment_diagnostic,
        operator_alignment_loss,
    )
    from experiments.empirical_spectral_audit.run_pixel_world_model_synthetic import (
        ACTION_VECTORS,
        PixelWorldModel,
    )


@dataclass(frozen=True)
class PixelNuisanceConfig:
    image_size: int = 16
    latent_dim: int = 8
    train_samples: int = 256
    validation_samples: int = 128
    test_samples: int = 256
    rollout_trajectories: int = 128
    rollout_horizon: int = 8
    epochs: int = 120
    early_stopping_patience: int = 15
    learning_rate: float = 2e-3
    sigreg_weight: float = 0.09
    sigreg_knots: int = 9
    sigreg_num_proj: int = 32
    alignment_min_samples: int = 16
    alignment_ridge: float = 1e-3
    train_nuisance_strength: float = 0.05
    id_nuisance_strength: float = 0.05
    ood_nuisance_strength: float = 0.30


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _sample_nuisance(
    count: int, image_size: int, strength: float, generator: torch.Generator
) -> torch.Tensor:
    raw = torch.randn(count, 1, image_size, image_size, generator=generator)
    smooth = F.avg_pool2d(raw, kernel_size=5, stride=1, padding=2)
    scale = smooth.flatten(1).std(dim=1, keepdim=True).clamp_min(1e-6)
    smooth = smooth / scale.reshape(-1, 1, 1, 1)
    brightness = torch.randn(count, 1, 1, 1, generator=generator)
    return float(strength) * (0.75 * smooth + 0.25 * brightness)


def _render_positions(
    positions: torch.Tensor, nuisance: torch.Tensor, image_size: int
) -> torch.Tensor:
    axis = torch.arange(image_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    x = positions[:, 0].float().reshape(-1, 1, 1)
    y = positions[:, 1].float().reshape(-1, 1, 1)
    dx = (xx[None] - x).abs()
    dy = (yy[None] - y).abs()
    dx = torch.minimum(dx, image_size - dx)
    dy = torch.minimum(dy, image_size - dy)
    blob = torch.exp(-(dx.square() + dy.square()) / (2.0 * 1.2**2))
    second = torch.roll(blob, shifts=(3, 5), dims=(-2, -1)) * 0.35
    return (blob + second).unsqueeze(1) + nuisance


def _make_one_step_split(
    count: int,
    config: PixelNuisanceConfig,
    generator: torch.Generator,
    nuisance_strength: float,
) -> dict[str, torch.Tensor]:
    positions = torch.randint(0, config.image_size, (count, 2), generator=generator)
    action_ids = torch.arange(count).remainder(ACTION_VECTORS.size(0))
    action_ids = action_ids[torch.randperm(count, generator=generator)]
    actions = ACTION_VECTORS[action_ids]
    nuisance = _sample_nuisance(count, config.image_size, nuisance_strength, generator)
    target_positions = (positions + actions.to(torch.long)) % config.image_size
    return {
        "source": _render_positions(positions, nuisance, config.image_size),
        "target": _render_positions(target_positions, nuisance, config.image_size),
        "actions": actions,
        "positions": positions,
        "nuisance": nuisance,
    }


def _make_rollout_split(
    count: int,
    config: PixelNuisanceConfig,
    generator: torch.Generator,
    nuisance_strength: float,
) -> dict[str, torch.Tensor]:
    positions = torch.randint(0, config.image_size, (count, 2), generator=generator)
    action_ids = torch.randint(
        0, ACTION_VECTORS.size(0), (count, config.rollout_horizon), generator=generator
    )
    actions = ACTION_VECTORS[action_ids]
    nuisance = _sample_nuisance(count, config.image_size, nuisance_strength, generator)
    cumulative = torch.zeros_like(positions)
    frames = [_render_positions(positions, nuisance, config.image_size)]
    for step in range(config.rollout_horizon):
        cumulative = cumulative + actions[:, step].to(torch.long)
        frames.append(
            _render_positions((positions + cumulative) % config.image_size, nuisance, config.image_size)
        )
    return {"frames": torch.stack(frames, dim=1), "actions": actions}


@torch.no_grad()
def _rollout_mse(
    model: PixelWorldModel, split: dict[str, torch.Tensor], horizons: tuple[int, ...]
) -> dict[str, float]:
    frames = split["frames"]
    actions = split["actions"]
    truth = model.encode(frames.reshape(-1, *frames.shape[2:])).reshape(
        frames.size(0), frames.size(1), -1
    )
    current = truth[:, 0]
    output: dict[str, float] = {}
    for step in range(actions.size(1)):
        current = model.predict(current, actions[:, step])
        if step + 1 in horizons:
            output[str(step + 1)] = float((current - truth[:, step + 1]).square().mean())
    return output


@torch.no_grad()
def _one_step_mse(model: PixelWorldModel, split: dict[str, torch.Tensor]) -> float:
    source = model.encode(split["source"])
    target = model.encode(split["target"])
    prediction = model.predict(source, split["actions"])
    return float((prediction - target).square().mean())


@torch.no_grad()
def _tangent_diagnostics(
    model: PixelWorldModel,
    split: dict[str, torch.Tensor],
    config: PixelNuisanceConfig,
    generator: torch.Generator,
) -> dict[str, float]:
    count = split["positions"].size(0)
    nuisance_a = _sample_nuisance(count, config.image_size, config.ood_nuisance_strength, generator)
    nuisance_b = _sample_nuisance(count, config.image_size, config.ood_nuisance_strength, generator)
    same_position_a = _render_positions(split["positions"], nuisance_a, config.image_size)
    same_position_b = _render_positions(split["positions"], nuisance_b, config.image_size)
    source = model.encode(split["source"])
    action_target = model.encode(split["target"])
    nuisance_a_latent = model.encode(same_position_a)
    nuisance_b_latent = model.encode(same_position_b)
    action_delta = action_target - source
    nuisance_delta = nuisance_a_latent - nuisance_b_latent
    prediction_delta = model.predict(source, split["actions"]) - source
    action_norm = action_delta.norm(dim=-1).clamp_min(1e-8)
    nuisance_norm = nuisance_delta.norm(dim=-1).clamp_min(1e-8)
    cosine = torch.nn.functional.cosine_similarity(prediction_delta, action_delta, dim=-1)
    return {
        "nuisance_sensitivity": float(nuisance_delta.square().mean()),
        "action_tangent_energy": float(action_delta.square().mean()),
        "nuisance_to_action_norm_ratio": float((nuisance_norm / action_norm).mean()),
        "predicted_action_tangent_cosine": float(cosine.mean()),
    }


def _fit(
    mode: str,
    alignment_weight: float,
    seed: int,
    config: PixelNuisanceConfig,
) -> tuple[PixelWorldModel, dict[str, torch.Tensor], dict[str, torch.Tensor], int, float]:
    if mode not in {"none", "global", "conditional"}:
        raise ValueError(f"unknown alignment mode: {mode}")
    _seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    train = _make_one_step_split(config.train_samples, config, generator, config.train_nuisance_strength)
    validation = _make_one_step_split(config.validation_samples, config, generator, config.train_nuisance_strength)
    model = PixelWorldModel(config.image_size, config.latent_dim)
    sigreg = SIGReg(knots=config.sigreg_knots, num_proj=config.sigreg_num_proj)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = math.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        source_latent = model.encode(train["source"])
        target_latent = model.encode(train["target"])
        prediction = model.predict(source_latent, train["actions"])
        prediction_loss = (prediction - target_latent).square().mean()
        alignment = operator_alignment_loss(
            source_latent.unsqueeze(0),
            prediction.unsqueeze(0),
            actions=train["actions"].unsqueeze(0),
            mode=mode,
            num_bins=2,
            min_samples=config.alignment_min_samples,
            ridge=config.alignment_ridge,
        )
        loss = prediction_loss + config.sigreg_weight * sigreg(source_latent.unsqueeze(0)) + alignment_weight * alignment
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_source = model.encode(validation["source"])
            val_target = model.encode(validation["target"])
            val_prediction = model.predict(val_source, validation["actions"])
            val_mse = float((val_prediction - val_target).square().mean())
        if val_mse < best_validation:
            best_validation = val_mse
            best_epoch = epoch
            stale = 0
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("validation never produced a checkpoint")
    model.load_state_dict(best_state)
    return model, train, validation, best_epoch, best_validation


def run_one(
    mode: str, alignment_weight: float, seed: int, config: PixelNuisanceConfig
) -> dict[str, Any]:
    model, train, _validation, best_epoch, best_validation = _fit(mode, alignment_weight, seed, config)
    generator = torch.Generator().manual_seed(10_000 + seed)
    id_split = _make_one_step_split(config.test_samples, config, generator, config.id_nuisance_strength)
    ood_split = _make_one_step_split(config.test_samples, config, generator, config.ood_nuisance_strength)
    id_rollout = _make_rollout_split(config.rollout_trajectories, config, generator, config.id_nuisance_strength)
    ood_rollout = _make_rollout_split(config.rollout_trajectories, config, generator, config.ood_nuisance_strength)
    with torch.no_grad():
        source = model.encode(id_split["source"])
        prediction = model.predict(source, id_split["actions"])
        global_diag = operator_alignment_diagnostic(source.unsqueeze(0), prediction.unsqueeze(0), actions=id_split["actions"].unsqueeze(0), mode="global", min_samples=config.alignment_min_samples)
        conditional_diag = operator_alignment_diagnostic(source.unsqueeze(0), prediction.unsqueeze(0), actions=id_split["actions"].unsqueeze(0), mode="conditional", num_bins=2, min_samples=config.alignment_min_samples)
    record: dict[str, Any] = {
        "mode": mode,
        "alignment_weight": float(alignment_weight),
        "sigreg_weight": config.sigreg_weight,
        "seed": seed,
        "device": "cpu",
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation,
        "id_one_step_mse": _one_step_mse(model, id_split),
        "ood_one_step_mse": _one_step_mse(model, ood_split),
        "id_rollout_mse": _rollout_mse(model, id_rollout, (1, 4, 8)),
        "ood_rollout_mse": _rollout_mse(model, ood_rollout, (1, 4, 8)),
        "id_tangent_diagnostics": _tangent_diagnostics(model, id_split, config, generator),
        "ood_tangent_diagnostics": _tangent_diagnostics(model, ood_split, config, generator),
        "diagnostics": {
            "global_commutator": float(global_diag["operator_alignment"]),
            "conditional_commutator": float(conditional_diag["operator_alignment"]),
        },
        "protocol": {
            "transition": "object translation on a torus; background nuisance held fixed across transition",
            "train_nuisance_strength": config.train_nuisance_strength,
            "id_nuisance_strength": config.id_nuisance_strength,
            "ood_nuisance_strength": config.ood_nuisance_strength,
            "learned_encoder_and_predictor": True,
            "synthetic_pixel_nuisance_only": True,
        },
    }
    if not _finite(record):
        raise FloatingPointError(f"non-finite pixel nuisance record: {mode}:{alignment_weight}:{seed}")
    return record


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    paths = {
        "id_one_step_mse": lambda row: row["id_one_step_mse"],
        "ood_one_step_mse": lambda row: row["ood_one_step_mse"],
        "id_rollout_h8_mse": lambda row: row["id_rollout_mse"]["8"],
        "ood_rollout_h8_mse": lambda row: row["ood_rollout_mse"]["8"],
        "ood_nuisance_to_action_ratio": lambda row: row["ood_tangent_diagnostics"]["nuisance_to_action_norm_ratio"],
        "ood_predicted_action_tangent_cosine": lambda row: row["ood_tangent_diagnostics"]["predicted_action_tangent_cosine"],
        "global_commutator": lambda row: row["diagnostics"]["global_commutator"],
        "conditional_commutator": lambda row: row["diagnostics"]["conditional_commutator"],
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(f"{row['mode']}:{row['alignment_weight']}", []).append(row)
    output: dict[str, Any] = {}
    for key, rows in grouped.items():
        output[key] = {"n_seeds": len(rows)}
        for name, extractor in paths.items():
            values = [float(extractor(row)) for row in rows]
            mean = sum(values) / len(values)
            std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else 0.0
            output[key][name] = {"mean": mean, "std": std}
    return output


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def _write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Learned pixel world with visual nuisance OOD",
        "",
        "This is a CPU-only learned pixel-world benchmark. The object undergoes exact torus translations while an independent smooth background nuisance is held fixed across each transition; OOD evaluation increases nuisance amplitude.",
        "",
        "| mode | weight | seeds | ID one-step | OOD one-step | ID rollout h=8 | OOD rollout h=8 | OOD nuisance/action ratio | OOD action-tangent cosine | global commutator | conditional commutator |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, values in report["summary"].items():
        mode, weight = key.split(":", 1)
        def fmt(name: str) -> str:
            return f"{values[name]['mean']:.6g} +/- {values[name]['std']:.2g}"
        lines.append(
            f"| {mode} | {float(weight):.6g} | {values['n_seeds']} | {fmt('id_one_step_mse')} | {fmt('ood_one_step_mse')} | "
            f"{fmt('id_rollout_h8_mse')} | {fmt('ood_rollout_h8_mse')} | {fmt('ood_nuisance_to_action_ratio')} | "
            f"{fmt('ood_predicted_action_tangent_cosine')} | {fmt('global_commutator')} | {fmt('conditional_commutator')} |"
        )
    lines.extend([
        "",
        "Evidence boundary: nuisance/action tangent measurements are diagnostic probes in a synthetic learned pixel world. They do not establish invariant LeWM representations, PushT robustness, online planning, causal control, or GPU performance.",
    ])
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--alignment-weights", type=float, nargs="+", default=[0.01, 0.1])
    args = parser.parse_args()
    config = PixelNuisanceConfig(epochs=args.epochs)
    conditions = [("none", 0.0)]
    conditions.extend(("global", weight) for weight in args.alignment_weights)
    conditions.extend(("conditional", weight) for weight in args.alignment_weights)
    records: list[dict[str, Any]] = []
    start = time.time()
    for mode, weight in conditions:
        for seed in args.seeds:
            print(f"[pixel-nuisance] mode={mode} weight={weight} seed={seed}: measuring visual-nuisance OOD", flush=True)
            records.append(run_one(mode, weight, seed, config))
    report = {
        "protocol": {
            **asdict(config),
            "seeds": args.seeds,
            "alignment_weights": args.alignment_weights,
            "conditions": conditions,
            "same_generated_protocol": True,
            "learned_encoder_and_predictor": True,
            "synthetic_pixel_nuisance_only": True,
            "elapsed_seconds": time.time() - start,
        },
        "records": records,
        "summary": _summary(records),
        "claim_boundary": [
            "nuisance/action tangent ratios are diagnostic probes, not proof of invariance",
            "OOD is a controlled nuisance-amplitude shift in a synthetic pixel world",
            "no nonlinear LeWM, PushT, online planning, causal-control, or GPU claim",
        ],
    }
    if not _finite(report):
        raise ValueError("pixel nuisance report contains non-finite values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
