"""Train a tiny pixel-to-latent action-conditioned world model.

This benchmark is the next bridge after the linear action-conditioned sweep:
the encoder and predictor are trained jointly from rendered pixels with the
LeWM-style next-embedding objective and SIGReg.  The world is a small torus,
so translations have exact action composition and long-horizon ground truth.

The artifact is deliberately scoped.  It is synthetic pixel-world evidence,
not an end-to-end LeWM or PushT result.  It tests whether an alignment
intervention remains measurable after introducing a learned pixel encoder; it
does not establish planning, causal controllability, or transfer to a real
environment.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

try:
    from module import SIGReg
except ModuleNotFoundError:  # direct ``python experiments/.../script.py`` invocation
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from module import SIGReg
try:
    from experiments.empirical_spectral_audit.regularizers import (
        operator_alignment_diagnostic,
        operator_alignment_loss,
    )
except ModuleNotFoundError:  # standalone reproducibility-package invocation
    from regularizers import operator_alignment_diagnostic, operator_alignment_loss


# The different norms intentionally create two action-local groups for the
# existing magnitude-binned conditional regularizer: horizontal actions move
# one pixel and vertical actions move two pixels on the torus.
ACTION_VECTORS = torch.tensor(
    [[0.0, 1.0], [2.0, 0.0], [0.0, -1.0], [-2.0, 0.0]], dtype=torch.float32
)


@dataclass(frozen=True)
class PixelBenchmarkConfig:
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
    sigreg_weight: float = 0.01
    sigreg_knots: int = 9
    sigreg_num_proj: int = 32
    alignment_bins: int = 2
    alignment_min_samples: int = 16
    alignment_ridge: float = 1e-3


class PixelWorldModel(nn.Module):
    """Small encoder/predictor pair with the same latent prediction interface as LeWM."""

    def __init__(self, image_size: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(16 * image_size * image_size, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + 2, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
        )

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixels.float())

    def predict(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predictor(torch.cat((latent, action.float()), dim=-1))


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _render_base_images(
    count: int, image_size: int, generator: torch.Generator
) -> torch.Tensor:
    """Render periodic Gaussian blobs at random integer positions."""

    positions = torch.randint(0, image_size, (count, 2), generator=generator)
    axis = torch.arange(image_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    dx = (xx[None] - positions[:, 0, None, None]).abs()
    dy = (yy[None] - positions[:, 1, None, None]).abs()
    dx = torch.minimum(dx, image_size - dx)
    dy = torch.minimum(dy, image_size - dy)
    blob = torch.exp(-(dx.square() + dy.square()) / (2.0 * 1.2**2))
    # A weak deterministic second blob prevents the encoder from solving the
    # task only by measuring one symmetric peak while retaining exact shifts.
    second = torch.roll(blob, shifts=(3, 5), dims=(-2, -1)) * 0.35
    return (blob + second).unsqueeze(1)


def _roll_images(images: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """Apply per-sample integer torus translations."""

    return torch.stack(
        [
            torch.roll(image, shifts=(int(shift[0]), int(shift[1])), dims=(-2, -1))
            for image, shift in zip(images, shifts)
        ]
    )


def _make_one_step_split(
    count: int, config: PixelBenchmarkConfig, generator: torch.Generator
) -> dict[str, torch.Tensor]:
    source = _render_base_images(count, config.image_size, generator)
    # Balanced action IDs make the median-norm bin a genuine two-way split
    # even for finite samples; a random unbalanced draw could put the quantile
    # exactly on the larger norm and silently collapse the conditional mode.
    action_ids = torch.arange(count).remainder(ACTION_VECTORS.size(0))
    action_ids = action_ids[torch.randperm(count, generator=generator)]
    # Make the two action-norm groups genuinely heterogeneous in the training
    # distribution.  This is a controlled conditional-geometry benchmark, not
    # an assumption that action and state are independent in every environment.
    vertical = action_ids.remainder(2).bool()
    if bool(vertical.any()):
        source[vertical] = torch.roll(source[vertical], shifts=(0, 4), dims=(-2, -1))
    actions = ACTION_VECTORS[action_ids]
    target = _roll_images(source, actions)
    return {"source": source, "target": target, "actions": actions, "action_ids": action_ids}


def _make_rollout_split(
    count: int, config: PixelBenchmarkConfig, generator: torch.Generator
) -> dict[str, torch.Tensor]:
    base = _render_base_images(count, config.image_size, generator)
    action_ids = torch.randint(
        0, ACTION_VECTORS.size(0),
        (count, config.rollout_horizon),
        generator=generator,
    )
    actions = ACTION_VECTORS[action_ids]
    cumulative = torch.zeros(count, 2)
    frames = [base]
    for step in range(config.rollout_horizon):
        cumulative = cumulative + actions[:, step]
        frames.append(_roll_images(base, cumulative))
    return {
        "frames": torch.stack(frames, dim=1),
        "actions": actions,
        "action_ids": action_ids,
    }


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
    result: dict[str, float] = {}
    for step in range(actions.size(1)):
        current = model.predict(current, actions[:, step])
        horizon = step + 1
        if horizon in horizons:
            result[str(horizon)] = float((current - truth[:, horizon]).square().mean())
    return result


@torch.no_grad()
def _composition_errors(
    model: PixelWorldModel,
    base: torch.Tensor,
    image_size: int,
) -> dict[str, float]:
    initial = model.encode(base)
    errors: dict[str, float] = {}
    for first in range(ACTION_VECTORS.size(0)):
        first_action = ACTION_VECTORS[first].expand(base.size(0), -1)
        after_first = model.predict(initial, first_action)
        for second in range(ACTION_VECTORS.size(0)):
            second_action = ACTION_VECTORS[second].expand(base.size(0), -1)
            learned = model.predict(after_first, second_action)
            total_shift = ACTION_VECTORS[first] + ACTION_VECTORS[second]
            target_image = _roll_images(base, total_shift.expand(base.size(0), -1))
            target = model.encode(target_image)
            denominator = target.square().mean().sqrt().clamp_min(1e-8)
            errors[f"{first}->{second}"] = float(
                (learned - target).square().mean().sqrt() / denominator
            )
    return errors


def _stable_rank(latent: torch.Tensor) -> float:
    centered = latent - latent.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(latent.size(0) - 1, 1)
    trace = covariance.trace()
    return float((trace.square() / covariance.square().sum().clamp_min(1e-12)).detach())


def _diagnostics(
    model: PixelWorldModel, split: dict[str, torch.Tensor]
) -> dict[str, object]:
    with torch.no_grad():
        source = model.encode(split["source"])
        prediction = model.predict(source, split["actions"])
    source_3d = source.unsqueeze(0)
    prediction_3d = prediction.unsqueeze(0)
    actions_3d = split["actions"].unsqueeze(0)
    global_value = operator_alignment_diagnostic(
        source_3d, prediction_3d, actions=actions_3d, mode="global", min_samples=16
    )
    conditional_value = operator_alignment_diagnostic(
        source_3d,
        prediction_3d,
        actions=actions_3d,
        mode="conditional",
        num_bins=2,
        min_samples=16,
    )
    return {
        "global_commutator": global_value["operator_alignment"],
        "conditional_commutator": conditional_value["operator_alignment"],
        "stable_rank": _stable_rank(source),
    }


def run_one(
    mode: str,
    alignment_weight: float,
    seed: int,
    config: PixelBenchmarkConfig,
) -> dict[str, object]:
    _seed_everything(seed)
    data_generator = torch.Generator().manual_seed(seed)
    train = _make_one_step_split(config.train_samples, config, data_generator)
    validation = _make_one_step_split(config.validation_samples, config, data_generator)
    test = _make_one_step_split(config.test_samples, config, data_generator)
    rollout = _make_rollout_split(config.rollout_trajectories, config, data_generator)
    model = PixelWorldModel(config.image_size, config.latent_dim)
    sigreg = SIGReg(knots=config.sigreg_knots, num_proj=config.sigreg_num_proj)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float]] = []
    best_validation_mse = math.inf
    best_epoch = 0
    stale_epochs = 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        source_latent = model.encode(train["source"])
        target_latent = model.encode(train["target"])
        prediction = model.predict(source_latent, train["actions"])
        prediction_loss = (prediction - target_latent).square().mean()
        alignment_loss = operator_alignment_loss(
            source_latent.unsqueeze(0),
            prediction.unsqueeze(0),
            actions=train["actions"].unsqueeze(0),
            mode=mode,
            num_bins=config.alignment_bins,
            min_samples=config.alignment_min_samples,
            ridge=config.alignment_ridge,
        )
        sigreg_loss = sigreg(source_latent.unsqueeze(0))
        loss = prediction_loss + config.sigreg_weight * sigreg_loss + alignment_weight * alignment_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_source = model.encode(validation["source"])
            val_target = model.encode(validation["target"])
            val_prediction = model.predict(val_source, validation["actions"])
            validation_mse = float((val_prediction - val_target).square().mean())
        history.append(
            {
                "epoch": float(epoch),
                "prediction_loss": float(prediction_loss.detach()),
                "sigreg_loss": float(sigreg_loss.detach()),
                "alignment_loss": float(alignment_loss.detach()),
                "validation_mse": validation_mse,
            }
        )
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: parameter.detach().clone()
                for name, parameter in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("validation never produced a checkpoint")
    model.load_state_dict(best_state)
    with torch.no_grad():
        test_source = model.encode(test["source"])
        test_target = model.encode(test["target"])
        test_prediction = model.predict(test_source, test["actions"])
        one_step_mse = float((test_prediction - test_target).square().mean())
        composition = _composition_errors(model, rollout["frames"][:, 0], config.image_size)
    return {
        "mode": mode,
        "alignment_weight": alignment_weight,
        "seed": seed,
        "device": "cpu",
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_validation_mse": best_validation_mse,
        "test_one_step_mse": one_step_mse,
        "rollout_mse": _rollout_mse(model, rollout, (1, 2, 4, 8)),
        "action_composition_error": composition,
        "action_composition_error_mean": sum(composition.values()) / len(composition),
        "diagnostics": _diagnostics(model, test),
        "history": history,
    }


def _summary(records: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(f"{record['mode']}:{record['alignment_weight']}", []).append(record)
    output: dict[str, object] = {}
    for key, rows in grouped.items():
        one_step = [float(row["test_one_step_mse"]) for row in rows]
        horizons = {
            h: [float(row["rollout_mse"][h]) for row in rows]  # type: ignore[index]
            for h in rows[0]["rollout_mse"]  # type: ignore[union-attr]
        }
        output[key] = {
            "n_seeds": len(rows),
            "best_epoch_mean": sum(float(row["best_epoch"]) for row in rows) / len(rows),
            "epochs_run_mean": sum(float(row["epochs_run"]) for row in rows) / len(rows),
            "best_validation_mse_mean": sum(float(row["best_validation_mse"]) for row in rows) / len(rows),
            "test_one_step_mse_mean": sum(one_step) / len(one_step),
            "test_one_step_mse_std": float(torch.tensor(one_step).std(unbiased=False)),
            "rollout_mse_mean": {h: sum(values) / len(values) for h, values in horizons.items()},
            "action_composition_error_mean": sum(float(row["action_composition_error_mean"]) for row in rows) / len(rows),
            "global_commutator_mean": sum(float(row["diagnostics"]["global_commutator"]) for row in rows) / len(rows),  # type: ignore[index]
            "conditional_commutator_mean": sum(float(row["diagnostics"]["conditional_commutator"]) for row in rows) / len(rows),  # type: ignore[index]
            "stable_rank_mean": sum(float(row["diagnostics"]["stable_rank"]) for row in rows) / len(rows),  # type: ignore[index]
        }
    return output


def _write_markdown(report: dict[str, object], output: Path) -> None:
    lines = [
        "# Pixel-to-latent action-conditioned world-model benchmark",
        "",
        "This is a synthetic pixel-world training experiment, not a learned LeWM or PushT result.",
        "The encoder and predictor are trained jointly with next-embedding prediction plus SIGReg.",
        "",
        "## Protocol",
        "",
        f"- device: `{report['protocol']['device']}`; seeds: `{report['protocol']['seeds']}`; epochs: `{report['protocol']['epochs']}`",
        f"- modes: `{report['protocol']['modes']}`; weights: `{report['protocol']['weights']}`",
        "- the torus has exact translation composition; horizontal actions move one pixel and vertical actions move two pixels",
        "- the one-step training split uses a fixed action-group-specific scene offset to create heterogeneous local geometry",
        "- rollout uses the sampled action sequence and compares predicted latent states with encoded future frames",
        "- all modes reuse identical generated splits, initialization seeds, and evaluation code",
        "- each run restores the checkpoint with the lowest held-out validation MSE; patience and actual epochs are recorded per seed",
        "",
        "## Aggregate results",
        "",
        "| intervention | weight | seeds | one-step MSE | rollout h=8 MSE | composition error | global commutator | conditional commutator | stable rank |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, values in report["summary"].items():
        mode, weight = key.split(":", 1)
        lines.append(
            f"| {mode} | {float(weight):.6g} | {values['n_seeds']} | "
            f"{values['test_one_step_mse_mean']:.6g} | {values['rollout_mse_mean']['8']:.6g} | "
            f"{values['action_composition_error_mean']:.6g} | {values['global_commutator_mean']:.6g} | "
            f"{values['conditional_commutator_mean']:.6g} | {values['stable_rank_mean']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This artifact tests whether the alignment intervention remains measurable after adding a learned pixel encoder. It does not establish nonlinear LeWM gains, PushT performance, causal controllability, online planning, or GPU performance.",
        ]
    )
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--weights", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    args = parser.parse_args()
    config = PixelBenchmarkConfig(epochs=args.epochs)
    modes = ["none", "global", "conditional"]
    records: list[dict[str, object]] = []
    start = time.time()
    for mode in modes:
        weights = [0.0] if mode == "none" else args.weights
        for weight in weights:
            for seed in args.seeds:
                print(
                    f"[pixel-benchmark] mode={mode} weight={weight} seed={seed}: "
                    "measuring learned encoder/predictor consequences"
                )
                records.append(run_one(mode, weight, seed, config))
    report = {
        "protocol": {
            **asdict(config),
            "device": args.device,
            "seeds": args.seeds,
            "modes": modes,
            "weights": args.weights,
            "action_vectors": ACTION_VECTORS.tolist(),
            "conditional_scene_offset": [0, 4],
            "rollout_action_schedule": "random_sequence",
            "learned_encoder_and_predictor": True,
            "synthetic_pixel_only": True,
            "elapsed_seconds": time.time() - start,
        },
        "records": records,
        "summary": _summary(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, args.output.with_suffix(".md"))
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
