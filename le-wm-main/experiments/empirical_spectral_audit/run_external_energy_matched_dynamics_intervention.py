"""Energy-matched external rollout controls for dynamics-sensitive subspaces.

The ordinary coefficient-removal intervention can confound subspace identity
with perturbation magnitude: two bases may remove different amounts of latent
energy on the same prediction.  This audit therefore uses the dynamics top-8
removal as an energy reference and rescales covariance/random controls at each
autoregressive step to apply the same squared Euclidean perturbation norm.

This remains a frozen latent rollout diagnostic.  It does not intervene on the
physical environment and does not establish causal control or planning
success.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.empirical_spectral_audit.run_external_dynamics_basis_intervention import (  # noqa: E402
    INTERVENTIONS,
)
from experiments.empirical_spectral_audit.run_lewm_external_pusht_temporal_audit import (  # noqa: E402
    CONTEXT_LENGTH,
    FRAME_STRIDE,
    _bootstrap_clustered,
    _load_video_frames,
    _make_action_blocks,
    _window_starts,
)


ENERGY_MATCHED_INTERVENTIONS = {
    "none": None,
    "dynamics_top8": ("dynamics_basis", False),
    "conditional_dynamics_top8": ("conditional_dynamics_basis", False),
    "covariance_top8": ("covariance_basis", False),
    "covariance_top8_energy_matched_to_dynamics": ("covariance_basis", True),
    "random_top8_energy_matched_to_dynamics": ("random_basis", True),
}


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, int)):
        return bool(np.isfinite(value))
    return True


def _orthonormal_basis(dim: int, seed: int) -> torch.Tensor:
    generator = np.random.default_rng(seed)
    matrix = generator.normal(size=(dim, dim))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return torch.from_numpy((q * signs).astype(np.float32))


def _selected_perturbation(
    prediction: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
    start: int = 0,
    end: int = 8,
) -> torch.Tensor:
    centered = prediction - mean
    coefficients = centered @ basis
    return -(coefficients[..., start:end] @ basis[:, start:end].T)


def _apply_energy_matched_intervention(
    prediction: torch.Tensor,
    mean: torch.Tensor,
    control_basis: torch.Tensor,
    target_basis: torch.Tensor,
    matched: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    control_perturbation = _selected_perturbation(prediction, mean, control_basis)
    control_energy = torch.sum(control_perturbation.square(), dim=-1, keepdim=True)
    if matched:
        target_perturbation = _selected_perturbation(prediction, mean, target_basis)
        target_energy = torch.sum(target_perturbation.square(), dim=-1, keepdim=True)
        scale = torch.sqrt(target_energy / control_energy.clamp_min(1e-12))
    else:
        target_energy = control_energy
        scale = torch.ones_like(control_energy)
    applied_perturbation = scale * control_perturbation
    result = prediction + applied_perturbation
    return result, {
        "reference_energy": float(target_energy.mean().detach().cpu()),
        "control_energy_before_scale": float(control_energy.mean().detach().cpu()),
        "applied_energy": float(applied_perturbation.square().sum(dim=-1).mean().detach().cpu()),
        "scale": float(scale.mean().detach().cpu()),
    }


def _predict_intervened(
    model: torch.nn.Module,
    embedding: torch.Tensor,
    action_embedding: torch.Tensor,
    horizons: list[int],
    mean: torch.Tensor,
    basis_map: dict[str, torch.Tensor],
    intervention: tuple[str, bool] | None,
) -> tuple[dict[int, torch.Tensor], dict[int, dict[str, float]]]:
    context_emb = embedding[:, :CONTEXT_LENGTH].clone()
    context_act = action_embedding[:, :CONTEXT_LENGTH].clone()
    predictions: dict[int, torch.Tensor] = {}
    diagnostics: dict[int, dict[str, float]] = {}
    with torch.inference_mode():
        for step in range(1, max(horizons) + 1):
            prediction = model.predict(context_emb, context_act)[:, -1]
            if intervention is None:
                intervened = prediction
                diagnostic = {
                    "reference_energy": 0.0,
                    "control_energy_before_scale": 0.0,
                    "applied_energy": 0.0,
                    "scale": 1.0,
                }
            else:
                basis_name, matched = intervention
                intervened, diagnostic = _apply_energy_matched_intervention(
                    prediction,
                    mean.to(prediction.device),
                    basis_map[basis_name].to(prediction.device),
                    basis_map["dynamics_basis"].to(prediction.device),
                    matched,
                )
            if step in horizons:
                predictions[step] = intervened.detach().cpu()
                diagnostics[step] = diagnostic
            if step < max(horizons):
                next_action = action_embedding[
                    :, CONTEXT_LENGTH - 1 + step : CONTEXT_LENGTH + step
                ]
                context_emb = torch.cat((context_emb, intervened[:, None]), dim=1)[
                    :, -CONTEXT_LENGTH:
                ]
                context_act = torch.cat((context_act, next_action), dim=1)[
                    :, -CONTEXT_LENGTH:
                ]
    return predictions, diagnostics


def run_report(
    val_root: Path,
    basis_path: Path,
    policy: str,
    output: Path,
    horizons: list[int],
    windows_per_episode: int,
    device_name: str,
    random_seed: int,
) -> dict[str, Any]:
    from experiments.io_utils import load_policy
    from utils import get_img_preprocessor

    with np.load(basis_path, allow_pickle=False) as archive:
        required = {
            "latent_mean",
            "covariance_basis",
            "dynamics_basis",
            "conditional_dynamics_basis",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise KeyError(f"basis artifact missing fields: {missing}")
        mean_np = np.asarray(archive["latent_mean"], dtype=np.float32)
        basis_np = {
            name: np.asarray(archive[name], dtype=np.float32)
            for name in (
                "covariance_basis",
                "dynamics_basis",
                "conditional_dynamics_basis",
            )
        }
    dim = int(mean_np.shape[0])
    if any(value.shape != (dim, dim) for value in basis_np.values()):
        raise ValueError("basis dimensions do not match latent mean")
    basis_np["random_basis"] = _orthonormal_basis(dim, random_seed).numpy()

    output.parent.mkdir(parents=True, exist_ok=True)
    model = load_policy(policy, device=device_name)
    device = torch.device(device_name)
    mean = torch.from_numpy(mean_np)
    basis_map = {name: torch.from_numpy(value) for name, value in basis_np.items()}
    actions = torch.load(
        val_root / "rel_actions.pth", map_location="cpu", weights_only=False
    ).float()
    with (val_root / "seq_lengths.pkl").open("rb") as handle:
        lengths = [int(v) for v in pickle.load(handle)]
    preprocess = get_img_preprocessor("pixels", "pixels", 224)
    records: list[dict[str, Any]] = []
    for episode_id, length in enumerate(lengths):
        starts = _window_starts(
            length,
            context_length=CONTEXT_LENGTH,
            max_horizon=max(horizons),
            windows_per_episode=windows_per_episode,
        )
        print(f"episode={episode_id} length={length} windows={len(starts)}", flush=True)
        for start in starts:
            count = CONTEXT_LENGTH + max(horizons)
            frame_indices = [FRAME_STRIDE * (start + i) for i in range(count)]
            frames = _load_video_frames(
                val_root / "obses" / f"episode_{episode_id:03d}.mp4", frame_indices
            )
            pixels = preprocess({"pixels": frames})["pixels"].unsqueeze(0).to(device)
            action_blocks = _make_action_blocks(
                actions[episode_id], start, count, "observed"
            ).unsqueeze(0).to(device)
            with torch.inference_mode():
                encoded = model.encode({"pixels": pixels, "action": action_blocks})
                for intervention_name, intervention in ENERGY_MATCHED_INTERVENTIONS.items():
                    predictions, diagnostics = _predict_intervened(
                        model,
                        encoded["emb"],
                        encoded["act_emb"],
                        horizons,
                        mean,
                        basis_map,
                        intervention,
                    )
                    for horizon in horizons:
                        target = encoded["emb"][:, CONTEXT_LENGTH - 1 + horizon]
                        mse = float(torch.square(predictions[horizon] - target).mean())
                        row = {
                            "episode_id": int(episode_id),
                            "start_observation": int(start),
                            "intervention": intervention_name,
                            "horizon": int(horizon),
                            "mse": mse,
                            **diagnostics[horizon],
                        }
                        records.append(row)

    summary: dict[str, Any] = {
        "by_intervention_horizon": {},
        "paired_delta_vs_none": {},
        "paired_delta_vs_dynamics": {},
        "energy_matching": {},
    }
    for intervention_name in ENERGY_MATCHED_INTERVENTIONS:
        for horizon in horizons:
            key = f"{intervention_name}/h{horizon}"
            rows = [
                row
                for row in records
                if row["intervention"] == intervention_name and row["horizon"] == horizon
            ]
            values = np.asarray([row["mse"] for row in rows], dtype=np.float64)
            groups = np.asarray([row["episode_id"] for row in rows])
            summary["by_intervention_horizon"][key] = {
                "mse": _bootstrap_clustered(
                    values, groups, repeats=2000, seed=20268003 + horizon
                ),
                "windows": int(values.size),
            }
            if intervention_name != "none":
                baseline = np.asarray(
                    [
                        row["mse"]
                        for row in records
                        if row["intervention"] == "none" and row["horizon"] == horizon
                    ],
                    dtype=np.float64,
                )
                summary["paired_delta_vs_none"][key] = _bootstrap_clustered(
                    values - baseline,
                    groups,
                    repeats=2000,
                    seed=20269003 + horizon,
                )
            if intervention_name not in {"none", "dynamics_top8"}:
                reference = np.asarray(
                    [
                        row["mse"]
                        for row in records
                        if row["intervention"] == "dynamics_top8"
                        and row["horizon"] == horizon
                    ],
                    dtype=np.float64,
                )
                summary["paired_delta_vs_dynamics"][key] = _bootstrap_clustered(
                    values - reference,
                    groups,
                    repeats=2000,
                    seed=20270003 + horizon,
                )
            energy = {
                field: np.asarray([row[field] for row in rows], dtype=np.float64)
                for field in (
                    "reference_energy",
                    "control_energy_before_scale",
                    "applied_energy",
                    "scale",
                )
            }
            summary["energy_matching"][key] = {
                field: {
                    "mean": float(values_.mean()),
                    "min": float(values_.min()),
                    "max": float(values_.max()),
                }
                for field, values_ in energy.items()
            }

    report = {
        "artifact": "external_lewm_energy_matched_dynamics_basis_intervention",
        "status": "frozen_external_energy_matched_subspace_intervention",
        "source": {
            "validation_root": str(val_root),
            "basis_artifact": str(basis_path),
            "episodes": len(lengths),
            "latent_dim": dim,
        },
        "protocol": {
            "context_length": CONTEXT_LENGTH,
            "frame_stride": FRAME_STRIDE,
            "horizons": horizons,
            "windows_per_episode": int(windows_per_episode),
            "interventions": ENERGY_MATCHED_INTERVENTIONS,
            "basis_coordinate_system": "same frozen pusht/lewm policy latent coordinate system",
            "reference_intervention": "dynamics top-8 coefficient removal",
            "matching": "per-step squared Euclidean perturbation energy matched to dynamics top-8",
            "random_basis_seed": int(random_seed),
            "action_normalization": "public release normalization",
            "bootstrap": "episode-cluster paired bootstrap, 2000 replicates",
            "device": device_name,
            "policy": policy,
        },
        "summary": summary,
        "records": records,
        "claim_boundary": [
            "Energy matching removes a perturbation-magnitude confound but does not make the latent intervention causal.",
            "The intervention is in frozen latent coordinates and does not alter the physical environment or actions.",
            "The result does not establish online planning, task success, physical controllability, or retraining improvement.",
            "Cross-basis confidence intervals are interpreted as contrasts of frozen rollout sensitivity, not universal basis rankings.",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    output.with_name("notes.txt").write_text(_notes(report))
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Energy-matched external frozen LeWM dynamics intervention",
        "",
        "The covariance and random controls are rescaled at every autoregressive step so their squared Euclidean latent perturbation matches the dynamics top-8 reference on the same prediction. This removes perturbation magnitude as a direct confound.",
        "",
        "| intervention | h | MSE | delta vs none | delta vs dynamics | applied/reference energy ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, value in report["summary"]["by_intervention_horizon"].items():
        intervention, horizon = key.split("/h")
        mse = value["mse"]
        none_delta = report["summary"]["paired_delta_vs_none"].get(key)
        dynamics_delta = report["summary"]["paired_delta_vs_dynamics"].get(key)
        energy = report["summary"]["energy_matching"][key]
        ratio = energy["applied_energy"]["mean"] / max(energy["reference_energy"]["mean"], 1e-12)
        none_text = "--" if none_delta is None else f"{none_delta['mean']:.6g}"
        dynamics_text = "--" if dynamics_delta is None else f"{dynamics_delta['mean']:.6g}"
        lines.append(
            f"| {intervention} | {horizon} | {mse['mean']:.6g} | "
            f"{none_text} | {dynamics_text} | {ratio:.6f} |"
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        *[f"- {item}" for item in report["claim_boundary"]],
    ])
    path.write_text("\n".join(lines) + "\n")


def _notes(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "2026-10-03: Added per-step energy-matched covariance and random controls against dynamics top-8 removal.",
        "2026-10-03: Matching uses squared Euclidean latent perturbation energy on each autoregressive prediction.",
    ]
    for key in (
        "dynamics_top8/h8",
        "covariance_top8/h8",
        "covariance_top8_energy_matched_to_dynamics/h8",
        "random_top8_energy_matched_to_dynamics/h8",
    ):
        delta = summary["paired_delta_vs_none"].get(key)
        contrast = summary["paired_delta_vs_dynamics"].get(key)
        energy = summary["energy_matching"][key]
        delta_text = "NA" if delta is None else f"{delta['mean']:.6f}"
        contrast_text = "NA" if contrast is None else f"{contrast['mean']:.6f}"
        lines.append(
            f"{key}: delta_vs_none={delta_text} "
            f"delta_vs_dynamics={contrast_text} "
            f"applied/reference_energy={energy['applied_energy']['mean'] / max(energy['reference_energy']['mean'], 1e-12):.6f}."
        )
    lines.append("Boundary: frozen rollout sensitivity only; no causal-control, planning-success, or retraining claim.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--policy", default="pusht/lewm")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--windows-per-episode", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--random-seed", type=int, default=20261003)
    args = parser.parse_args()
    run_report(
        val_root=args.val_root,
        basis_path=args.basis,
        policy=args.policy,
        output=args.output,
        horizons=args.horizons,
        windows_per_episode=args.windows_per_episode,
        device_name=args.device,
        random_seed=args.random_seed,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
