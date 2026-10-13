"""Functionally intervene on residual-sensitive frozen latent bases.

Residual cross-covariance and residual-covariance bases are fitted on a
separate frozen latent export and transferred to external public PushT
rollouts.  Raw coefficient removal and per-step energy-matched variants are
reported against dynamics and covariance controls.

This is a frozen latent rollout diagnostic, not a causal physical intervention
or planning-success evaluation.
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

from experiments.empirical_spectral_audit.run_external_energy_matched_dynamics_intervention import (  # noqa: E402
    _apply_energy_matched_intervention,
    _orthonormal_basis,
)
from experiments.empirical_spectral_audit.run_lewm_external_pusht_temporal_audit import (  # noqa: E402
    CONTEXT_LENGTH,
    FRAME_STRIDE,
    _bootstrap_clustered,
    _load_video_frames,
    _make_action_blocks,
    _window_starts,
)


INTERVENTIONS = {
    "none": None,
    "dynamics_top8": ("dynamics_basis", False),
    "covariance_top8": ("covariance_basis", False),
    "residual_top8": ("residual_basis", False),
    "residual_covariance_top8": ("residual_covariance_basis", False),
    "residual_top8_energy_matched_to_dynamics": ("residual_basis", True),
    "residual_covariance_top8_energy_matched_to_dynamics": (
        "residual_covariance_basis",
        True,
    ),
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

    basis_names = (
        "covariance_basis",
        "dynamics_basis",
        "residual_basis",
        "residual_covariance_basis",
    )
    with np.load(basis_path, allow_pickle=False) as archive:
        required = {"latent_mean", *basis_names}
        missing = sorted(required - set(archive.files))
        if missing:
            raise KeyError(f"basis artifact missing fields: {missing}")
        mean_np = np.asarray(archive["latent_mean"], dtype=np.float32)
        basis_np = {
            name: np.asarray(archive[name], dtype=np.float32) for name in basis_names
        }
    dim = int(mean_np.shape[0])
    if any(value.shape != (dim, dim) for value in basis_np.values()):
        raise ValueError("basis dimensions do not match latent mean")
    basis_np["conditional_dynamics_basis"] = basis_np["dynamics_basis"]
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
                for intervention_name, intervention in INTERVENTIONS.items():
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
                        records.append(
                            {
                                "episode_id": int(episode_id),
                                "start_observation": int(start),
                                "intervention": intervention_name,
                                "horizon": int(horizon),
                                "mse": mse,
                                **diagnostics[horizon],
                            }
                        )

    summary: dict[str, Any] = {
        "by_intervention_horizon": {},
        "paired_delta_vs_none": {},
        "paired_delta_vs_dynamics": {},
        "energy_matching": {},
    }
    for intervention_name in INTERVENTIONS:
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
                "mse": _bootstrap_clustered(values, groups, repeats=2000, seed=20273003 + horizon),
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
                    values - baseline, groups, repeats=2000, seed=20274003 + horizon
                )
            if intervention_name not in {"none", "dynamics_top8"}:
                dynamics = np.asarray(
                    [
                        row["mse"]
                        for row in records
                        if row["intervention"] == "dynamics_top8" and row["horizon"] == horizon
                    ],
                    dtype=np.float64,
                )
                summary["paired_delta_vs_dynamics"][key] = _bootstrap_clustered(
                    values - dynamics, groups, repeats=2000, seed=20275003 + horizon
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
        "artifact": "external_lewm_residual_basis_rollout_intervention",
        "status": "frozen_external_residual_basis_intervention",
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
            "interventions": INTERVENTIONS,
            "matching": "matched arms apply per-step squared Euclidean perturbation energy equal to dynamics top-8",
            "random_basis_seed": int(random_seed),
            "bootstrap": "episode-cluster paired bootstrap, 2000 replicates",
            "device": device_name,
            "policy": policy,
        },
        "summary": summary,
        "records": records,
        "claim_boundary": [
            "The residual bases are fitted on a separate frozen latent export and transferred without refitting.",
            "Coefficient removal tests functional sensitivity of frozen rollout coordinates, not causal physical control.",
            "Energy matching removes a perturbation-magnitude confound but does not establish planning success or retraining improvement.",
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
        "# External frozen residual-basis rollout intervention",
        "",
        "Residual-sensitive bases are removed from predicted embeddings, with matched arms rescaled to the dynamics top-8 per-step perturbation energy.",
        "",
        "| intervention | h | MSE | delta vs none | delta vs dynamics | applied/reference energy ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, value in report["summary"]["by_intervention_horizon"].items():
        intervention, horizon = key.split("/h")
        none_delta = report["summary"]["paired_delta_vs_none"].get(key)
        dynamics_delta = report["summary"]["paired_delta_vs_dynamics"].get(key)
        energy = report["summary"]["energy_matching"][key]
        ratio = energy["applied_energy"]["mean"] / max(energy["reference_energy"]["mean"], 1e-12)
        lines.append(
            f"| {intervention} | {horizon} | {value['mse']['mean']:.6g} | "
            f"{'--' if none_delta is None else f'{none_delta['mean']:.6g}'} | "
            f"{'--' if dynamics_delta is None else f'{dynamics_delta['mean']:.6g}'} | {ratio:.6f} |"
        )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    path.write_text("\n".join(lines) + "\n")


def _notes(report: dict[str, Any]) -> str:
    lines = ["2026-10-03: Added residual-basis external rollout interventions."]
    for key in (
        "residual_top8/h8",
        "residual_covariance_top8/h8",
        "residual_top8_energy_matched_to_dynamics/h8",
        "residual_covariance_top8_energy_matched_to_dynamics/h8",
    ):
        none_delta = report["summary"]["paired_delta_vs_none"].get(key)
        dynamics_delta = report["summary"]["paired_delta_vs_dynamics"].get(key)
        energy = report["summary"]["energy_matching"][key]
        lines.append(
            f"{key}: delta_vs_none={'NA' if none_delta is None else f'{none_delta['mean']:.6f}'} "
            f"delta_vs_dynamics={'NA' if dynamics_delta is None else f'{dynamics_delta['mean']:.6f}'} "
            f"energy_ratio={energy['applied_energy']['mean'] / max(energy['reference_energy']['mean'], 1e-12):.6f}."
        )
    lines.append("Boundary: frozen latent rollout sensitivity only.")
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
