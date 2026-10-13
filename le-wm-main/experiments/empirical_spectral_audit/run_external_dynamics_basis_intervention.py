"""External rollout intervention using dynamics-sensitive frozen subspaces.

The basis artifact is fit from a disjoint frozen LeWM latent export. It is
then transferred, without refitting, to the public external PushT validation
windows from the same frozen policy coordinate system. At each autoregressive
step, selected coefficients are projected to the training latent mean. A
global covariance basis and raw/action-residualized dynamics bases are
compared under the same intervention.

This is a cross-source frozen latent intervention, not physical environment
control or online planning.
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

from experiments.empirical_spectral_audit.run_lewm_external_pusht_temporal_audit import (  # noqa: E402
    CONTEXT_LENGTH,
    FRAME_STRIDE,
    _bootstrap_clustered,
    _load_video_frames,
    _make_action_blocks,
    _predict_horizons,
    _window_starts,
)


INTERVENTIONS = {
    "none": None,
    "covariance_top8": ("covariance_basis", 0, 8),
    "dynamics_top8": ("dynamics_basis", 0, 8),
    "conditional_dynamics_top8": ("conditional_dynamics_basis", 0, 8),
    "covariance_mid8_32": ("covariance_basis", 8, 32),
    "dynamics_mid8_32": ("dynamics_basis", 8, 32),
    "conditional_dynamics_mid8_32": ("conditional_dynamics_basis", 8, 32),
}


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, int)):
        return bool(np.isfinite(value))
    return True


def _apply_basis_intervention(
    prediction: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
    intervention: tuple[str, int, int] | None,
    basis_map: dict[str, torch.Tensor],
) -> torch.Tensor:
    if intervention is None:
        return prediction
    basis_name, start, end = intervention
    selected_basis = basis_map[basis_name]
    centered = prediction - mean
    coefficients = centered @ selected_basis
    coefficients[..., start:end] = 0.0
    return coefficients @ selected_basis.T + mean


def _predict_intervened(
    model: torch.nn.Module,
    embedding: torch.Tensor,
    action_embedding: torch.Tensor,
    horizons: list[int],
    mean: torch.Tensor,
    basis_map: dict[str, torch.Tensor],
    intervention: tuple[str, int, int] | None,
) -> dict[int, torch.Tensor]:
    context_emb = embedding[:, :CONTEXT_LENGTH].clone()
    context_act = action_embedding[:, :CONTEXT_LENGTH].clone()
    predictions: dict[int, torch.Tensor] = {}
    with torch.inference_mode():
        for step in range(1, max(horizons) + 1):
            prediction = model.predict(context_emb, context_act)[:, -1]
            prediction = _apply_basis_intervention(
                prediction,
                mean.to(prediction.device),
                next(iter(basis_map.values())),
                intervention,
                {name: value.to(prediction.device) for name, value in basis_map.items()},
            )
            if step in horizons:
                predictions[step] = prediction.detach().cpu()
            if step < max(horizons):
                next_action = action_embedding[
                    :, CONTEXT_LENGTH - 1 + step : CONTEXT_LENGTH + step
                ]
                context_emb = torch.cat((context_emb, prediction[:, None]), dim=1)[
                    :, -CONTEXT_LENGTH:
                ]
                context_act = torch.cat((context_act, next_action), dim=1)[
                    :, -CONTEXT_LENGTH:
                ]
    return predictions


def run_report(
    val_root: Path,
    basis_path: Path,
    policy: str,
    output: Path,
    horizons: list[int],
    windows_per_episode: int,
    device_name: str,
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
                    predictions = _predict_intervened(
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
                            }
                        )

    summary: dict[str, Any] = {
        "by_intervention_horizon": {},
        "paired_delta_vs_none": {},
        "paired_delta_vs_covariance": {},
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
                "mse": _bootstrap_clustered(
                    values, groups, repeats=2000, seed=20267003 + horizon
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
                    seed=20268003 + horizon,
                )
    for intervention_name in INTERVENTIONS:
        if intervention_name in {"none", "covariance_top8", "covariance_mid8_32"}:
            continue
        reference = (
            "covariance_top8"
            if intervention_name.endswith("top8")
            else "covariance_mid8_32"
        )
        for horizon in horizons:
            key = f"{intervention_name}/h{horizon}"
            values = np.asarray(
                [
                    row["mse"]
                    for row in records
                    if row["intervention"] == intervention_name
                    and row["horizon"] == horizon
                ],
                dtype=np.float64,
            )
            reference_values = np.asarray(
                [
                    row["mse"]
                    for row in records
                    if row["intervention"] == reference
                    and row["horizon"] == horizon
                ],
                dtype=np.float64,
            )
            groups = np.asarray(
                [
                    row["episode_id"]
                    for row in records
                    if row["intervention"] == intervention_name
                    and row["horizon"] == horizon
                ]
            )
            summary["paired_delta_vs_covariance"][key] = _bootstrap_clustered(
                values - reference_values,
                groups,
                repeats=2000,
                seed=20269003 + horizon,
            )

    report = {
        "artifact": "external_lewm_dynamics_basis_rollout_intervention",
        "status": "frozen_external_dynamics_subspace_intervention",
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
            "interventions": {
                name: value for name, value in INTERVENTIONS.items()
            },
            "basis_coordinate_system": "same frozen pusht/lewm policy latent coordinate system",
            "intervention": "zero selected basis coefficients around the training latent mean at every autoregressive step",
            "action_normalization": "public release normalization",
            "bootstrap": "episode-cluster paired bootstrap, 2000 replicates",
            "device": device_name,
            "policy": policy,
        },
        "summary": summary,
        "records": records,
        "claim_boundary": [
            "This transfers frozen dynamics-sensitive bases to external held-out latent rollouts; it is not end-to-end retraining.",
            "The intervention is in latent coordinates and does not intervene on the physical environment.",
            "The result does not establish causal controllability, online planning, task success, or universal world-model performance.",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# External frozen LeWM dynamics-basis rollout intervention",
        "",
        "A basis fit on a disjoint frozen latent export is transferred to the external public PushT validation windows. Selected coefficients are zeroed around the training latent mean at every autoregressive step; unmodified rollout is the paired baseline.",
        "",
        "| intervention | h | MSE | 95% cluster CI | delta vs none | delta 95% CI |",
        "| --- | ---: | ---: | --- | ---: | --- |",
    ]
    for key, value in report["summary"]["by_intervention_horizon"].items():
        intervention, horizon = key.split("/h")
        mse = value["mse"]
        if intervention == "none":
            lines.append(
                f"| {intervention} | {horizon} | {mse['mean']:.6g} | [{mse['ci95'][0]:.6g}, {mse['ci95'][1]:.6g}] | -- | -- |"
            )
        else:
            delta = report["summary"]["paired_delta_vs_none"][key]
            lines.append(
                f"| {intervention} | {horizon} | {mse['mean']:.6g} | [{mse['ci95'][0]:.6g}, {mse['ci95'][1]:.6g}] | "
                f"{delta['mean']:.6g} | [{delta['ci95'][0]:.6g}, {delta['ci95'][1]:.6g}] |"
            )
    lines.extend(
        [
            "",
            "## Paired contrast against same-band covariance basis",
            "",
            "| intervention | h | delta vs covariance basis | 95% cluster CI |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for key, value in report["summary"]["paired_delta_vs_covariance"].items():
        intervention, horizon = key.split("/h")
        lines.append(
            f"| {intervention} | {horizon} | {value['mean']:.6g} | "
            f"[{value['ci95'][0]:.6g}, {value['ci95'][1]:.6g}] |"
        )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--policy", default="pusht/lewm")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--windows-per-episode", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_report(
        val_root=args.val_root,
        basis_path=args.basis,
        policy=args.policy,
        output=args.output,
        horizons=args.horizons,
        windows_per_episode=args.windows_per_episode,
        device_name=args.device,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
