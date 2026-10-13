"""Autoregressive spectral-band interventions on the external frozen LeWM.

At every autoregressive step, a selected band of the predicted embedding is
projected to the mean in a fixed covariance eigenbasis.  The modified state is
then fed into the next predictor step.  This tests whether spectral bands
matter for multi-step latent rollout rather than merely correlating with a
post-hoc residual.

The basis is frozen from the observed h=1 future-target vectors produced by
the external public PushT audit.  The experiment remains a frozen rollout
diagnostic: it does not claim online control, planner success, or causality in
the environment.
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
    _window_starts,
)


BANDS = {
    "drop_none": None,
    "drop_top8": (0, 8),
    "drop_mid8_32": (8, 32),
    "drop_tail128_192": (128, 192),
}


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, int)):
        return bool(np.isfinite(value))
    return True


def _basis_from_vectors(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path, allow_pickle=False) as archive:
        target = np.asarray(archive["observed/h1/target"], dtype=np.float64)
    mean = target.mean(axis=0)
    centered = target - mean
    covariance = centered.T @ centered / max(target.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    basis = eigenvectors[:, order]
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(basis.astype(np.float32))


def _apply_band_dropout(
    prediction: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
    band: tuple[int, int] | None,
) -> torch.Tensor:
    if band is None:
        return prediction
    centered = prediction - mean
    coefficients = centered @ basis
    coefficients[..., band[0] : band[1]] = 0.0
    return coefficients @ basis.T + mean


def _predict_intervened(
    model: torch.nn.Module,
    embedding: torch.Tensor,
    action_embedding: torch.Tensor,
    horizons: list[int],
    mean: torch.Tensor,
    basis: torch.Tensor,
    intervention: str,
) -> dict[int, torch.Tensor]:
    context_emb = embedding[:, :CONTEXT_LENGTH].clone()
    context_act = action_embedding[:, :CONTEXT_LENGTH].clone()
    predictions: dict[int, torch.Tensor] = {}
    with torch.inference_mode():
        for step in range(1, max(horizons) + 1):
            prediction = model.predict(context_emb, context_act)[:, -1]
            prediction = _apply_band_dropout(
                prediction,
                mean.to(prediction.device),
                basis.to(prediction.device),
                BANDS[intervention],
            )
            if step in horizons:
                predictions[step] = prediction.detach().cpu()
            if step < max(horizons):
                next_action = action_embedding[:, CONTEXT_LENGTH - 1 + step : CONTEXT_LENGTH + step]
                context_emb = torch.cat((context_emb, prediction[:, None]), dim=1)[:, -CONTEXT_LENGTH:]
                context_act = torch.cat((context_act, next_action), dim=1)[:, -CONTEXT_LENGTH:]
    return predictions


def run_report(
    val_root: Path,
    basis_vectors: Path,
    policy: str,
    output: Path,
    horizons: list[int],
    windows_per_episode: int,
    device_name: str,
) -> dict[str, Any]:
    from experiments.io_utils import load_policy
    from utils import get_img_preprocessor

    output.parent.mkdir(parents=True, exist_ok=True)
    model = load_policy(policy, device=device_name)
    device = torch.device(device_name)
    mean, basis = _basis_from_vectors(basis_vectors)
    actions = torch.load(val_root / "rel_actions.pth", map_location="cpu", weights_only=False).float()
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
            action_blocks = _make_action_blocks(actions[episode_id], start, count, "observed")
            action_blocks = action_blocks.unsqueeze(0).to(device)
            with torch.inference_mode():
                encoded = model.encode({"pixels": pixels, "action": action_blocks})
                for intervention in BANDS:
                    predictions = _predict_intervened(
                        model,
                        encoded["emb"],
                        encoded["act_emb"],
                        horizons,
                        mean,
                        basis,
                        intervention,
                    )
                    for horizon in horizons:
                        target = encoded["emb"][:, CONTEXT_LENGTH - 1 + horizon]
                        mse = float(torch.square(predictions[horizon] - target).mean())
                        records.append(
                            {
                                "episode_id": int(episode_id),
                                "start_observation": int(start),
                                "intervention": intervention,
                                "horizon": int(horizon),
                                "mse": mse,
                            }
                        )

    summary: dict[str, Any] = {"by_intervention_horizon": {}, "paired_delta_vs_none": {}}
    for intervention in BANDS:
        for horizon in horizons:
            rows = [
                row
                for row in records
                if row["intervention"] == intervention and row["horizon"] == horizon
            ]
            values = np.asarray([row["mse"] for row in rows], dtype=np.float64)
            groups = np.asarray([row["episode_id"] for row in rows])
            summary["by_intervention_horizon"][f"{intervention}/h{horizon}"] = {
                "mse": _bootstrap_clustered(values, groups, repeats=2000, seed=20263003 + horizon),
                "windows": int(values.size),
            }
            if intervention != "drop_none":
                baseline = np.asarray(
                    [
                        row["mse"]
                        for row in records
                        if row["intervention"] == "drop_none" and row["horizon"] == horizon
                    ],
                    dtype=np.float64,
                )
                summary["paired_delta_vs_none"][f"{intervention}/h{horizon}"] = _bootstrap_clustered(
                    values - baseline, groups, repeats=2000, seed=20264003 + horizon
                )

    report = {
        "artifact": "external_lewm_autoregressive_band_rollout_intervention",
        "status": "frozen_external_rollout_intervention",
        "source": {
            "validation_root": str(val_root),
            "basis_vectors": str(basis_vectors),
            "episodes": len(lengths),
        },
        "protocol": {
            "context_length": CONTEXT_LENGTH,
            "frame_stride": FRAME_STRIDE,
            "horizons": horizons,
            "windows_per_episode": int(windows_per_episode),
            "interventions": BANDS,
            "basis": "observed h=1 future-target covariance eigenbasis from external audit",
            "intervention": "zero selected predicted-embedding coefficients around the frozen target mean before the next predictor step",
            "bootstrap": "episode-cluster paired bootstrap, 2000 replicates",
            "device": device_name,
            "policy": policy,
        },
        "summary": summary,
        "records": records,
        "claim_boundary": [
            "This is a frozen autoregressive latent rollout intervention, not an environment intervention.",
            "The band basis is estimated from the external target vectors and has rank limitation from 42 windows in 192 dimensions.",
            "The experiment does not establish online planning, task success, causal controllability, or retraining improvement.",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# External frozen LeWM autoregressive band intervention",
        "",
        "Each intervention zeros one covariance-relative spectral band of the predicted embedding before feeding it to the next predictor step. Unmodified rollout is the paired baseline.",
        "",
        "| intervention | h | MSE | 95% cluster CI | delta vs none | delta 95% CI |",
        "| --- | ---: | ---: | --- | ---: | --- |",
    ]
    for key, value in report["summary"]["by_intervention_horizon"].items():
        intervention, horizon = key.split("/h")
        mse = value["mse"]
        if intervention == "drop_none":
            lines.append(
                f"| {intervention} | {horizon} | {mse['mean']:.6g} | [{mse['ci95'][0]:.6g}, {mse['ci95'][1]:.6g}] | -- | -- |"
            )
        else:
            delta = report["summary"]["paired_delta_vs_none"][key]
            lines.append(
                f"| {intervention} | {horizon} | {mse['mean']:.6g} | [{mse['ci95'][0]:.6g}, {mse['ci95'][1]:.6g}] | "
                f"{delta['mean']:.6g} | [{delta['ci95'][0]:.6g}, {delta['ci95'][1]:.6g}] |"
            )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--basis-vectors", type=Path, required=True)
    parser.add_argument("--policy", default="pusht/lewm")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--windows-per-episode", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_report(
        val_root=args.val_root,
        basis_vectors=args.basis_vectors,
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
