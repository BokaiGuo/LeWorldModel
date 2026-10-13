"""External multi-horizon transport of frozen prediction errors.

Residual-sensitive bases are fitted on a disjoint frozen latent export and
transferred to public PushT validation windows.  For each unmodified model
prediction at h=1/2/4/8, the target-minus-prediction error is projected into
covariance, dynamics, residual, conditional-dynamics, residual-covariance, and
random bases.  Episode-cluster bootstrap intervals and paired contrasts test
whether residual concentration transfers across the source boundary.

This is an error-transport diagnostic.  It is not a causal intervention,
physical-control evaluation, or planning-success result.
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


BASIS_NAMES = (
    "covariance_basis",
    "dynamics_basis",
    "residual_basis",
    "conditional_dynamics_basis",
    "residual_covariance_basis",
    "random_basis",
)
BANDS = {
    "top8": (0, 8),
    "middle8_32": (8, 32),
    "tail128_192": (128, 192),
}


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, int)):
        return bool(np.isfinite(value))
    return True


def _orthonormal_basis(dim: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    matrix = generator.normal(size=(dim, dim))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return (q * signs).astype(np.float32)


def _band_profile(error: np.ndarray, basis: np.ndarray) -> dict[str, float]:
    if error.ndim != 2 or basis.shape != (error.shape[1], error.shape[1]):
        raise ValueError("error and basis dimensions do not match")
    coordinates = error @ basis
    energy = np.square(coordinates)
    total = np.maximum(np.sum(energy, axis=1), 1e-12)
    return {
        band_name: float(np.mean(np.sum(energy[:, start:end], axis=1) / total))
        for band_name, (start, end) in BANDS.items()
    }


def _predict_baseline(
    model: torch.nn.Module,
    embedding: torch.Tensor,
    action_embedding: torch.Tensor,
    horizons: list[int],
) -> dict[int, torch.Tensor]:
    context_emb = embedding[:, :CONTEXT_LENGTH].clone()
    context_act = action_embedding[:, :CONTEXT_LENGTH].clone()
    predictions: dict[int, torch.Tensor] = {}
    with torch.inference_mode():
        for step in range(1, max(horizons) + 1):
            prediction = model.predict(context_emb, context_act)[:, -1]
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
    random_seed: int,
) -> dict[str, Any]:
    from experiments.io_utils import load_policy
    from utils import get_img_preprocessor

    with np.load(basis_path, allow_pickle=False) as archive:
        required = set(BASIS_NAMES[:-1]) | {"latent_mean"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise KeyError(f"basis artifact missing fields: {missing}")
        mean_np = np.asarray(archive["latent_mean"], dtype=np.float32)
        basis_np = {
            name: np.asarray(archive[name], dtype=np.float32) for name in BASIS_NAMES[:-1]
        }
    dim = int(mean_np.shape[0])
    if any(value.shape != (dim, dim) for value in basis_np.values()):
        raise ValueError("basis dimensions do not match latent mean")
    basis_np["random_basis"] = _orthonormal_basis(dim, random_seed)

    output.parent.mkdir(parents=True, exist_ok=True)
    model = load_policy(policy, device=device_name)
    device = torch.device(device_name)
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
                predictions = _predict_baseline(
                    model, encoded["emb"], encoded["act_emb"], horizons
                )
                for horizon in horizons:
                    target = encoded["emb"][:, CONTEXT_LENGTH - 1 + horizon].detach().cpu().numpy()
                    prediction = predictions[horizon].numpy()
                    error = target - prediction
                    for basis_name in BASIS_NAMES:
                        profile = _band_profile(error, basis_np[basis_name])
                        for band_name, share in profile.items():
                            start_band, end_band = BANDS[band_name]
                            coordinates = error @ basis_np[basis_name]
                            band_energy = float(
                                np.mean(np.square(coordinates[:, start_band:end_band]).sum(axis=1))
                            )
                            records.append(
                                {
                                    "episode_id": int(episode_id),
                                    "start_observation": int(start),
                                    "horizon": int(horizon),
                                    "basis": basis_name,
                                    "band": band_name,
                                    "energy_share": float(share),
                                    "band_energy": band_energy,
                                    "total_error_energy": float(np.mean(np.square(error).sum(axis=1))),
                                }
                            )

    summary: dict[str, Any] = {
        "energy_share": {},
        "paired_delta_vs_covariance": {},
        "total_error_energy": {},
    }
    for horizon in horizons:
        for basis_name in BASIS_NAMES:
            rows = [
                row
                for row in records
                if row["horizon"] == horizon and row["basis"] == basis_name
            ]
            groups = np.asarray([row["episode_id"] for row in rows[:: len(BANDS)]])
            summary["total_error_energy"][f"{basis_name}/h{horizon}"] = float(
                np.mean([row["total_error_energy"] for row in rows])
            )
            for band_name in BANDS:
                band_rows = [row for row in rows if row["band"] == band_name]
                values = np.asarray([row["energy_share"] for row in band_rows], dtype=np.float64)
                groups = np.asarray([row["episode_id"] for row in band_rows])
                key = f"{basis_name}/{band_name}/h{horizon}"
                summary["energy_share"][key] = _bootstrap_clustered(
                    values, groups, repeats=2000, seed=20271003 + horizon
                )
                if basis_name != "covariance_basis":
                    baseline = np.asarray(
                        [
                            row["energy_share"]
                            for row in records
                            if row["horizon"] == horizon
                            and row["basis"] == "covariance_basis"
                            and row["band"] == band_name
                        ],
                        dtype=np.float64,
                    )
                    summary["paired_delta_vs_covariance"][key] = _bootstrap_clustered(
                        values - baseline,
                        groups,
                        repeats=2000,
                        seed=20272003 + horizon,
                    )

    report = {
        "artifact": "external_lewm_residual_error_transport",
        "status": "frozen_external_multihorizon_error_transport",
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
            "bases": BASIS_NAMES,
            "bands": BANDS,
            "prediction": "unmodified autoregressive frozen LeWM prediction",
            "error": "encoded future target minus predicted embedding",
            "random_basis_seed": int(random_seed),
            "bootstrap": "episode-cluster paired bootstrap, 2000 replicates",
            "device": device_name,
            "policy": policy,
        },
        "summary": summary,
        "records": records,
        "claim_boundary": [
            "The bases are fitted on a separate frozen latent export and transported without refitting to external public validation windows.",
            "Positive or negative energy-share contrasts describe error geometry, not causal error relocation or model improvement.",
            "The audit uses unmodified rollout and does not establish physical control, online planning, or task success.",
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
        "# External frozen LeWM residual error transport",
        "",
        "Unmodified prediction errors from external public PushT windows are projected into bases fitted on a separate frozen latent export.",
        "",
        "| basis | band | h | mean error-energy share | 95% CI | delta vs covariance | delta CI |",
        "| --- | --- | ---: | ---: | --- | ---: | --- |",
    ]
    for key, value in report["summary"]["energy_share"].items():
        basis, band, horizon = key.split("/")
        delta = report["summary"]["paired_delta_vs_covariance"].get(key)
        lines.append(
            f"| {basis} | {band} | {horizon[1:]} | {value['mean']:.6f} | [{value['ci95'][0]:.6f}, {value['ci95'][1]:.6f}] | "
            f"{'--' if delta is None else f'{delta['mean']:.6f}'} | "
            f"{'--' if delta is None else f'[{delta['ci95'][0]:.6f}, {delta['ci95'][1]:.6f}]'} |"
        )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    path.write_text("\n".join(lines) + "\n")


def _notes(report: dict[str, Any]) -> str:
    key = "residual_basis/top8/h8"
    value = report["summary"]["energy_share"][key]
    covariance = report["summary"]["energy_share"]["covariance_basis/top8/h8"]
    delta = report["summary"]["paired_delta_vs_covariance"][key]
    return (
        "2026-10-03: Transported frozen residual-sensitive bases to external public PushT unmodified rollouts.\n"
        f"At h=8, residual basis top-8 error share={value['mean']:.6f}, covariance top-8 share={covariance['mean']:.6f}, paired delta={delta['mean']:.6f}.\n"
        "Boundary: external error-geometry transport only; no causal control or planning claim.\n"
    )


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
