"""Cross-source physical-state probes restricted to frozen latent bands.

This audit transfers bases fitted on a separate frozen LeWM latent export to
external PushT validation trajectories.  For each basis and coordinate band,
it fits a physical-state probe on true latent states from non-held-out
episodes, then compares true and autoregressive latent readouts.  The result
tests what different spectral subspaces make observable; it is not a causal
intervention or control evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.empirical_spectral_audit.run_external_physical_observability_audit import (  # noqa: E402
    STATE_NAMES,
    _decode,
    _fit_probe,
    _normalized_errors,
    _state_targets,
)
from experiments.empirical_spectral_audit.run_external_trajectory_operator_audit import (  # noqa: E402
    CONTEXT_LENGTH,
    _bootstrap_clustered,
    _load_trajectories,
    _normalization_parameters,
)


BANDS = {
    "top8": (0, 8),
    "middle8_32": (8, 32),
    "middle32_128": (32, 128),
    "tail128_192": (128, 192),
}
BASIS_NAMES = (
    "covariance_basis",
    "dynamics_basis",
    "residual_basis",
    "residual_covariance_basis",
    "random_basis",
)


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, int)):
        return bool(np.isfinite(value))
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _random_basis(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return (q * signs).astype(np.float64)


def _load_bases(path: Path, random_seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"latent_mean", *BASIS_NAMES[:-1]}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"basis artifact missing fields: {missing}")
        mean = np.asarray(archive["latent_mean"], dtype=np.float64)
        bases = {name: np.asarray(archive[name], dtype=np.float64) for name in BASIS_NAMES[:-1]}
    dim = int(mean.shape[0])
    if any(value.shape != (dim, dim) for value in bases.values()):
        raise ValueError("basis dimensions do not match latent mean")
    bases["random_basis"] = _random_basis(dim, random_seed)
    for name, basis in bases.items():
        if not np.isfinite(basis).all() or not np.allclose(basis.T @ basis, np.eye(dim), atol=1e-5):
            raise ValueError(f"basis is not finite and orthonormal: {name}")
    return mean, bases


def _band_coordinates(
    latent: np.ndarray, mean: np.ndarray, basis: np.ndarray, band: tuple[int, int]
) -> np.ndarray:
    start, end = band
    return (latent - mean) @ basis[:, start:end]


def _crossfit(
    trajectories: list[dict[str, Any]],
    mean: np.ndarray,
    bases: dict[str, np.ndarray],
    ridge: float,
    bootstrap_repeats: int,
) -> dict[str, Any]:
    all_targets = np.concatenate([row["physical"] for row in trajectories], axis=0)
    scale = np.maximum(np.std(all_targets, axis=0, ddof=1), 1e-6)
    episode_ids = sorted({int(row["episode_id"]) for row in trajectories})
    records: list[dict[str, Any]] = []
    for test_episode in episode_ids:
        train_rows = [row for row in trajectories if int(row["episode_id"]) != test_episode]
        test_rows = [row for row in trajectories if int(row["episode_id"]) == test_episode]
        for basis_name, basis in bases.items():
            for band_name, band in BANDS.items():
                train_true = np.concatenate(
                    [_band_coordinates(row["true"], mean, basis, band) for row in train_rows], axis=0
                )
                train_target = np.concatenate([row["physical"] for row in train_rows], axis=0)
                probe = _fit_probe(train_true, train_target, ridge=ridge)
                for row in test_rows:
                    true_coordinates = _band_coordinates(row["true"], mean, basis, band)
                    predicted_coordinates = _band_coordinates(row["predicted"], mean, basis, band)
                    true_decoded = _decode(probe, true_coordinates)
                    predicted_decoded = _decode(probe, predicted_coordinates)
                    for horizon in (1, 4, 8):
                        true_mse, true_rmse = _normalized_errors(
                            true_decoded[horizon], row["physical"][horizon], scale
                        )
                        predicted_mse, predicted_rmse = _normalized_errors(
                            predicted_decoded[horizon], row["physical"][horizon], scale
                        )
                        records.append(
                            {
                                "episode_id": test_episode,
                                "start_observation": int(row["start_observation"]),
                                "basis": basis_name,
                                "band": band_name,
                                "horizon": horizon,
                                "true_decode_mse": true_mse,
                                "predicted_decode_mse": predicted_mse,
                                "predicted_minus_true_mse": predicted_mse - true_mse,
                                "predicted_minus_true_rmse_by_state": (
                                    predicted_rmse - true_rmse
                                ).tolist(),
                            }
                        )
    summary: dict[str, Any] = {"by_basis_band_horizon": {}}
    for basis_name in bases:
        for band_name in BANDS:
            key_prefix = f"{basis_name}/{band_name}"
            for horizon in (1, 4, 8):
                rows = [
                    row
                    for row in records
                    if row["basis"] == basis_name
                    and row["band"] == band_name
                    and row["horizon"] == horizon
                ]
                groups = np.asarray([row["episode_id"] for row in rows], dtype=np.int64)
                summary["by_basis_band_horizon"][f"{key_prefix}/h{horizon}"] = {
                    "windows": len(rows),
                    "true_decode_mse": _bootstrap_clustered(
                        np.asarray([row["true_decode_mse"] for row in rows]),
                        groups,
                        bootstrap_repeats,
                        20268000 + horizon,
                    ),
                    "predicted_decode_mse": _bootstrap_clustered(
                        np.asarray([row["predicted_decode_mse"] for row in rows]),
                        groups,
                        bootstrap_repeats,
                        20269000 + horizon,
                    ),
                    "predicted_minus_true_mse": _bootstrap_clustered(
                        np.asarray([row["predicted_minus_true_mse"] for row in rows]),
                        groups,
                        bootstrap_repeats,
                        20270000 + horizon,
                    ),
                    "h8_object_x_delta_rmse": None,
                }
                if horizon == 8:
                    summary["by_basis_band_horizon"][f"{key_prefix}/h8"][
                        "h8_object_x_delta_rmse"
                    ] = _bootstrap_clustered(
                        np.asarray([
                            row["predicted_minus_true_rmse_by_state"][2] for row in rows
                        ]),
                        groups,
                        bootstrap_repeats,
                        20271000,
                    )
    return {"records": records, "summary": summary, "state_scale": scale.tolist()}


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["crossfit"]["summary"]["by_basis_band_horizon"]
    lines = [
        "# External physical-state observability by transferred latent band",
        "",
        "Each probe is fitted on true latent states from 20 episodes and evaluated on the held-out episode. Bases are fitted on a separate frozen latent export; values are normalized per physical state dimension.",
        "",
        "| basis | band | h=8 true decode MSE | h=8 predicted decode MSE | h=8 predicted−true MSE | h=8 object-x delta RMSE |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for basis_name in BASES_ORDER:
        for band_name in BANDS:
            values = [summary[f"{basis_name}/{band_name}/h{h}"] for h in (1, 4, 8)]
            fmt = lambda value: f"{value['mean']:.6g} [{value['ci95'][0]:.6g}, {value['ci95'][1]:.6g}]"
            lines.append(
                f"| {basis_name} | {band_name} | {fmt(values[2]['true_decode_mse'])} | "
                f"{fmt(values[2]['predicted_decode_mse'])} | {fmt(values[2]['predicted_minus_true_mse'])} | "
                f"{fmt(values[2]['h8_object_x_delta_rmse'])} |"
            )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    path.write_text("\n".join(lines) + "\n")


BASES_ORDER = list(BASIS_NAMES)


def run_report(
    val_root: Path,
    archive: Path,
    basis_path: Path,
    policy: str,
    output: Path,
    device_name: str = "cpu",
    normalization_mode: str = "public",
    windows_per_episode: int = 2,
    max_horizon: int = 8,
    random_seed: int = 20261003,
    ridge: float = 1e-2,
    bootstrap_repeats: int = 2000,
) -> dict[str, Any]:
    from experiments.io_utils import load_policy

    device = torch.device(device_name)
    model = load_policy(policy, device=str(device))
    actions = torch.load(val_root / "rel_actions.pth", map_location="cpu", weights_only=False).float()
    action_mean, action_std, action_source = _normalization_parameters(actions, normalization_mode)
    trajectories = _load_trajectories(
        model,
        val_root,
        device,
        normalization_mode,
        windows_per_episode,
        max_horizon,
    )
    states = torch.load(val_root / "states.pth", map_location="cpu", weights_only=False).float()
    trajectories = _state_targets(trajectories, states, max_horizon)
    mean, bases = _load_bases(basis_path, random_seed)
    crossfit = _crossfit(
        trajectories,
        mean=mean,
        bases=bases,
        ridge=ridge,
        bootstrap_repeats=bootstrap_repeats,
    )
    report = {
        "artifact": "external_frozen_lewm_physical_band_probe",
        "status": "complete",
        "source": {
            "archive": str(archive),
            "archive_bytes": int(archive.stat().st_size),
            "archive_sha256": _sha256(archive),
            "validation_root": str(val_root),
            "basis_artifact": str(basis_path),
            "basis_sha256": _sha256(basis_path),
            "episodes": int(len({row["episode_id"] for row in trajectories})),
            "windows": int(len(trajectories)),
            "state_fields": list(STATE_NAMES),
        },
        "model": {"policy": policy, "device": str(device)},
        "protocol": {
            "context_length": CONTEXT_LENGTH,
            "max_horizon": int(max_horizon),
            "windows_per_episode": int(windows_per_episode),
            "bases": list(bases),
            "bands": BANDS,
            "probe": "ridge physical-state probe on held-out true latent trajectories",
            "crossfit": "leave-one-episode-out",
            "action_normalization": {
                "mode": normalization_mode,
                "raw_action_divisor": 100.0,
                "mean": action_mean.tolist(),
                "std": action_std.tolist(),
                "source": action_source,
            },
            "state_normalization": "external validation standard deviation per physical field",
            "ridge": float(ridge),
            "random_basis_seed": int(random_seed),
            "bootstrap": "episode-cluster bootstrap",
            "bootstrap_repeats": int(bootstrap_repeats),
        },
        "crossfit": crossfit,
        "claim_boundary": [
            "This tests which transferred frozen latent bands make external physical variables linearly observable.",
            "The bases are not refit on the external validation windows, and the probe is not a policy decoder or environment intervention.",
            "Bands have different dimensions; absolute decode MSE must therefore be compared to the dimension-matched random-basis band, not across bands alone.",
            "Band-specific decode contrasts do not establish causal control, online planning, task success, or a universally useful top eigenspace.",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    output.with_name("notes.txt").write_text(
        "Cross-source physical band probes are held-out observability diagnostics, not causal interventions.\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--policy", default="pusht/lewm")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--action-normalization", choices=["public", "val_zscore", "raw_div100"], default="public")
    parser.add_argument("--windows-per-episode", type=int, default=2)
    parser.add_argument("--max-horizon", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=20261003)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    run_report(
        val_root=args.val_root,
        archive=args.archive,
        basis_path=args.basis,
        policy=args.policy,
        output=args.output,
        device_name=args.device,
        normalization_mode=args.action_normalization,
        windows_per_episode=args.windows_per_episode,
        max_horizon=args.max_horizon,
        random_seed=args.random_seed,
        ridge=args.ridge,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
