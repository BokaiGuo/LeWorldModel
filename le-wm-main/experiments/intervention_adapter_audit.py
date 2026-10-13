"""Feature-space adapter intervention audit for LeWorldModel latent tangents.

This is the cheap, no-retraining intervention from the spectral-transfer plan:
build A=(I+lambda Z_a^T Z_a)^-1 from a target factor tangent and measure
whether target tangent energy is suppressed more than non-target factors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.directional_metrics import (
    apply_feature_adapter,
    band_energy_profile,
    compute_eigenspace,
    feature_space_adapter,
    tangent_retention,
    total_energy,
)
from experiments.io_utils import save_json, to_numpy, write_run_status


DEFAULT_NON_TARGETS = ["dynamics", "planning", "predicted_dynamics", "prediction_residual"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents", required=True, help="latents.npz from latent_eigenspace_audit.py")
    parser.add_argument("--output", required=True, help="JSON output path")
    parser.add_argument("--target", default="nuisance", help="target tangent key to suppress")
    parser.add_argument("--non-targets", nargs="+", default=DEFAULT_NON_TARGETS)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=3072)
    return parser.parse_args()


def choose_rows(n: int, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or n <= max_samples:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_samples, replace=False))


def rank_bands(dim: int) -> list[tuple[int, int]]:
    cuts = [0, min(8, dim), min(32, dim), min(128, dim), dim]
    return [(lo, hi) for lo, hi in zip(cuts[:-1], cuts[1:]) if hi > lo]


def load_arrays(path: Path, target: str, non_targets: list[str], max_samples: int, seed: int) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        keys = set(data.files)
        if "prediction_residual" not in keys and {"dynamics", "predicted_dynamics"}.issubset(keys):
            keys.add("prediction_residual")
        if target not in keys:
            raise KeyError(f"target tangent {target!r} not found in {path}; available keys: {sorted(keys)}")
        factors = [target] + [name for name in non_targets if name in keys and name != target]
        latents = data["latents"]
        rows = choose_rows(latents.shape[0], max_samples, seed)
        arrays = {"latents": latents[rows].astype("float32", copy=False)}
        for name in factors:
            if name == "prediction_residual" and name not in data.files:
                value = data["dynamics"][rows] - data["predicted_dynamics"][rows]
            else:
                value = data[name][rows]
            arrays[name] = value.astype("float32", copy=False)
    return arrays


def profile_delta_l1(before: dict[str, float], after: dict[str, float]) -> float:
    keys = set(before) | set(after)
    return float(sum(abs(before.get(k, 0.0) - after.get(k, 0.0)) for k in keys))


def audit(args: argparse.Namespace) -> dict[str, Any]:
    arrays = load_arrays(Path(args.latents), args.target, args.non_targets, args.max_samples, args.seed)
    factors = [name for name in arrays if name != "latents"]
    eig = compute_eigenspace(arrays["latents"])
    bands = rank_bands(arrays["latents"].shape[1])
    baseline_profiles = {
        name: band_energy_profile(eig.eigenvectors, arrays[name], bands) for name in factors
    }
    baseline_energy = {name: float(total_energy(arrays[name]).item()) for name in factors}

    runs: dict[str, Any] = {}
    for strength in args.lambdas:
        adapter = feature_space_adapter(arrays[args.target], strength=strength)
        retention = {name: tangent_retention(adapter, arrays[name]) for name in factors}
        after_profiles = {}
        profile_delta = {}
        for name in factors:
            transformed = to_numpy(apply_feature_adapter(adapter, arrays[name]))
            after_profiles[name] = band_energy_profile(eig.eigenvectors, transformed, bands)
            profile_delta[name] = profile_delta_l1(baseline_profiles[name], after_profiles[name])
        target_ret = max(retention[args.target], 1e-12)
        runs[str(strength)] = {
            "retention": retention,
            "non_target_over_target_retention": {
                name: float(retention[name] / target_ret) for name in factors if name != args.target
            },
            "band_profiles_after": after_profiles,
            "band_profile_delta_l1": profile_delta,
        }

    return {
        "latents": str(args.latents),
        "target": args.target,
        "factors": factors,
        "num_samples": int(arrays["latents"].shape[0]),
        "dim": int(arrays["latents"].shape[1]),
        "bands": [list(band) for band in bands],
        "baseline_energy": baseline_energy,
        "baseline_band_profiles": baseline_profiles,
        "adapter": {
            "formula": "A=(I+lambda Z_target^T Z_target / n)^-1",
            "lambdas": runs,
        },
    }


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    write_run_status(
        output_path.parent,
        "phase3_adapter_intervention",
        "running",
        target=args.target,
        max_samples=args.max_samples,
    )
    result = audit(args)
    save_json(output_path, result)
    write_run_status(
        output_path.parent,
        "phase3_adapter_intervention",
        "completed",
        target=args.target,
        num_samples=result["num_samples"],
        factors=result["factors"],
    )


if __name__ == "__main__":
    main()
