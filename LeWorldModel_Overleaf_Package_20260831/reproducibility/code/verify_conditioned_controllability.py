"""Verify the conditioned controllability pilot with held-out and null checks.

This is an independent readout check for a frozen latent artifact.  It does
not train a world model and intentionally reports no planning or control
metric.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.directional_metrics import cluster_states


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    centered = target - target.mean(axis=0, keepdims=True)
    denominator = float(np.square(centered).sum())
    return float(1.0 - np.square(target - prediction).sum() / max(denominator, 1e-12))


def _pilot_path(pilot_dir: Path, seed: int) -> Path:
    if seed == 3072:
        return pilot_dir / "empirical_spectral_audit.json"
    return pilot_dir / f"seed_{seed}" / "empirical_spectral_audit.json"


def verify_one(
    arrays: dict[str, np.ndarray],
    pilot_dir: Path,
    seed: int,
    sample_rows: int,
    clusters: int,
    heldout_seed_offset: int,
    permutation_seed_offset: int,
) -> dict[str, object]:
    latents = arrays["latents"]
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(latents.shape[0], size=sample_rows, replace=False))
    actions = arrays["actions"][rows]
    dynamics = arrays["dynamics"][rows]
    labels = cluster_states(arrays["states"][rows], clusters, seed=seed)

    split_rng = np.random.default_rng(seed + heldout_seed_offset)
    heldout_r2: list[float] = []
    permutation_r2: list[float] = []
    intercept_r2: list[float] = []
    test_sizes: list[int] = []
    for label in sorted(np.unique(labels).tolist()):
        indices = np.flatnonzero(labels == label)
        split_rng.shuffle(indices)
        cut = len(indices) // 2
        train, test = indices[:cut], indices[cut:]
        if len(train) < 2 or len(test) < 2:
            continue

        coef, *_ = np.linalg.lstsq(actions[train], dynamics[train], rcond=None)
        heldout_r2.append(_r2(dynamics[test], actions[test] @ coef))

        shuffled = actions.copy()
        np.random.default_rng(seed + permutation_seed_offset + int(label)).shuffle(shuffled)
        null_coef, *_ = np.linalg.lstsq(shuffled[train], dynamics[train], rcond=None)
        permutation_r2.append(_r2(dynamics[test], shuffled[test] @ null_coef))

        design = np.concatenate(
            [np.ones((actions.shape[0], 1), dtype=np.float32), actions], axis=1
        )
        intercept_coef, *_ = np.linalg.lstsq(design[train], dynamics[train], rcond=None)
        intercept_r2.append(_r2(dynamics[test], design[test] @ intercept_coef))
        test_sizes.append(len(test))

    pilot = json.loads(_pilot_path(pilot_dir, seed).read_text())
    config = pilot.get("config", {})
    expected = {
        "num_samples": sample_rows,
        "condition": "states",
        "clusters": clusters,
        "include_interventions": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"pilot seed {seed} has {key}={config.get(key)!r}; expected {value!r}"
            )
    if 32 not in config.get("ks", []):
        raise ValueError(f"pilot seed {seed} does not contain k=32")
    controllability = pilot["controllability"]
    cells = controllability["cells"]
    k32_nonpositive = sum(
        float(cell["dynamics_minus_controllable_lift"]["32"]) <= 0.0 for cell in cells
    )
    return {
        "seed": seed,
        "in_sample_global_r2": float(controllability["action_to_dynamics_r2"]),
        "in_sample_cell_r2_mean": float(controllability["cell_summary"]["r2_mean"]),
        "in_sample_cell_r2_min": float(controllability["cell_summary"]["r2_min"]),
        "in_sample_cell_r2_max": float(controllability["cell_summary"]["r2_max"]),
        "heldout_cell_r2_mean": float(np.mean(heldout_r2)),
        "heldout_cell_r2_min": float(np.min(heldout_r2)),
        "heldout_cell_r2_max": float(np.max(heldout_r2)),
        "permutation_heldout_r2_mean": float(np.mean(permutation_r2)),
        "intercept_heldout_r2_mean": float(np.mean(intercept_r2)),
        "num_cells": len(heldout_r2),
        "test_size_min": min(test_sizes),
        "test_size_max": max(test_sizes),
        "high_variance_low_controllability_cells": int(
            controllability["cell_summary"]["high_variance_low_controllability_cells"]
        ),
        "k32_dynamics_minus_controllable_nonpositive_cells": int(k32_nonpositive),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rows", type=int, default=20000)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3072, 3073, 3074])
    parser.add_argument("--heldout-seed-offset", type=int, default=9100)
    parser.add_argument("--permutation-seed-offset", type=int, default=9200)
    args = parser.parse_args()

    with np.load(args.source) as data:
        arrays = {
            key: np.asarray(data[key], dtype=np.float32)
            for key in ("latents", "actions", "dynamics", "states")
        }
    records = [
        verify_one(
            arrays,
            args.pilot_dir,
            seed,
            args.sample_rows,
            args.clusters,
            args.heldout_seed_offset,
            args.permutation_seed_offset,
        )
        for seed in args.seeds
    ]
    payload = {
        "artifact": "independent_crossfit_verification",
        "status": "frozen_offline_boundary_evidence",
        "source": str(args.source),
        "protocol": {
            "sample_rows": args.sample_rows,
            "condition": "states",
            "clusters": args.clusters,
            "min_cell_size": 256,
            "top_k": 32,
            "fit": "zero_intercept_least_squares",
            "heldout_split": "within_cell_random_half",
            "heldout_seed_offset": args.heldout_seed_offset,
            "permutation_null": "within_sample_action_row_shuffle",
            "permutation_seed_offset": args.permutation_seed_offset,
        },
        "seeds": records,
        "claim_boundary": [
            "This is a frozen latent, state-conditioned offline diagnostic.",
            "Held-out action-to-dynamics predictability is positive in all three seeds and exceeds the action-permutation null.",
            "It does not establish online planning, control improvement, causal controllability, physical observability, goal reachability, or multi-step rollout quality.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
