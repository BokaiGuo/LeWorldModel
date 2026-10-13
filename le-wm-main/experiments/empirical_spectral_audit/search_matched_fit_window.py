"""Construct a common training-only fit window for learned-model matching.

The search never reads outcome exports.  It uses only the declared fit pool to
find a single contiguous root window shared by all candidates, then writes
lossless enriched subsets for downstream record construction.  Matching
tolerances are the same as the strict learned-transfer audit.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from experiments.empirical_spectral_audit.build_learned_model_selection_records import _ridge
from experiments.empirical_spectral_audit.enriched_record_schema import load_payload, make_payload


def spectrum(payload: dict, start: int, width: int, latent_indices: list[int]) -> tuple[float, np.ndarray]:
    rows = payload["enriched_records"][start : start + width]
    source, predicted, target = [], [], []
    for row in rows:
        item = row["latent_by_horizon"]["1"]
        source.append(np.asarray(row["source_latent"], dtype=np.float64)[latent_indices])
        predicted.append(np.asarray(item["predicted"], dtype=np.float64)[latent_indices])
        target.append(np.asarray(item["target"], dtype=np.float64)[latent_indices])
    x = np.asarray(source)
    y = np.asarray(predicted)
    centered_x = x - x.mean(0, keepdims=True)
    centered_y = y - y.mean(0, keepdims=True)
    operator = _ridge(centered_x, centered_y, 1e-3)
    values = np.linalg.eigvals(operator)
    return float(np.mean((y - np.asarray(target)) ** 2)), values[np.lexsort((values.imag, values.real))]


def distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    scale = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1e-12)
    return float(np.max(np.abs(left - right)) / scale)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-export", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--mse-tolerance", type=float, default=0.05)
    parser.add_argument("--spectrum-tolerance", type=float, default=0.05)
    parser.add_argument("--latent-indices", type=int, nargs="+", default=None)
    args = parser.parse_args()
    if args.window_size < 8 or args.stride < 1:
        raise ValueError("window-size must be >= 8 and stride must be positive")
    payloads = [(path, load_payload(path)) for path in args.fit_export]
    source_dim = int(payloads[0][1]["latent_dimension"])
    latent_indices = list(range(source_dim)) if args.latent_indices is None else [int(index) for index in args.latent_indices]
    if not latent_indices or min(latent_indices) < 0 or max(latent_indices) >= source_dim:
        raise ValueError("latent-indices must be non-empty and within the exported latent dimension")
    lengths = {len(payload["enriched_records"]) for _, payload in payloads}
    if len(lengths) != 1 or min(lengths) < args.window_size:
        raise ValueError("fit exports must have equal pools at least as large as window-size")
    reference_ids = [row["episode_id"] for row in payloads[0][1]["enriched_records"]]
    for path, payload in payloads[1:]:
        ids = [row["episode_id"] for row in payload["enriched_records"]]
        if ids != reference_ids:
            raise ValueError(f"fit pools are not row-aligned: {path}")

    candidates = []
    for start in range(0, len(reference_ids) - args.window_size + 1, args.stride):
        values = [spectrum(payload, start, args.window_size, latent_indices) for _, payload in payloads]
        mse = np.asarray([item[0] for item in values])
        anchor_index = int(np.argmin(mse))
        distances = np.asarray([distance(values[anchor_index][1], item[1]) for item in values])
        eligible = np.flatnonzero(
            (mse <= float(mse.min()) * (1.0 + args.mse_tolerance))
            & (distances <= args.spectrum_tolerance)
        )
        architectures = sorted({payloads[int(index)][1]["architecture"] for index in eligible})
        seeds = sorted({str(payloads[int(index)][1]["seed"]) for index in eligible})
        candidates.append({
            "start": int(start),
            "window_size": int(args.window_size),
            "mse": mse.tolist(),
            "max_spectrum_distance": float(distances.max()),
            "eligible_indices": eligible.tolist(),
            "eligible_architectures": architectures,
            "eligible_seeds": seeds,
            "identifiable": bool(len(eligible) >= 2 and len(architectures) >= 2 and len(seeds) >= 2),
        })

    candidates.sort(key=lambda row: (
        not row["identifiable"],
        -len(row["eligible_indices"]),
        row["max_spectrum_distance"],
        row["start"],
    ))
    selected = candidates[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_dir = args.output_dir / "selected_fit"
    selected_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for path, payload in payloads:
        subset = copy.deepcopy(payload)
        start = int(selected["start"])
        subset["enriched_records"] = subset["enriched_records"][start : start + args.window_size]
        if latent_indices != list(range(source_dim)):
            subset["latent_dimension"] = len(latent_indices)
            for row in subset["enriched_records"]:
                row["source_latent"] = [row["source_latent"][index] for index in latent_indices]
                for item in row["latent_by_horizon"].values():
                    item["predicted"] = [item["predicted"][index] for index in latent_indices]
                    item["target"] = [item["target"][index] for index in latent_indices]
        subset["split"] = dict(subset["split"])
        subset["split"].update({
            "fit_window_start": start,
            "fit_window_size": int(args.window_size),
            "selection_protocol": "training-only common contiguous window; outcome exports not read",
            "latent_indices": latent_indices,
        })
        subset = make_payload(
            architecture=subset["architecture"], checkpoint=subset["checkpoint"],
            environment=subset["environment"], seed=subset["seed"],
            training_stage=subset["training_stage"], latent_dimension=subset["latent_dimension"],
            action_distribution=subset["action_distribution"], split=subset["split"],
            horizons=subset["horizons"], enriched_records=subset["enriched_records"],
            checkpoint_sha256=subset.get("checkpoint_sha256"),
            dataset_sha256=subset.get("dataset_sha256"), protocol=subset.get("protocol"),
        )
        target = selected_dir / path.name.replace("_fit_pool256", "_fit_selected")
        target.write_text(json.dumps(subset, indent=2, sort_keys=True) + "\n")
        written.append(str(target))

    report = {
        "protocol": {
            "selection_data": "fit exports only",
            "outcome_exports_read": False,
            "mse_tolerance": args.mse_tolerance,
            "spectrum_tolerance": args.spectrum_tolerance,
            "common_window": True,
            "latent_indices": latent_indices,
        },
        "selected": selected,
        "candidate_windows": candidates,
        "selected_fit_exports": written,
        "claim_boundary": "A selected window is a training-only construction; it becomes a matched learned-model stratum only if the downstream disjoint outcome audit is identifiable.",
    }
    (args.output_dir / "fit_window_search.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected": selected, "written": len(written)}, indent=2))


if __name__ == "__main__":
    main()
