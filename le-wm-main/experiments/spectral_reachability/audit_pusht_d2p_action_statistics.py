"""Stream only archived action rows; never fit or deploy a new normalizer.

An archive-statistics match is NOT checkpoint training provenance. No pixels,
states, outcomes, model weights, or neural computation are read by this audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

PUBLIC_MEAN = np.array([-.0087, .0068])
PUBLIC_STD = np.array([.2019, .2002])


def file_stat(path):
    value = path.stat()
    return {"path": str(path.resolve()), "size": value.st_size,
            "mtime_ns": value.st_mtime_ns, "device": value.st_dev, "inode": value.st_ino}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def action_statistics(actions, chunk_rows=65536):
    if chunk_rows <= 0 or len(actions.shape) != 2 or actions.shape[1] != 2:
        raise ValueError("positive bounded chunk size and two-coordinate actions required")
    if np.dtype(actions.dtype).kind != "f":
        raise ValueError("floating action archive required")
    count, dropped, total = 0, 0, actions.shape[0]
    mean, m2 = np.zeros(2), np.zeros(2)
    lower, upper = np.full(2, np.inf), np.full(2, -np.inf)
    hashed = hashlib.sha256()
    for start in range(0, total, chunk_rows):
        raw = np.asarray(actions[start:min(total, start + chunk_rows)])
        hashed.update(np.ascontiguousarray(raw).tobytes())
        # Mirrors get_column_normalizer's row-wise NaN exclusion, not per-axis
        # filtering. Infinity is an invalid input, never silently discarded.
        rows = raw[~np.isnan(raw).any(axis=1)].astype(np.float64)
        dropped += len(raw) - len(rows)
        if not np.isfinite(rows).all():
            raise ValueError("non-NaN training rows contain infinity")
        n = len(rows)
        if not n:
            continue
        delta = rows.mean(axis=0) - mean
        centered = rows - rows.mean(axis=0)
        m2 += np.sum(centered * centered, axis=0) + delta * delta * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
        lower, upper = np.minimum(lower, rows.min(axis=0)), np.maximum(upper, rows.max(axis=0))
    if count < 2 or np.any(m2 <= 0):
        raise ValueError("at least two nondegenerate finite training rows required")
    return {"total_rows": total, "valid_rows": count, "nan_rows_excluded": dropped,
            "mean": mean.tolist(), "sample_std": np.sqrt(m2 / (count - 1)).tolist(),
            "population_std": np.sqrt(m2 / count).tolist(), "minimum": lower.tolist(),
            "maximum": upper.tolist(), "dtype": str(actions.dtype), "shape": list(actions.shape),
            "action_bytes_sha256": hashed.hexdigest(), "hash_scope": "action_field_only_not_full_archive"}


def audit(dataset, config, chunk_rows):
    before = file_stat(dataset)
    with h5py.File(dataset, "r") as handle:
        stats = action_statistics(handle["action"], chunk_rows)
        keys = sorted(handle.keys())  # names only; no other field values
        attributes = {str(k): str(v) for k, v in handle.attrs.items()}
    if before != file_stat(dataset):
        raise ValueError("source archive changed during action-only scan")
    model_config = json.loads(config.read_text())
    mean, std = np.asarray(stats["mean"]), np.asarray(stats["sample_std"])
    # Decimal rounding is an equality diagnostic, not an efficacy-tuned tolerance.
    return {"schema": "pusht_d2p_archive_action_statistics.v1", "status": "completed",
            "source": before, "fields_read": ["action"], "field_names": keys,
            "hdf5_attributes": attributes, "statistics": stats,
            "public_convention": {"mean": PUBLIC_MEAN.tolist(), "std": PUBLIC_STD.tolist()},
            "comparison": {"mean_minus_public": (mean - PUBLIC_MEAN).tolist(),
                           "sample_std_minus_public": (std - PUBLIC_STD).tolist(),
                           "sample_std_over_public": (std / PUBLIC_STD).tolist(),
                           "mean_round4_matches": bool(np.array_equal(np.round(mean, 4), PUBLIC_MEAN)),
                           "std_round4_matches": bool(np.array_equal(np.round(std, 4), PUBLIC_STD))},
            "checkpoint_config": {"path": str(config.resolve()), "sha256": digest(config),
                                  "content": model_config},
            "training_recipe": {"row_filter": "exclude_rows_with_any_nan", "std_ddof": 1,
                                "accumulation": "independent_float64_chunked_moments",
                                "not_bitwise_torch_float32_reduction": True},
            "checkpoint_training_statistics_certified": False, "normalizer_changed": False,
            "P1_admission": False, "physical_future_labels_read": False,
            "neural_computation": False,
            "scope": "Descriptive archived-action statistics only. Neither a match nor mismatch proves which dataset/normalizer produced the official weights. No deployment selection or fitting."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=65536)
    args = parser.parse_args()
    archive = args.output.with_suffix(".implementation")
    if args.output.exists() or archive.exists():
        raise FileExistsError("new immutable statistics receipt required")
    source = Path(__file__).resolve()
    source_hash = digest(source)
    result = audit(args.dataset, args.checkpoint_config, args.chunk_rows)
    if source_hash != digest(source):
        raise ValueError("audit implementation changed during scan")
    archive.mkdir()
    shutil.copyfile(source, archive / "analyzer.py")
    if digest(archive / "analyzer.py") != source_hash:
        raise ValueError("implementation archive differs")
    result["implementation_sha256"] = source_hash
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ("statistics", "comparison")}, indent=2))


if __name__ == "__main__":
    main()
