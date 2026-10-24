"""Merge sealed P1 proposal pools from multiple predeclared damping values.

The pools retain their source manifests and action witnesses.  The merged HDF5
is training input only; the manifest keeps the member-level contracts so the
P1 runner can require that the deployed damping is explicitly covered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SCHEMA = "reacher_d2p_p1_proposal_pair_mixture.v1"
POOL_SCHEMA = "reacher_d2p_p1_proposal_pair_pool.v1"
REQUIRED = {
    "context_frames", "root_frame", "goal_frame", "candidate_successor",
    "zero_successor", "history_raw_actions", "candidate_raw_action",
    "episode_index", "source_seed", "selected_proposal_index",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge(*, output: Path, source_manifests: list[Path]) -> dict[str, Any]:
    if output.exists() or len(source_manifests) < 2:
        raise ValueError("mixture output must be new and contain at least two source pools")
    sources: list[tuple[Path, dict[str, Any], Path]] = []
    seen_dampings: set[float] = set()
    for supplied in source_manifests:
        manifest_path = supplied.resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != POOL_SCHEMA or manifest.get("status") != "completed" or manifest.get("p0_p1_calibration_or_outcome_roots_used") is not False:
            raise ValueError("mixture members must be completed, uncontaminated P1 proposal pools")
        pool = Path(manifest.get("pairs_hdf5", "")).resolve()
        if not pool.is_file() or manifest.get("pairs_hdf5_sha256") != _sha256(pool):
            raise ValueError("mixture member pair pool differs from its manifest")
        contract = manifest.get("proposal_contract")
        if not isinstance(contract, dict):
            raise ValueError("mixture member lacks a proposal contract")
        damping = float(contract.get("damping", float("nan")))
        if not np.isfinite(damping) or damping in seen_dampings:
            raise ValueError("mixture must contain one finite, distinct source per damping")
        seen_dampings.add(damping)
        sources.append((manifest_path, manifest, pool))
    output.mkdir(parents=True)
    destination = output / "pairs.h5"
    total = 0
    try:
        with h5py.File(destination, "x") as target:
            created = False
            for source_index, (_, manifest, pool) in enumerate(sources):
                with h5py.File(pool, "r") as handle:
                    if not REQUIRED <= set(handle):
                        raise ValueError("mixture member is missing required pair datasets")
                    rows = int(handle["episode_index"].shape[0])
                    if rows < 2:
                        raise ValueError("mixture member has too few selected pairs")
                    if any(int(handle[key].shape[0]) != rows for key in REQUIRED):
                        raise ValueError("mixture member datasets do not have a common row count")
                    if not created:
                        for key in sorted(REQUIRED):
                            dataset = handle[key]
                            target.create_dataset(
                                key, shape=(0, *dataset.shape[1:]), maxshape=(None, *dataset.shape[1:]),
                                dtype=dataset.dtype, chunks=dataset.chunks, compression=dataset.compression,
                                compression_opts=dataset.compression_opts,
                            )
                        created = True
                    else:
                        for key in sorted(REQUIRED):
                            source_dataset, target_dataset = handle[key], target[key]
                            if source_dataset.shape[1:] != target_dataset.shape[1:] or source_dataset.dtype != target_dataset.dtype:
                                raise ValueError("mixture members must have identical pair dataset geometry")
                    for start in range(0, rows, 4):
                        stop = min(rows, start + 4)
                        for key in sorted(REQUIRED):
                            values = np.asarray(handle[key][start:stop])
                            if key == "episode_index":
                                values = values.astype(np.int64, copy=False) + source_index * 1_000_000
                            destination_store = target[key]
                            at = int(destination_store.shape[0])
                            destination_store.resize((at + len(values), *destination_store.shape[1:]))
                            destination_store[at:] = values
                    total += rows
    except Exception:
        raise
    members = []
    for manifest_path, manifest, pool in sources:
        members.append({
            "pool_manifest": str(manifest_path), "pool_manifest_sha256": _sha256(manifest_path),
            "pairs_hdf5": str(pool), "pairs_hdf5_sha256": _sha256(pool),
            "seed_range": manifest["seed_range"], "selected_pairs": manifest["selected_pairs"],
            "proposal_contract": manifest["proposal_contract"],
        })
    result = {
        "schema": SCHEMA, "status": "completed", "pairs_hdf5": str(destination.resolve()),
        "pairs_hdf5_sha256": _sha256(destination), "members": members,
        "available_dampings": sorted(seen_dampings), "selected_pairs": total,
        "p0_p1_calibration_or_outcome_roots_used": False,
        "claim_boundary": "Multi-damping P1 proposal-distribution training source. A later P1 runner must verify that its deployed damping is explicitly covered by a member contract.",
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, action="append", required=True)
    args = parser.parse_args()
    result = merge(output=args.output, source_manifests=args.source_manifest)
    print(json.dumps({"status": result["status"], "selected_pairs": result["selected_pairs"], "available_dampings": result["available_dampings"]}, sort_keys=True))


if __name__ == "__main__":
    main()
