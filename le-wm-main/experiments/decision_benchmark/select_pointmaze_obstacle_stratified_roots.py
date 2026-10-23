"""Seal an equal-allocation held-out V16 obstacle-challenge root panel.

Selection reads only the generator's static start-cell labels.  It does not
read controller outcomes, oracle rollouts, or report files, which prevents a
post-hoc composition repair from becoming another source of selection bias.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "pointmaze_v16_obstacle_stratified_panel.v1"
MANIFEST_SCHEMA = "pointmaze_geometry_stratified_obstacle_extension.v1"


def select(manifest_path: Path, *, excluded_roots: list[int], per_cell: int, seed: int) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unexpected obstacle-challenge manifest schema")
    if int(per_cell) < 1:
        raise ValueError("per_cell must be positive")
    cells = manifest.get("static_start_cells_qpos", [])
    labels = manifest.get("sampled_start_cell_indices", [])
    base = int(manifest["base_root_count"])
    if not isinstance(cells, list) or len(labels) != int(manifest["new_episode_count"]):
        raise ValueError("malformed obstacle challenge manifest")
    excluded = {int(root) for root in excluded_roots}
    if len(excluded) != len(excluded_roots):
        raise ValueError("excluded roots must be unique")
    by_cell: dict[int, list[int]] = {index: [] for index in range(len(cells))}
    for offset, label in enumerate(labels):
        if not isinstance(label, int) or label not in by_cell:
            raise ValueError("invalid start-cell label")
        root = base + offset
        if root not in excluded:
            by_cell[label].append(root)
    if any(len(roots) < int(per_cell) for roots in by_cell.values()):
        raise ValueError("insufficient unused roots for equal-allocation stratum panel")
    rng = np.random.default_rng(int(seed))
    selected_by_cell = {str(index): sorted(int(value) for value in rng.choice(roots, size=int(per_cell), replace=False)) for index, roots in by_cell.items()}
    roots = sorted(root for values in selected_by_cell.values() for root in values)
    return {
        "schema": SCHEMA,
        "manifest": str(manifest_path.resolve()),
        "selection_contract": "equal allocation from static generator start-cell labels only; controller outcomes, oracle outcomes, and score reports unavailable to selection",
        "seed": int(seed),
        "per_cell": int(per_cell),
        "excluded_roots": sorted(excluded),
        "selected_roots_by_start_cell_index": selected_by_cell,
        "selected_roots": roots,
        "selected_root_count": len(roots),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--excluded-roots", type=int, nargs="*", default=[])
    parser.add_argument("--per-cell", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite sealed stratified-panel artifact")
    result = select(args.manifest, excluded_roots=args.excluded_roots, per_cell=args.per_cell, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(args.output), "selected_root_count": result["selected_root_count"], "selected_roots": result["selected_roots"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
