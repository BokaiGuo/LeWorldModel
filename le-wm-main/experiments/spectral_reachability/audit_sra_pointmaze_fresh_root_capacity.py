"""Fail closed when the finite SRA PointMaze RGB root pool is exhausted."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _episode_ids(dataset_root: Path) -> list[int]:
    observations = dataset_root / "obses"
    values = []
    for path in observations.glob("episode_*.pth"):
        try:
            values.append(int(path.stem.removeprefix("episode_")))
        except ValueError:
            continue
    values = sorted(set(values))
    if not values:
        raise ValueError(f"no PointMaze RGB episode files found under {observations}")
    return values


def _root_lists(value: Any, *, label: str = "") -> list[tuple[str, list[int]]]:
    """Collect explicit root/episode lists without guessing from free text."""

    rows: list[tuple[str, list[int]]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{label}.{key}" if label else str(key)
            lowered = str(key).lower()
            if (
                any(token in lowered for token in ("roots", "episodes", "root_ids"))
                and isinstance(child, list)
                and all(isinstance(item, int) for item in child)
            ):
                rows.append((path, [int(item) for item in child]))
            elif lowered in {"root", "root_id", "episode", "episode_id"} and isinstance(child, int):
                rows.append((path, [int(child)]))
            rows.extend(_root_lists(child, label=path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(_root_lists(child, label=f"{label}[{index}]"))
    return rows


def audit(*, dataset_root: Path, receipts: list[Path], required_fresh_roots: int) -> dict[str, Any]:
    if required_fresh_roots < 2:
        raise ValueError("a root-disjoint panel requires at least two fresh roots")
    if not receipts:
        raise ValueError("fresh-root capacity audit requires at least one receipt")
    available = _episode_ids(dataset_root)
    available_set = set(available)
    provenance: list[dict[str, Any]] = []
    consumed: set[int] = set()
    for receipt in sorted({Path(path).resolve() for path in receipts}):
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        groups = {
            key: sorted(set(root for root in roots if root in available_set))
            for key, roots in _root_lists(payload)
        }
        groups = {key: roots for key, roots in groups.items() if roots}
        roots = sorted({root for values in groups.values() for root in values})
        consumed.update(roots)
        provenance.append({
            "receipt": str(receipt),
            "sha256": _sha256(receipt),
            "root_fields": groups,
            "consumed_dataset_roots": roots,
        })
    remaining = sorted(available_set - consumed)
    sufficient = len(remaining) >= required_fresh_roots
    return {
        "schema": "sra_pointmaze_fresh_root_capacity_audit.v1",
        "status": "pass" if sufficient else "blocked_fresh_root_capacity",
        "dataset_root": str(dataset_root.resolve()),
        "available_rgb_root_count": len(available),
        "available_rgb_roots": available,
        "receipt_count": len(provenance),
        "consumed_rgb_root_count": len(consumed),
        "consumed_rgb_roots": sorted(consumed),
        "unconsumed_rgb_roots": remaining,
        "required_fresh_roots": int(required_fresh_roots),
        "gates": {
            "at_least_two_fresh_roots_requested": True,
            "sufficient_unconsumed_rgb_roots": sufficient,
        },
        "receipts": provenance,
        "claim_boundary": (
            "This is a data-provenance capacity audit. A block means a fresh root-disjoint SRA panel "
            "cannot be run from this finite RGB archive; it is not a model, planning, or theory negative."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--required-fresh-roots", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable root-capacity audit: {args.output}")
    result = audit(
        dataset_root=args.dataset_root,
        receipts=list(args.receipt),
        required_fresh_roots=args.required_fresh_roots,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "unconsumed_rgb_roots": result["unconsumed_rgb_roots"]}, sort_keys=True))


if __name__ == "__main__":
    main()
