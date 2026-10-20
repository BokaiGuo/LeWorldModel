"""Export a provenance-checked student feature table from predicted latents.

This is the projection-only GDD-WM control input.  It contains only frozen
predicted terminal latent features and the goal coordinates; it never includes
candidate IDs, teacher ranks, or realized outcome values as features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "gdd_wm_student_feature_table.v0.1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("dataset must be a JSON object with records")
    return payload


def _key(row: dict[str, Any]) -> str:
    return "::".join((str(row["instance_id"]), str(row["teacher_set_id"]), str(row["candidate_id"])))


def build(dataset_path: Path, output: Path, *, goal_anchor: str = "zero") -> dict[str, Any]:
    dataset = _load(dataset_path)
    rows = dataset["records"]
    cache: dict[Path, Any] = {}
    features: list[np.ndarray] = []
    keys: list[str] = []
    for row in rows:
        ref = str(row.get("predicted_terminal_latent_ref", row.get("predicted_latent_ref", "")))
        if "arrays.npz::" not in ref:
            raise ValueError(f"missing predicted latent ref for {_key(row)}")
        source = Path(str(row.get("predicted_latent_source_dataset", row.get("source_candidate_dataset", dataset_path))))
        if not source.is_absolute():
            source = Path.cwd() / source
        arrays_path = source.parent / "arrays.npz"
        if arrays_path not in cache:
            cache[arrays_path] = np.load(arrays_path, allow_pickle=False)
        array_key = ref.split("arrays.npz::", 1)[1]
        arrays = cache[arrays_path]
        if array_key not in arrays.files:
            raise ValueError(f"missing array {array_key}")
        latent = np.asarray(arrays[array_key], dtype=np.float64).reshape(-1)
        goal = np.asarray(row.get("goal_raw", []), dtype=np.float64).reshape(-1)
        if latent.size == 0 or goal.size == 0 or not np.isfinite(latent).all() or not np.isfinite(goal).all():
            raise ValueError(f"non-finite feature for {_key(row)}")
        keys.append(_key(row))
        features.append(np.concatenate([latent, goal]))
    feature_array = np.stack(features)
    if goal_anchor == "zero":
        anchor = np.zeros(feature_array.shape[1], dtype=np.float64)
    elif goal_anchor == "train_teacher_best":
        train_rows = [i for i, row in enumerate(rows) if str(row.get("split")) == "train"]
        if not train_rows:
            raise ValueError("train_teacher_best requires train rows")
        groups: dict[str, list[int]] = {}
        for index in train_rows:
            groups.setdefault(str(rows[index]["instance_id"]), []).append(index)
        best = [max(indices, key=lambda i: (float(rows[i]["teacher_score"]), str(rows[i]["candidate_id"]))) for indices in groups.values()]
        anchor = np.mean(feature_array[best], axis=0)
    else:
        raise ValueError(f"unknown goal_anchor {goal_anchor!r}")
    metadata = {
        "schema": SCHEMA,
        "dataset_sha256": _sha256(dataset_path),
        "key_schema": "instance_id::teacher_set_id::candidate_id",
        "candidate_set_context_input_forbidden": True,
        "teacher_rank_input_forbidden": True,
        "held_out_outcome_input_forbidden": True,
        "dynamics_frozen": True,
        "feature_source": "predicted_terminal_latent_plus_goal_raw",
        "feature_dimension": int(feature_array.shape[1]),
        "goal_anchor": goal_anchor,
        "goal_anchor_source": "train_teacher_best_global" if goal_anchor == "train_teacher_best" else "zero",
        "goal_anchor_outcome_used_for_selection": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        keys=np.asarray(keys),
        features=feature_array,
        goal_features=np.repeat(anchor[None, :], len(feature_array), axis=0),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    report = {
        "schema": "gdd_wm_predicted_latent_feature_table.v0.1",
        "status": "built",
        "summary": {"record_count": len(rows), "feature_dimension": int(feature_array.shape[1]), "goal_anchor": goal_anchor, "output": str(output)},
        "claim_boundary": "projection feature export only; not teacher or planning evidence",
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--goal-anchor", choices=("zero", "train_teacher_best"), default="zero")
    args = parser.parse_args()
    print(json.dumps(build(args.dataset, args.output, goal_anchor=args.goal_anchor), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
