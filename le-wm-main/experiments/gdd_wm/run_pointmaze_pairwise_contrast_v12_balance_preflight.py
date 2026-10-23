"""Host-only fresh-root static admission audit for the V12 balanced C1 test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.decision_benchmark import extend_pointmaze_state_action_pool as extension_tool
from experiments.gdd_wm import preflight_pointmaze_pairwise_contrast_v12 as v12


ROOT = Path(__file__).resolve().parents[2]
BASE_DATASET = ROOT / "data" / "point_maze_v54_pairwise_contrast_v11_extension_20261022"
DATASET = ROOT / "data" / "point_maze_v55_pairwise_contrast_v12_balance_extension_20261022"
OUTPUT = ROOT / "results" / "pointmaze_pairwise_contrast_v12_balance_static_preflight_20261022.json"
ROOT_RANGE = (20_552, 21_384)
SPLIT_COUNTS = {"train": 256, "calibration": 256, "outcome": 256}
EXTENSION_SEED = 202611211
ROOT_SEED, HELD_ACTION_SEED, TRAIN_ACTION_SEED = 202611212, 202611213, 202611214


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _prior_contexts() -> tuple[Path, ...]:
    contexts: list[Path] = []
    for path in sorted((ROOT / "results").rglob("*context*.json")):
        try:
            payload = _read_json(path)
            metadata, records = payload.get("metadata"), payload.get("records")
            if (
                isinstance(metadata, dict) and isinstance(metadata.get("candidate_family"), str)
                and isinstance(records, list)
                and all(isinstance(row, dict) and {"start_episode_id", "candidate_action_hash", "candidate_template_fingerprint"} <= set(row) for row in records)
            ):
                contexts.append(path.resolve())
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
            continue
    if not contexts:
        raise RuntimeError("V12 needs at least one compatible historical context for identity exclusion")
    return tuple(contexts)


def _ensure_extension() -> dict[str, object]:
    expected = {
        "source_root": str(BASE_DATASET.resolve()), "seed": EXTENSION_SEED,
        "new_episode_count": ROOT_RANGE[1] - ROOT_RANGE[0], "base_root_count": ROOT_RANGE[0],
        "extended_root_count": ROOT_RANGE[1], "new_root_id_range": [ROOT_RANGE[0], ROOT_RANGE[1] - 1],
    }
    manifest = DATASET / "extension_manifest.json"
    if DATASET.exists():
        if not manifest.is_file():
            raise RuntimeError("V12 extension directory exists without immutable manifest")
        payload = _read_json(manifest)
        mismatch = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
        if mismatch:
            raise RuntimeError(f"V12 extension contract mismatch: {mismatch}")
        return {"resumed_existing_extension": True, "manifest": str(manifest.resolve())}
    return extension_tool.build_extension(
        BASE_DATASET, DATASET, episode_count=ROOT_RANGE[1] - ROOT_RANGE[0], seed=EXTENSION_SEED,
    )


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to reopen immutable V12 static audit: {OUTPUT}")
    extension = _ensure_extension()
    contexts = _prior_contexts()
    result = v12.preflight(
        dataset_root=DATASET, contexts=list(contexts), root_seed=ROOT_SEED,
        held_action_seed=HELD_ACTION_SEED, train_action_seed=TRAIN_ACTION_SEED,
        root_id_min=ROOT_RANGE[0], root_id_max_exclusive=ROOT_RANGE[1], split_counts=SPLIT_COUNTS,
    )
    result["extension"] = extension
    result["prior_context_count"] = len(contexts)
    result["prior_context_sha256"] = {str(path): _sha256(path) for path in contexts}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "status": result["status"], "checks": result["checks"]}, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed_static_preflight" else 2


if __name__ == "__main__":
    raise SystemExit(main())
