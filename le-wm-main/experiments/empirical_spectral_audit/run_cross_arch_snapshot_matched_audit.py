"""Export every real visual-adapter snapshot and run the strict matched audit.

The checkpoint pool is fixed before reading outcomes: three architectures, five
seeds, and the six pre-existing training stages.  Each checkpoint is exported
with the same enriched schema and the same episode-disjoint fit/held-out
protocol.  The downstream matcher fixes stage, latent dimension, action
distribution, prediction quality, and predictor spectrum; it is therefore
allowed to return zero strata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.empirical_spectral_audit.enriched_record_schema import load_payload


DEFAULT_STAGES = (10, 20, 30, 50, 75, 100)
DEFAULT_VARIANTS = ("scalar_legacy", "phase_legacy", "mode_uncertain")
DEFAULT_SEEDS = (8411, 8412, 8413, 8414, 8415)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected(checkpoint: Path, stage: int, split: str) -> tuple[str, str, int, str]:
    variant = checkpoint.parent.parent.parent.name
    seed = checkpoint.parent.parent.name.removeprefix("seed_")
    return f"causal_visual_{variant}", seed, int(stage), split


def _valid(path: Path, expected: tuple[str, str, int, str], rows: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = load_payload(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    architecture, seed, stage, split = expected
    return (
        payload["architecture"] == architecture
        and str(payload["seed"]) == seed
        and int(payload["training_stage"]) == stage
        and payload["split"].get("role") == split
        and len(payload["enriched_records"]) == rows
    )


def _run_export(
    exporter: Path,
    checkpoint: Path,
    manifest: Path,
    output: Path,
    split: str,
    roots: int,
    device: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(exporter),
        "--checkpoint",
        str(checkpoint),
        "--split-manifest",
        str(manifest),
        "--output",
        str(output),
        "--split",
        split,
        "--roots",
        str(roots),
        "--root-offset",
        "0",
        "--device",
        device,
    ]
    print("[export]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _record_pairs(export_dir: Path, record_dir: Path) -> tuple[list[Path], list[Path]]:
    fit = sorted(export_dir.glob("*_stage*_fit_pool256.json"))
    outcome: list[Path] = []
    for fit_path in fit:
        outcome_path = fit_path.with_name(fit_path.name.replace("_fit_pool256", "_outcome"))
        if not outcome_path.exists():
            raise FileNotFoundError(f"missing outcome export for {fit_path}")
        outcome.append(outcome_path)
    if len(fit) != 90:
        raise RuntimeError(f"expected 90 fit exports, found {len(fit)}")
    return fit, outcome


def _inventory(paths: list[Path], output: Path) -> dict[str, Any]:
    rows = []
    for path in paths:
        payload = load_payload(path)
        rows.append(
            {
                "path": str(path),
                "status": "canonical_enriched",
                "schema": payload["record_schema"],
                "architecture": payload["architecture"],
                "environment": payload["environment"],
                "seed": str(payload["seed"]),
                "training_stage": int(payload["training_stage"]),
                "latent_dimension": int(payload["latent_dimension"]),
                "action_distribution": payload["action_distribution"],
                "split": payload["split"].get("role"),
                "row_count": len(payload["enriched_records"]),
                "sha256": _sha256(path),
            }
        )
    result = {
        "protocol": {
            "checkpoint_pool": "three real architectures x five independent seeds x six pre-existing stages",
            "fit_rows": 256,
            "outcome_rows": 64,
            "episode_disjoint": True,
            "same_action_protocol": True,
            "synthetic_conjugation": False,
            "outcome_used_for_export_selection": False,
        },
        "counts": {
            "canonical_exports": len(rows),
            "fit_exports": sum(item["split"] == "fit" for item in rows),
            "outcome_exports": sum(item["split"] == "outcome" for item in rows),
        },
        "exports": rows,
        "claim_boundary": (
            "This inventory establishes a real cross-architecture snapshot pool. "
            "A matched learned-model stratum is claimed only when the strict audit "
            "identifies at least two architectures and two seeds within one exact stage."
        ),
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("results/strong_claim_protocol_v2_20261009/checkpoints"))
    parser.add_argument("--output-root", type=Path, default=Path("results/cross_arch_enriched_snapshots_20261011"))
    parser.add_argument("--stages", nargs="+", type=int, default=list(DEFAULT_STAGES))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--skip-records", action="store_true")
    args = parser.parse_args()

    stages = tuple(sorted(set(args.stages)))
    if stages != DEFAULT_STAGES:
        raise ValueError("the preregistered snapshot pool requires stages 10,20,30,50,75,100")
    exporter = Path(__file__).with_name("export_visual_adapter_enriched.py").resolve()
    export_dir = args.output_root / "visual_adapter_exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    checkpoints: list[tuple[Path, Path, int]] = []
    for variant in DEFAULT_VARIANTS:
        for seed in DEFAULT_SEEDS:
            seed_dir = args.checkpoint_root / variant / f"seed_{seed}"
            manifest = seed_dir / "split_manifest.json"
            if not manifest.exists():
                raise FileNotFoundError(manifest)
            for stage in stages:
                checkpoint = seed_dir / "snapshots" / f"epoch_{stage:04d}.pt"
                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)
                checkpoints.append((checkpoint, manifest, stage))

    manifest_hashes = {_sha256(manifest) for _, manifest, _ in checkpoints}
    if len(manifest_hashes) != 1:
        raise RuntimeError("checkpoint pool does not share one immutable root split")

    for checkpoint, split_manifest, stage in checkpoints:
        variant = checkpoint.parent.parent.parent.name
        seed = checkpoint.parent.parent.name.removeprefix("seed_")
        stem = f"{variant}_seed{seed}_stage{stage:04d}"
        fit_path = export_dir / f"{stem}_fit_pool256.json"
        outcome_path = export_dir / f"{stem}_outcome.json"
        if not _valid(fit_path, _expected(checkpoint, stage, "fit"), 256):
            _run_export(exporter, checkpoint, split_manifest, fit_path, "fit", 256, args.device)
        if not _valid(outcome_path, _expected(checkpoint, stage, "outcome"), 64):
            _run_export(exporter, checkpoint, split_manifest, outcome_path, "outcome", 64, args.device)

    all_exports = sorted(export_dir.glob("*.json"))
    inventory = _inventory(all_exports, args.output_root / "visual_adapter_inventory.json")
    if inventory["counts"] != {"canonical_exports": 180, "fit_exports": 90, "outcome_exports": 90}:
        raise RuntimeError(f"unexpected export inventory: {inventory['counts']}")

    fit_paths, outcome_paths = _record_pairs(export_dir, args.output_root / "visual_adapter_records")
    record_dir = args.output_root / "visual_adapter_records"
    record_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_records:
        builder = Path(__file__).with_name("build_learned_model_selection_records.py").resolve()
        command = [sys.executable, str(builder), "--output", str(record_dir)]
        for fit, outcome in zip(fit_paths, outcome_paths):
            command.extend(("--fit-export", str(fit), "--outcome-export", str(outcome)))
        print("[records] building 90 real snapshot records", flush=True)
        subprocess.run(command, check=True)

        matcher = Path(__file__).with_name("run_matched_learned_transfer_benchmark.py").resolve()
        audit_dir = args.output_root / "matched_audit"
        print("[audit] strict cross-architecture snapshot matching", flush=True)
        subprocess.run(
            [sys.executable, str(matcher), "--record-root", str(record_dir), "--output", str(audit_dir)],
            check=True,
        )

    run_manifest = {
        "schema": "cross_arch_snapshot_matched_audit.v1",
        "checkpoint_count": len(checkpoints),
        "export_count": inventory["counts"],
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "stages": list(stages),
        "variants": list(DEFAULT_VARIANTS),
        "seeds": list(DEFAULT_SEEDS),
        "strict_matching": {"mse_tolerance": 0.05, "spectrum_tolerance": 0.05},
        "outcome_selection_used": False,
    }
    (args.output_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(run_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
