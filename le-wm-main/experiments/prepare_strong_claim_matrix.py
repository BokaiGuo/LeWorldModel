"""Freeze the pre-registered experiment matrix for strong world-model claims.

The command writes an immutable manifest plus shell commands; it never executes
training itself.  This separates protocol freezing from result-producing runs.
The matrix is intentionally stringent: it provides a 3-variant x 5-seed x
6-snapshot checkpoint pool, action-support OOD causal tests, and checkpoint
selection records.  A run failing any gate remains a negative result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Variant:
    name: str
    utility_mode: str
    adapter_architecture: str
    latent_dim: int


VARIANTS = (
    Variant("scalar_legacy", "scalar", "legacy", 96),
    Variant("phase_legacy", "phase", "legacy", 96),
    Variant("mode_uncertain", "scalar", "mode_uncertain", 96),
)
SEEDS = (8411, 8412, 8413, 8414, 8415)
DEFAULT_SNAPSHOTS = (10, 20, 30, 50, 75, 100)


def quote(parts: list[str]) -> str:
    return " ".join(shlex.quote(value) for value in parts)


def training_command(root: Path, output: Path, variant: Variant, seed: int, data_seed: int, device: str, snapshots: tuple[int, ...]) -> str:
    return quote([
        sys.executable, "experiments/run_causal_visual_pusht_world_model.py",
        "--output", str(output / "checkpoints" / variant.name / f"seed_{seed}"),
        "--seed", str(seed), "--data-seed", str(data_seed),
        "--train-roots", "1600", "--val-roots", "240", "--epochs", "100",
        "--snapshot-epochs", *(str(epoch) for epoch in snapshots),
        "--latent-dim", str(variant.latent_dim), "--vision-encoder", "mask_moments", "--temporal-context",
        "--utility-mode", variant.utility_mode, "--adapter-architecture", variant.adapter_architecture,
        "--train-action-support", "cardinal", "--train-action-max-magnitude", "0.70",
        "--contact-servo", "--direct-utility", "--skip-planning", "--device", device,
    ])


def selection_command(output: Path, variant: Variant, seed: int, snapshot: int, device: str) -> str:
    checkpoint = output / "checkpoints" / variant.name / f"seed_{seed}" / "snapshots" / f"epoch_{snapshot:04d}.pt"
    return quote([
        sys.executable, "experiments/audit_causal_visual_checkpoint.py",
        "--checkpoint", str(checkpoint), "--checkpoint-label", f"{variant.name}/seed_{seed}/epoch_{snapshot}",
        # Selection is within a fixed architecture-and-seed trajectory.  This
        # prevents utility-head/adapter differences from masquerading as a
        # snapshot-stage geometry effect.
        "--group", f"PushT/in_distribution/{variant.name}/seed_{seed}",
        "--output", str(output / "selection_records" / variant.name / f"seed_{seed}" / f"epoch_{snapshot:04d}"),
        "--diagnostic-roots", "240", "--planning-episodes", "32", "--planning-steps", "100",
        "--planning-horizon", "6", "--cem-samples", "192", "--cem-iterations", "4", "--device", device,
    ])


def causal_command(output: Path, variant: Variant, seed: int, device: str) -> str:
    checkpoint = output / "checkpoints" / variant.name / f"seed_{seed}" / "snapshots" / "epoch_0100.pt"
    return quote([
        sys.executable, "experiments/audit_visual_causal_interventions.py",
        "--checkpoint", str(checkpoint), "--output", str(output / "causal_ood" / variant.name / f"seed_{seed}"),
        "--seed", str(91_000 + seed), "--roots", "400", "--fan-size", "4", "--action-support", "heldout_diagonal",
        "--action-magnitude", "0.90", "--horizons", "1", "2", "4", "8", "--device", device,
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-seed", type=int, default=73001)
    parser.add_argument("--snapshot-epochs", nargs="+", type=int, default=list(DEFAULT_SNAPSHOTS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    snapshots = tuple(sorted(set(args.snapshot_epochs)))
    if len(snapshots) < 5 or len(snapshots) > 8 or any(epoch < 1 or epoch > 100 for epoch in snapshots):
        raise ValueError("snapshot_epochs must contain 5--8 unique epochs in [1, 100]")
    train, selection, causal = [], [], []
    for variant in VARIANTS:
        for seed in SEEDS:
            train.append(training_command(Path.cwd(), args.output, variant, seed, args.data_seed, args.device, snapshots))
            for snapshot in snapshots:
                selection.append(selection_command(args.output, variant, seed, snapshot, args.device))
            causal.append(causal_command(args.output, variant, seed, args.device))
    manifest: dict[str, Any] = {
        "schema": "strong_world_model_protocol.v1",
        "checkpoint_pool": {
            "variants": [asdict(item) for item in VARIANTS], "seeds": list(SEEDS), "snapshots": list(snapshots),
            "total_checkpoints": len(VARIANTS) * len(SEEDS) * len(snapshots),
            "same_observation_action_protocol": True, "fixed_data_seed": args.data_seed,
        },
        "splits": {
            "training": "root-level source manifest created by each training run",
            "selection": "immutable checkpoint diagnostics and fixed held-out planning episodes",
            "test": "separate planning episodes used only in checkpoint record",
            "ood_action": "cardinal training support; diagonal direction and larger magnitude intervention fan",
        },
        "gates": {
            "alignment": "selection regret must beat spectrum-only and prediction-MSE-only with group bootstrap CI",
            "causal": "held-out diagonal action rank and interval coverage must be reported by contact mode and horizon",
            "adapter": "only mode_uncertain vs matched legacy controls; strict success, XY and angle must jointly improve",
            "frozen_transfer": "not scheduled by this PushT-specific matrix; requires a single compatible frozen model and independently defined cross-environment adapter",
            "certificate": "commutator is assessed only as a member of the geometry composite, never a universal certificate",
        },
        "commands": {"training": train, "selection": selection, "causal_ood": causal},
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["sha256_before_execution"] = hashlib.sha256(encoded).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for name, commands in manifest["commands"].items():
        (args.output / f"{name}_commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(commands) + "\n")
    (args.output / "README.md").write_text(
        "# Frozen strong-claim protocol\n\n"
        "This directory freezes commands before any results are generated. Training, selection, and OOD causal evaluation are separate stages. "
        "The protocol does not declare cross-environment success: the current RGB adapter is PushT-specific.\n"
    )
    print(json.dumps({"output": str(args.output), "checkpoint_count": manifest["checkpoint_pool"]["total_checkpoints"], "hash": manifest["sha256_before_execution"]}, indent=2))


if __name__ == "__main__":
    main()
