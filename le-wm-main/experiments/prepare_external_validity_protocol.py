"""Freeze the external-validity experiments required beyond PushT ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any


def command(parts: list[str]) -> str:
    return shlex.join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {manifest_path}")
    args.output.mkdir(parents=True, exist_ok=False)
    prediction = command([
        sys.executable, "experiments/audit_frozen_cross_environment_prediction.py",
        "--policy", "pusht/lewm",
        "--environment", "swm/PushT-v1",
        "--environment", "swm/TwoRoom-v1",
        "--environment", "swm/SimplePointMaze-v0",
        "--output", str(args.output / "frozen_pusht_checkpoint_prediction"),
        "--calibration-episodes", "12", "--test-episodes", "24", "--blocks", "12",
        "--action-block", "5", "--horizons", "1", "2", "4", "--seed", "20261009", "--device", args.device,
    ])
    readiness = command([
        sys.executable, "experiments/audit_cross_environment_readiness.py",
        "--output", str(args.output / "frozen_planning_readiness"),
    ])
    manifest: dict[str, Any] = {
        "schema": "external_validity_protocol.v1",
        "scope": {
            "source_checkpoint": "pusht/lewm",
            "checkpoint_frozen": True,
            "forbidden": [
                "target-environment adapter training",
                "privileged state as runtime model input",
                "simulator successor verification for actions",
                "replacing a failed target-domain result with an environment-specific checkpoint",
            ],
            "domains": {
                "swm/PushT-v1": "contact manipulation source-domain reference",
                "swm/TwoRoom-v1": "continuous navigation with room topology",
                "swm/SimplePointMaze-v0": "non-contact planar navigation with a distinct action scale",
            },
        },
        "splits": {
            "action_calibration": "12 simulator episodes, disjoint seeds; used only for action z-score parameters",
            "test": "24 simulator episodes, disjoint seeds; never used to fit weights or select checkpoint",
            "planning": "separate gate; requires legal common RGB/action interface and a same-checkpoint goal-cost evaluation",
        },
        "gates": {
            "cross_environment_prediction": "report every environment, observed-vs-zero physical-action control, horizons 1/2/4, clustered confidence intervals",
            "cross_environment_planning": "not replaced by prediction: same frozen checkpoint must achieve held-out visual closed-loop success above paired zero/scripted baselines in at least two target mechanisms",
            "universal_claim": "prohibited unless both prediction and planning gates pass; this manifest alone cannot establish universality",
        },
        "commands": {"planning_readiness": readiness, "cross_environment_prediction": prediction},
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest["sha256_before_execution"] = digest
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.output / "README.md").write_text(
        "# Frozen external-validity protocol\n\n"
        "This protocol measures same-checkpoint cross-environment prediction and separately audits whether a legal common planning interface exists. "
        "A prediction result cannot substitute for the planning gate.\n"
    )
    print(json.dumps({"output": str(args.output), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
