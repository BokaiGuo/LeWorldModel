"""Run the fixed relative-effect geometry audit over a bank panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.gdd_wm.audit_relative_action_effect_geometry import audit


def _bootstrap_effects(
    report: dict,
    *,
    baseline: str = "root_centered_source_delta_action",
    samples: int = 5000,
    seed: int = 20261014,
) -> dict:
    """Compute paired root-bootstrap effects without using outcome for fitting."""

    variants = report["variants"]
    if baseline not in variants:
        raise ValueError(f"missing bootstrap baseline {baseline!r}")
    rng = np.random.default_rng(seed)
    output = {}
    for split in ("calibration", "outcome"):
        baseline_roots = variants[baseline]["root_metrics"].get(split, {})
        output[split] = {}
        for variant, result in variants.items():
            variant_roots = result["root_metrics"].get(split, {})
            roots = sorted(set(baseline_roots) & set(variant_roots))
            consistency_pairs = [
                root
                for root in roots
                if baseline_roots[root]["execution_consistency"] is not None
                and variant_roots[root]["execution_consistency"] is not None
            ]
            regret_pairs = [
                root
                for root in roots
                if baseline_roots[root]["regret"] is not None
                and variant_roots[root]["regret"] is not None
            ]

            def paired_ci(root_names: list[str], value_fn) -> dict:
                if not root_names:
                    return {"root_count": 0, "point": None, "ci95": [None, None]}
                point_values = np.asarray([value_fn(root) for root in root_names], dtype=np.float64)
                draws = rng.integers(0, len(root_names), size=(samples, len(root_names)))
                boot = np.mean(point_values[draws], axis=1)
                return {
                    "root_count": len(root_names),
                    "point": float(np.mean(point_values)),
                    "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                }

            output[split][variant] = {
                "baseline": baseline,
                "consistency_gain": paired_ci(
                    consistency_pairs,
                    lambda root: float(
                        variant_roots[root]["execution_consistency"]
                        - baseline_roots[root]["execution_consistency"]
                    ),
                ),
                "regret_reduction": paired_ci(
                    regret_pairs,
                    lambda root: float(baseline_roots[root]["regret"] - variant_roots[root]["regret"]),
                ),
            }
    return {
        "baseline": baseline,
        "bootstrap_samples": samples,
        "seed": seed,
        "estimand": "paired held-out root effect; positive means variant improves over baseline",
        "effects": output,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="name|candidate_dataset|arrays; repeat for each panel member",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261014)
    parser.add_argument(
        "--bootstrap-baseline",
        default="root_centered_source_delta_action",
        help="predeclared variant used as the paired bootstrap baseline",
    )
    args = parser.parse_args()
    panel = {}
    for specification in args.case:
        parts = specification.split("|")
        if len(parts) != 3 or not all(parts):
            raise ValueError("each --case must be name|candidate_dataset|arrays")
        name, dataset, arrays = parts
        if name in panel:
            raise ValueError(f"duplicate panel name: {name}")
        print(f"[relative-effect-panel] auditing {name}: root-centered interaction geometry")
        panel[name] = audit(Path(dataset), Path(arrays))
    report = {
        "schema": "gdd_wm_relative_action_effect_geometry_panel.v0.1",
        "status": "completed_diagnostic",
        "claim_boundary": (
            "cross-panel train-only relative-effect controls; not a D-JEPA teacher, "
            "not preference-stability evidence, and not closed-loop planning"
        ),
        "panel_member_order": list(panel),
        "panel": panel,
        "bootstrap": {
            name: _bootstrap_effects(
                result,
                baseline=args.bootstrap_baseline,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            )
            for name, result in panel.items()
        },
        "decision": {
            "variant_selection": "not_performed",
            "d1_ready": False,
            "d2_ready": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
