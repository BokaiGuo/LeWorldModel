"""Aggregate held-out RGB successor verification across learned-model cases.

The verifier reports candidate-level errors.  This utility first averages
within outcome roots, then keeps architecture/seed cases as a separate
uncertainty layer.  It is intentionally descriptive: it does not select a
checkpoint or authorize closed-loop planning.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _bootstrap_ci(values: np.ndarray, samples: int, rng: np.random.Generator) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "point": None, "ci95": [None, None]}
    draws = rng.integers(0, values.size, size=(samples, values.size))
    bootstrap = np.mean(values[draws], axis=1)
    return {
        "count": int(values.size),
        "point": float(np.mean(values)),
        "ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
    }


def _two_level_ci(
    cases: dict[str, dict[str, np.ndarray]],
    metric: str,
    case_names: list[str],
    samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    original = np.asarray([np.mean(cases[name][metric]) for name in case_names], dtype=np.float64)
    draws = []
    for _ in range(samples):
        sampled_cases = rng.integers(0, len(case_names), size=len(case_names))
        case_means = []
        for case_index in sampled_cases:
            values = cases[case_names[case_index]][metric]
            indices = rng.integers(0, values.size, size=values.size)
            case_means.append(float(np.mean(values[indices])))
        draws.append(float(np.mean(case_means)))
    draws = np.asarray(draws, dtype=np.float64)
    return {
        "count": int(draws.size),
        "point": float(np.mean(original)),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def _load_case(path: Path) -> dict[str, np.ndarray]:
    report = json.loads(path.read_text(encoding="utf-8"))
    gates = report.get("gates", {})
    required = ("renderer_consistency", "realized_rgb_latents_available", "pca_projected_successor_comparison")
    if not all(bool(gates.get(name)) for name in required):
        raise ValueError(f"RGB successor gates are not all passed in {path}")
    by_root: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"pred": [], "source": []})
    for row in report.get("rows", []):
        if str(row.get("split")) != "outcome":
            continue
        predicted = np.asarray(row["predicted_terminal_projection"], dtype=np.float64)
        realized = np.asarray(row["realized_terminal_projection"], dtype=np.float64)
        source = np.asarray(row["source_projection"], dtype=np.float64)
        root = str(row["instance_id"])
        by_root[root]["pred"].append(float(np.linalg.norm(predicted - realized)))
        by_root[root]["source"].append(float(np.linalg.norm(source - realized)))
    if not by_root:
        raise ValueError(f"no outcome rows in {path}")
    improvement = []
    better_fraction = []
    for values in by_root.values():
        pred = np.asarray(values["pred"], dtype=np.float64)
        source = np.asarray(values["source"], dtype=np.float64)
        improvement.append(1.0 - float(np.mean(pred)) / max(float(np.mean(source)), 1.0e-12))
        better_fraction.append(float(np.mean(pred < source)))
    return {
        "relative_l2_improvement": np.asarray(improvement, dtype=np.float64),
        "predicted_better_fraction": np.asarray(better_fraction, dtype=np.float64),
    }


def aggregate(
    cases_spec: list[str],
    output: Path,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    case_names = []
    cases: dict[str, dict[str, np.ndarray]] = {}
    paths = {}
    for specification in cases_spec:
        parts = specification.split("|")
        if len(parts) != 2 or not all(parts):
            raise ValueError("each case must be name|rgb_successor_report.json")
        name, path_text = parts
        if name in cases:
            raise ValueError(f"duplicate case {name!r}")
        case_names.append(name)
        paths[name] = str(Path(path_text).resolve())
        cases[name] = _load_case(Path(path_text))
    if not cases:
        raise ValueError("at least one case is required")
    rng = np.random.default_rng(seed)
    summary = {}
    for name in case_names:
        summary[name] = {
            metric: {
                "root_count": int(values.size),
                "point": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for metric, values in cases[name].items()
        }
    aggregate_report = {}
    for metric in ("relative_l2_improvement", "predicted_better_fraction"):
        pooled = np.concatenate([cases[name][metric] for name in case_names])
        case_means = np.asarray([np.mean(cases[name][metric]) for name in case_names], dtype=np.float64)
        aggregate_report[metric] = {
            "case_macro_mean": _bootstrap_ci(case_means, samples, rng),
            "root_cell_mean": _bootstrap_ci(pooled, samples, rng),
            "two_level_case_and_root_bootstrap": _two_level_ci(cases, metric, case_names, samples, rng),
            "all_case_points_positive": bool(all(np.mean(cases[name][metric]) > 0 for name in case_names)),
        }
    report = {
        "schema": "gdd_wm_rgb_successor_aggregate.v0.1",
        "status": "completed_diagnostic",
        "protocol": {
            "case_unit": "architecture_by_seed_checkpoint",
            "root_unit": "outcome_episode_root",
            "outcome_used_for_fit_or_variant_selection": False,
            "bootstrap_samples": samples,
            "bootstrap_seed": seed,
            "cases": paths,
        },
        "case_order": case_names,
        "case_summary": summary,
        "aggregate": aggregate_report,
        "claim_boundary": (
            "held-out dataset-matched RGB successor verification for simulated candidate actions; "
            "not closed-loop online planning, physical causality, D-JEPA teacher validation, "
            "or universal world-model evidence"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, help="name|rgb_successor_report.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261014)
    args = parser.parse_args()
    report = aggregate(args.case, args.output, samples=args.bootstrap_samples, seed=args.bootstrap_seed)
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
