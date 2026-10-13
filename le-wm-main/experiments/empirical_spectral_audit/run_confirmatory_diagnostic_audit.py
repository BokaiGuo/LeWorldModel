"""Cluster-level audit for the learned-checkpoint diagnostic benchmark.

The original benchmark intentionally records many split seeds.  Those splits
are useful stability probes, but they are not independent learned models or
episodes.  This audit therefore makes the source artifact the resampling unit
and reports the source-level estimand:

    mean_source[alignment regret - validation-error regret].

The h=4, 65-candidate report is declared primary.  h=2 with 65 candidates
and h=4 with 17 candidates are secondary sensitivity endpoints.  The script
uses an exact sign permutation over source clusters for one-sided tests and a
source-cluster bootstrap for intervals.  It never treats split seeds as
independent episode-level replication.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ALPHA = 0.05
PRIMARY = "h4_k65"


def _finite(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        if not np.isfinite(float(value)):
            raise FloatingPointError(f"non-finite value: {value}")


def _source_effects(payload: dict[str, Any], field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for record in payload.get("records", []):
        label = str(record["label"])
        if field in record:
            value = float(record[field])
        elif field == "alignment_minus_validation_regret":
            # Older benchmark artifacts stored this quantity only in the
            # pooled summary. Recompute it from the two record-level regrets
            # so the cluster audit never consumes a pooled statistic.
            value = float(
                record["alignment_selection_regret"]
                - record["validation_selection_regret"]
            )
        else:
            raise KeyError(f"record lacks required field {field!r}")
        grouped.setdefault(label, []).append(value)
    if not grouped:
        raise ValueError("benchmark report contains no records")
    return {label: float(np.mean(values)) for label, values in sorted(grouped.items())}


def _cluster_bootstrap(values: np.ndarray, seed: int, draws: int = 20000) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("cluster values must be a non-empty vector")
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
        "n_source_clusters": int(values.size),
    }


def _exact_one_sided_sign_pvalue(values: np.ndarray) -> float:
    """P(mean(delta) <= observed) under independent symmetric signs.

    Negative delta means alignment has lower regret than validation.  The
    returned p-value tests the directional alternative mean(delta) < 0.
    """
    values = np.asarray(values, dtype=np.float64)
    observed = float(values.mean())
    n = int(values.size)
    if n > 20:
        raise ValueError("exact sign enumeration is intended for small cluster counts")
    signs = np.asarray(
        [[1.0 if (mask >> i) & 1 else -1.0 for i in range(n)] for mask in range(1 << n)],
        dtype=np.float64,
    )
    null_means = (signs * np.abs(values)[None, :]).mean(axis=1)
    return float((np.count_nonzero(null_means <= observed + 1e-15) + 1) / (null_means.size + 1))


def _holm(pvalues: dict[str, float], alpha: float = ALPHA) -> dict[str, Any]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, (name, pvalue) in enumerate(ordered):
        value = min(1.0, (m - rank) * pvalue)
        running = max(running, value)
        adjusted[name] = running
    return {
        "raw": {name: float(value) for name, value in pvalues.items()},
        "holm_adjusted": adjusted,
        "alpha": alpha,
        "reject": {name: bool(adjusted[name] < alpha) for name in adjusted},
    }


def _read_endpoint(label: str, path: Path, seed: int) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    h4_field = "alignment_selection_regret"
    h4_reference = "validation_selection_regret"
    del h4_field, h4_reference
    deltas = _source_effects(payload, "alignment_minus_validation_regret")
    source_names = list(deltas)
    values = np.asarray([deltas[name] for name in source_names], dtype=np.float64)
    bootstrap = _cluster_bootstrap(values, seed)
    pvalue = _exact_one_sided_sign_pvalue(values)
    protocol = payload.get("protocol", {})
    return {
        "label": label,
        "path": str(path),
        "protocol": {
            "downstream_horizon": protocol.get("downstream_horizon"),
            "candidate_count": protocol.get("candidate_count"),
            "prediction_error_tolerance": protocol.get("prediction_error_tolerance"),
            "uncertainty_unit_original": protocol.get("uncertainty_unit"),
        },
        "estimand": "equal-weight mean across source checkpoint/environment artifacts of alignment regret minus validation-error regret",
        "source_effects": deltas,
        "source_cluster_bootstrap": bootstrap,
        "exact_one_sided_sign_pvalue": pvalue,
        "split_seed_count_total": len(payload.get("records", [])),
        "split_seeds_are_not_independent_units": True,
    }


def audit(endpoints: dict[str, Path], seed: int = 20261008) -> dict[str, Any]:
    if PRIMARY not in endpoints:
        raise ValueError(f"primary endpoint {PRIMARY!r} is required")
    entries = {
        label: _read_endpoint(label, path, seed + index)
        for index, (label, path) in enumerate(endpoints.items())
    }
    pvalues = {label: float(item["exact_one_sided_sign_pvalue"]) for label, item in entries.items()}
    correction = _holm(pvalues)
    report = {
        "protocol": {
            "primary_endpoint": PRIMARY,
            "primary_estimand": entries[PRIMARY]["estimand"],
            "secondary_endpoints": [label for label in entries if label != PRIMARY],
            "primary_test": "one-sided exact sign permutation over source clusters",
            "interval": "95% source-cluster bootstrap; split seeds nested within source are averaged before resampling",
            "multiple_comparison_family": "the declared primary plus the two declared sensitivity endpoints",
            "independent_unit": "source checkpoint/environment artifact, not split seed, episode, horizon, or candidate",
            "confirmatory_status": {label: (label == PRIMARY) for label in entries},
        },
        "endpoints": entries,
        "holm_correction": correction,
        "claim_boundary": [
            "The primary result is a source-cluster stability analysis, not an episode-level generalization claim.",
            "A positive result would support conditional model-selection utility for the declared fixed-spectrum probe family only.",
            "The audit does not establish end-to-end neural training gains, physical causality, or online planning.",
        ],
    }
    _finite(report)
    return report


def markdown(report: dict[str, Any]) -> str:
    protocol = report["protocol"]
    lines = [
        "# Confirmatory covariance--predictor diagnostic audit",
        "",
        "This audit treats each source checkpoint/environment artifact as one cluster. Split seeds are nested stability probes and are averaged within source before inference.",
        "",
        "## Predeclared protocol",
        "",
        f"- Primary endpoint: `{protocol['primary_endpoint']}` (horizon 4, 65 candidates).",
        "- Secondary endpoints: horizon 2 with 65 candidates and horizon 4 with 17 candidates, when supplied.",
        "- Estimand: equal-weight source-cluster mean of alignment regret minus validation-error regret.",
        "- Test: one-sided exact sign permutation over source clusters; intervals: 95% source-cluster bootstrap.",
        "- Split seeds, episodes, candidates, and horizons are not independent replication units.",
        "",
        "## Results",
        "",
        "| Endpoint | Horizon | Candidates | Source clusters | Mean delta | 95% cluster bootstrap CI | Exact p | Holm p | Reject |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    adjusted = report["holm_correction"]["holm_adjusted"]
    for label, item in report["endpoints"].items():
        summary = item["source_cluster_bootstrap"]
        pvalue = item["exact_one_sided_sign_pvalue"]
        horizon = item["protocol"]["downstream_horizon"]
        candidates = item["protocol"]["candidate_count"]
        lines.append(
            f"| {label} | {horizon} | {candidates} | {summary['n_source_clusters']} | "
            f"{summary['mean']:.6g} | [{summary['ci_low']:.6g}, {summary['ci_high']:.6g}] | "
            f"{pvalue:.6g} | {adjusted[label]:.6g} | {report['holm_correction']['reject'][label]} |"
        )
    lines.extend(["", "## Source-level effects", "", "| Endpoint | Source | Delta |", "| --- | --- | ---: |"])
    for label, item in report["endpoints"].items():
        for source, value in item["source_effects"].items():
            lines.append(f"| {label} | {source} | {value:.6g} |")
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", action="append", required=True, help="LABEL=diagnostic.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    endpoints: dict[str, Path] = {}
    for value in args.endpoint:
        if "=" not in value:
            raise ValueError(f"--endpoint must be LABEL=PATH, got {value!r}")
        label, path = value.split("=", 1)
        endpoints[label] = Path(path)
    report = audit(endpoints)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "confirmatory_diagnostic_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "confirmatory_diagnostic_audit.md").write_text(markdown(report))
    print(json.dumps(report["holm_correction"], indent=2))


if __name__ == "__main__":
    main()
