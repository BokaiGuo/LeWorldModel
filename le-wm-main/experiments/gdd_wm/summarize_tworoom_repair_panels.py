"""Combine independent paired TwoRoom verified-local repair panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scipy.stats import binomtest


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol_signature(payload: dict, path: Path) -> str:
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict) or not protocol:
        raise ValueError(f"{path}: planning trace requires a non-empty protocol")
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _successes(payload: dict, key: str, *, expected_n: int, expected_seeds: list[int], path: Path) -> list[int]:
    arm = payload.get(key)
    values = arm.get("episode_successes") if isinstance(arm, dict) else None
    seeds = arm.get("seeds") if isinstance(arm, dict) else None
    if not isinstance(values, list) or len(values) != expected_n or any(value not in (0, 1) for value in values):
        raise ValueError(f"{path}: {key} must retain {expected_n} binary episode successes")
    if seeds != expected_seeds:
        raise ValueError(f"{path}: {key} successes are not aligned to the paired seed order")
    return [int(value) for value in values]


def _validate_panel(path: Path, *, expected_intervention_key: str | None, prior_seeds: set[int]) -> tuple[dict, set[int], str, str, str]:
    """Bind a summary panel to its paired source traces before aggregation."""

    payload = _load(path)
    if payload.get("schema") != "gdd_wm_paired_tworoom_repair_consequence.v0.1":
        raise ValueError(f"unexpected schema: {path}")
    protocol = payload.get("protocol")
    success = payload.get("success")
    if not isinstance(protocol, dict) or not isinstance(success, dict):
        raise ValueError(f"{path}: missing protocol or success payload")
    if protocol.get("outcome_used_for_selection") is not False or protocol.get("same_seed_panel") is not True:
        raise ValueError(f"{path}: requires outcome-free same-seed paired protocol")
    intervention_key = protocol.get("intervention_key")
    if not isinstance(intervention_key, str) or not intervention_key:
        raise ValueError(f"{path}: missing intervention key")
    if expected_intervention_key is not None and intervention_key != expected_intervention_key:
        raise ValueError(f"{path}: intervention key differs across panels")
    n = int(payload.get("n", 0))
    baseline_path = Path(str(protocol.get("baseline", ""))).resolve()
    intervention_path = Path(str(protocol.get("intervention", ""))).resolve()
    if n <= 0 or not baseline_path.is_file() or not intervention_path.is_file():
        raise ValueError(f"{path}: missing paired planning traces")
    baseline_trace, intervention_trace = _load(baseline_path), _load(intervention_path)
    baseline_signature = _protocol_signature(baseline_trace, baseline_path)
    intervention_signature = _protocol_signature(intervention_trace, intervention_path)
    baseline_seeds = baseline_trace.get("paired", {}).get("seed_order")
    intervention_seeds = intervention_trace.get("paired", {}).get("seed_order")
    if not isinstance(baseline_seeds, list) or baseline_seeds != intervention_seeds or len(baseline_seeds) != n:
        raise ValueError(f"{path}: planning traces do not retain the same {n}-seed paired order")
    seeds = {int(value) for value in baseline_seeds}
    if len(seeds) != n or seeds & prior_seeds:
        raise ValueError(f"{path}: panel seeds are duplicate or overlap an earlier panel")
    baseline = _successes(
        baseline_trace, "frozen_lewm", expected_n=n, expected_seeds=baseline_seeds, path=baseline_path,
    )
    intervention = _successes(
        intervention_trace, intervention_key, expected_n=n, expected_seeds=baseline_seeds, path=intervention_path,
    )
    expected = {
        "baseline": sum(baseline),
        "intervention": sum(intervention),
        "baseline_only": sum(left == 1 and right == 0 for left, right in zip(baseline, intervention)),
        "intervention_only": sum(left == 0 and right == 1 for left, right in zip(baseline, intervention)),
    }
    observed = {name: int(success.get(name, -1)) for name in expected}
    if observed != expected:
        raise ValueError(f"{path}: summary success counts disagree with paired planning traces")
    return payload, seeds, intervention_key, baseline_signature, intervention_signature


def summarize(paths: list[Path], output: Path) -> dict:
    if len(paths) < 2:
        raise ValueError("at least two independent panels are required")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite TwoRoom repair-panel summary: {output}")
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("duplicate TwoRoom panel path is forbidden")
    panels = []
    baseline = intervention = baseline_only = intervention_only = total = 0
    gate_calls = fallback_count = validation_fallback_count = effect_policy_count = 0
    phase_counts = {"door": 0, "goal": 0}
    weighted_cosine_sum = 0.0
    cosine_weight = 0
    all_seeds: set[int] = set()
    intervention_key: str | None = None
    baseline_signature: str | None = None
    intervention_signature: str | None = None
    for path in resolved:
        payload, seeds, panel_intervention_key, panel_baseline_signature, panel_intervention_signature = _validate_panel(
            path, expected_intervention_key=intervention_key, prior_seeds=all_seeds,
        )
        if baseline_signature is not None and panel_baseline_signature != baseline_signature:
            raise ValueError(f"{path}: baseline planning protocol differs across panels")
        if intervention_signature is not None and panel_intervention_signature != intervention_signature:
            raise ValueError(f"{path}: intervention planning protocol differs across panels")
        intervention_key = panel_intervention_key
        baseline_signature = panel_baseline_signature
        intervention_signature = panel_intervention_signature
        all_seeds.update(seeds)
        success = payload["success"]
        n = int(payload["n"])
        total += n
        baseline += int(success["baseline"])
        intervention += int(success["intervention"])
        baseline_only += int(success["baseline_only"])
        intervention_only += int(success["intervention_only"])
        diagnostics = payload.get("runtime_diagnostics", {})
        gate_calls += int(diagnostics.get("gate_calls", 0))
        fallback_count += int(diagnostics.get("fallback_count", 0))
        validation_fallback_count += int(
            diagnostics.get("effect_policy_validation_fallback_count", 0)
        )
        effect_policy_count += int(diagnostics.get("effect_policy_count", 0))
        for phase, count in diagnostics.get("phase_counts", {}).items():
            phase_counts[phase] = phase_counts.get(phase, 0) + int(count)
        if diagnostics.get("cosine_mean") is not None and diagnostics.get("gate_calls", 0):
            weighted_cosine_sum += float(diagnostics["cosine_mean"]) * int(
                diagnostics["gate_calls"]
            )
            cosine_weight += int(diagnostics["gate_calls"])
        panels.append({
            "path": str(path.resolve()),
            "n": n,
            "baseline": int(success["baseline"]),
            "intervention": int(success["intervention"]),
            "baseline_only": int(success["baseline_only"]),
            "intervention_only": int(success["intervention_only"]),
            "intervention_key": payload["protocol"]["intervention_key"],
            "seed_count": len(seeds),
            "seed_min": min(seeds),
            "seed_max": max(seeds),
            "panel_sha256": _sha256(path),
            "baseline_trace_sha256": _sha256(Path(payload["protocol"]["baseline"])),
            "intervention_trace_sha256": _sha256(Path(payload["protocol"]["intervention"])),
            "baseline_protocol_signature": panel_baseline_signature,
            "intervention_protocol_signature": panel_intervention_signature,
            "outcome_used_for_selection": bool(
                payload["protocol"]["outcome_used_for_selection"]
            ),
            "fallback_rate": diagnostics.get("fallback_rate"),
            "gate_calls": int(diagnostics.get("gate_calls", 0)),
            "effect_policy_count": int(diagnostics.get("effect_policy_count", 0)),
            "effect_policy_validation_fallback_count": int(
                diagnostics.get("effect_policy_validation_fallback_count", 0)
            ),
            "cosine_mean": diagnostics.get("cosine_mean"),
            "cosine_p10": diagnostics.get("cosine_p10"),
            "phase_counts": diagnostics.get("phase_counts", {}),
        })
    discordant = baseline_only + intervention_only
    report = {
        "schema": "gdd_wm_paired_tworoom_repair_panels.v0.1",
        "protocol": {
            "independent_root_panels": True,
            "same_seed_within_panel": True,
            "outcome_used_for_selection": False,
            "trace_backed_success_counts": True,
            "cross_panel_seed_overlap": [],
            "intervention_key": intervention_key,
            "baseline_protocol_signature": baseline_signature,
            "intervention_protocol_signature": intervention_signature,
        },
        "panels": panels,
        "combined": {
            "n": total,
            "baseline_successes": baseline,
            "intervention_successes": intervention,
            "baseline_only": baseline_only,
            "intervention_only": intervention_only,
            "success_delta": intervention - baseline,
            "exact_mcnemar_p": float(
                binomtest(min(baseline_only, intervention_only), discordant,
                          p=0.5).pvalue if discordant else 1.0
            ),
            "runtime_diagnostics": {
                "gate_calls": gate_calls,
                "fallback_count": fallback_count,
                "fallback_rate": fallback_count / max(gate_calls, 1),
                "effect_policy_count": effect_policy_count,
                "effect_policy_validation_fallback_count": validation_fallback_count,
                "phase_counts": phase_counts,
                "weighted_cosine_mean": weighted_cosine_sum / max(cosine_weight, 1),
            },
        },
        "claim_boundary": (
            "positive point estimate is a TwoRoom held-out repair result, not a "
            "universal transfer claim; no outcome-based panel selection was used"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.panel, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
