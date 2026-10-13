"""Audit and aggregate the CPU visual-only PushT planning artifacts.

The simulator state is allowed for evaluator-side metrics, but must not be an
input to the tested vanilla policy.  This audit checks the artifact protocol,
rejects privileged-policy contamination, and reports paired vanilla-vs-zero
differences at the seed cluster level.  It is intentionally a negative-boundary
audit: it does not turn zero success into evidence of a successful planner.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import numpy as np


PRIVILEGED_POLICY_NAMES = {"heuristic", "privileged_heuristic", "spectral"}
FORBIDDEN_POLICY_INPUT_NAMES = {"state", "goal_state", "info_dict"}


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


def _bootstrap(values: np.ndarray, seed: int, draws: int = 20000) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
        "n_seed_clusters": int(len(values)),
    }


def _static_policy_audit(source_path: Path) -> dict[str, Any]:
    tree = ast.parse(source_path.read_text())
    make_policy = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "make_policy"), None)
    if make_policy is None:
        raise ValueError("could not locate make_policy")
    names = {node.id for node in ast.walk(make_policy) if isinstance(node, ast.Name)}
    return {
        "make_policy_present": True,
        "forbidden_policy_input_names_found": sorted(names & FORBIDDEN_POLICY_INPUT_NAMES),
        "privileged_policy_branch_names": sorted(names & PRIVILEGED_POLICY_NAMES),
        "vanilla_constructor_present": any(
            isinstance(node, ast.Attribute) and node.attr == "WorldModelPolicy"
            for node in ast.walk(make_policy)
        ),
        "evaluator_state_access_outside_make_policy": "state" in source_path.read_text(),
    }


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    _finite(payload)
    protocol = payload.get("protocol", {})
    reports = payload.get("reports", [])
    if not reports:
        raise ValueError(f"{path} has no reports")
    requested = set(protocol.get("policies", []))
    present = {str(report.get("policy")) for report in reports}
    if not present.issubset(requested):
        raise ValueError(f"artifact contains unrequested policies: {present - requested}")
    if requested & PRIVILEGED_POLICY_NAMES:
        raise ValueError(f"privileged policy requested in visual-only artifact: {requested & PRIVILEGED_POLICY_NAMES}")
    return payload


def audit(paths: list[Path], source_path: Path) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two seed artifacts are required")
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = _load(path)
        protocol = payload["protocol"]
        reports = {str(item["policy"]): item for item in payload["reports"]}
        if "vanilla" not in reports or "zero" not in reports:
            raise ValueError(f"{path} must contain paired vanilla and zero policies")
        zero = reports["zero"]["episodes"]
        vanilla = reports["vanilla"]["episodes"]
        if len(zero) != len(vanilla):
            raise ValueError(f"episode count mismatch in {path}")
        for left, right in zip(zero, vanilla):
            if int(left["episode"]) != int(right["episode"]):
                raise ValueError(f"episode ids are not paired in {path}")
            rows.append({
                "seed": int(protocol["seed"]),
                "episode": int(left["episode"]),
                "vanilla_minus_zero_final_distance": float(right["final_distance"] - left["final_distance"]),
                "vanilla_minus_zero_block_xy_distance": float(right["block_xy_distance"] - left["block_xy_distance"]),
                "vanilla_success": bool(right["success"]),
                "zero_success": bool(left["success"]),
            })
    seed_values: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        seed_values.setdefault(row["seed"], []).append(row)
    final_cluster = np.asarray([
        np.mean([row["vanilla_minus_zero_final_distance"] for row in values])
        for values in seed_values.values()
    ], dtype=np.float64)
    block_cluster = np.asarray([
        np.mean([row["vanilla_minus_zero_block_xy_distance"] for row in values])
        for values in seed_values.values()
    ], dtype=np.float64)
    protocol_first = json.loads(paths[0].read_text())["protocol"]
    static = _static_policy_audit(source_path)
    report = {
        "protocol": {
            "artifact_count": len(paths),
            "seed_clusters": sorted(seed_values),
            "paired_unit": "same seed and episode id across vanilla and zero",
            "cluster_unit_for_uncertainty": "seed; episodes averaged within seed before bootstrap",
            "policy_input_boundary": "vanilla receives transformed pixels and goal through WorldModelPolicy; simulator state is evaluator-side only",
            "horizon": protocol_first.get("horizon"),
            "device": protocol_first.get("device"),
        },
        "static_policy_audit": static,
        "sanity_checks": {
            "all_artifacts_cpu": all(json.loads(path.read_text())["protocol"].get("device") == "cpu" for path in paths),
            "no_privileged_requested_policy": True,
            "all_requested_policies_are_paired": True,
            "no_forbidden_policy_input_name_in_make_policy": not static["forbidden_policy_input_names_found"],
        },
        "aggregate": {
            "n_episode_pairs": len(rows),
            "n_seed_clusters": len(seed_values),
            "vanilla_success_total": int(sum(row["vanilla_success"] for row in rows)),
            "zero_success_total": int(sum(row["zero_success"] for row in rows)),
            "vanilla_minus_zero_final_distance": _bootstrap(final_cluster, 20261008),
            "vanilla_minus_zero_block_xy_distance": _bootstrap(block_cluster, 20261009),
        },
        "seed_cluster_rows": [
            {
                "seed": int(seed),
                "n_episode_pairs": len(values),
                "vanilla_success": int(sum(row["vanilla_success"] for row in values)),
                "zero_success": int(sum(row["zero_success"] for row in values)),
                "mean_vanilla_minus_zero_final_distance": float(np.mean([row["vanilla_minus_zero_final_distance"] for row in values])),
                "mean_vanilla_minus_zero_block_xy_distance": float(np.mean([row["vanilla_minus_zero_block_xy_distance"] for row in values])),
            }
            for seed, values in sorted(seed_values.items())
        ],
        "claim_boundary": [
            "All audited runs have zero binary success for vanilla and zero policies; this is negative boundary evidence, not a planning success claim.",
            "The paired distance comparison is evaluator-side simulator-state measurement and does not establish causal physical dynamics.",
            "The audit covers the supplied CPU artifacts and cannot establish visual planning generalization beyond these seeds and episodes.",
        ],
    }
    _finite(report)
    return report


def markdown(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    final = aggregate["vanilla_minus_zero_final_distance"]
    block = aggregate["vanilla_minus_zero_block_xy_distance"]
    lines = [
        "# Visual-only PushT planning audit",
        "",
        "This report audits the supplied CPU closed-loop artifacts. Simulator state is used only for evaluator-side distance/success metrics; the tested vanilla policy is checked for the absence of privileged state input in the policy-construction path.",
        "",
        "## Aggregate",
        "",
        f"- Episode pairs: {aggregate['n_episode_pairs']} across {aggregate['n_seed_clusters']} seed clusters.",
        f"- Binary success: vanilla {aggregate['vanilla_success_total']}/{aggregate['n_episode_pairs']}; zero {aggregate['zero_success_total']}/{aggregate['n_episode_pairs']}.",
        f"- Vanilla minus zero final-distance difference: {final['mean']:.6g} [{final['ci_low']:.6g}, {final['ci_high']:.6g}] (seed-cluster bootstrap).",
        f"- Vanilla minus zero block-XY difference: {block['mean']:.6g} [{block['ci_low']:.6g}, {block['ci_high']:.6g}] (seed-cluster bootstrap).",
        "",
        "## Seed clusters",
        "",
        "| Seed | Episode pairs | Vanilla successes | Zero successes | Final-distance difference | Block-XY difference |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["seed_cluster_rows"]:
        lines.append(f"| {row['seed']} | {row['n_episode_pairs']} | {row['vanilla_success']} | {row['zero_success']} | {row['mean_vanilla_minus_zero_final_distance']:.6g} | {row['mean_vanilla_minus_zero_block_xy_distance']:.6g} |")
    lines.extend([
        "",
        "## Audit checks",
        "",
        f"- CPU artifacts: `{report['sanity_checks']['all_artifacts_cpu']}`.",
        f"- No privileged policy requested: `{report['sanity_checks']['no_privileged_requested_policy']}`.",
        f"- No forbidden state/input name in `make_policy`: `{report['sanity_checks']['no_forbidden_policy_input_name_in_make_policy']}`.",
        "",
        "## Boundary",
        "",
        *[f"- {item}" for item in report["claim_boundary"]],
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", action="append", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=Path(__file__).with_name("run_online_pusht_planning.py"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.artifact, args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "visual_only_planning_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "visual_only_planning_audit.md").write_text(markdown(report))
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
