"""Aggregate sealed V16 obstacle-challenge panels without pooling time blocks.

The V16 obstacle benchmark has a design panel followed by independently held
out primary and replication panels.  This utility is deliberately analysis
only: it binds reports to one challenge manifest, verifies disjoint root sets,
and bootstraps *roots* (never trace blocks).  It also reports the statically
declared start-cell strata so that a positive first panel cannot be mistaken
for a stable intervention when the independent panel has a different mixture
of obstacle geometries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


SCORE_SCHEMA = "pointmaze_v16_position_anchored_score.v1"
MANIFEST_SCHEMA = "pointmaze_geometry_stratified_obstacle_extension.v1"
BASELINE = "rgb_greedy"
ORACLE = "privileged_one_step_oracle"
METHODS = ("rgb_position_anchored_velocity", "v14_full_state_dynamics_diagnostic")


def _bootstrap(values: np.ndarray, *, samples: int, seed: int) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("bootstrap needs one or more root-level values")
    if int(samples) < 100:
        raise ValueError("bootstrap needs at least 100 resamples")
    rng = np.random.default_rng(int(seed))
    draws = np.mean(rng.choice(values, size=(int(samples), len(values)), replace=True), axis=1)
    return {
        "mean_gain": float(values.mean()),
        "lower": float(np.quantile(draws, .025)),
        "upper": float(np.quantile(draws, .975)),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _cell_labels(manifest: dict[str, Any]) -> dict[int, str]:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unexpected obstacle-challenge manifest schema")
    base = int(manifest["base_root_count"])
    starts = manifest.get("static_start_cells_qpos")
    sampled = manifest.get("sampled_start_cell_indices")
    if not isinstance(starts, list) or not isinstance(sampled, list) or len(sampled) != int(manifest["new_episode_count"]):
        raise ValueError("malformed challenge start-cell mapping")
    labels: dict[int, str] = {}
    for offset, cell_index in enumerate(sampled):
        if not isinstance(cell_index, int) or cell_index < 0 or cell_index >= len(starts):
            raise ValueError("invalid challenge start-cell index")
        cell = np.asarray(starts[cell_index], dtype=np.float64)
        if cell.shape != (2,):
            raise ValueError("invalid challenge start cell")
        labels[base + offset] = f"({cell[0]:.1f},{cell[1]:.1f})"
    return labels


def _rows_by_root(report: dict[str, Any]) -> dict[int, dict[str, dict[str, Any]]]:
    if report.get("schema") != SCORE_SCHEMA:
        raise ValueError("unexpected V16 score schema")
    roots = [int(root) for root in report.get("roots", [])]
    if len(roots) < 8 or len(roots) != len(set(roots)):
        raise ValueError("score report must contain at least eight unique roots")
    by_root = {root: {} for root in roots}
    for row in report.get("rows", []):
        root, method = int(row["episode_id"]), str(row["method"])
        if root not in by_root or method in by_root[root]:
            raise ValueError("malformed or duplicate V16 score row")
        by_root[root][method] = row
    expected = {BASELINE, ORACLE, *METHODS}
    if any(set(rows) != expected for rows in by_root.values()):
        raise ValueError("each root must contain the same four V16 arms")
    return by_root


def _trace_match(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float | int | None]:
    left_trace, right_trace = list(left["trace"]), list(right["trace"])
    if not left_trace or not right_trace:
        return {"first_candidate_match": None, "shared_blocks": 0, "trace_candidate_match_rate": None}
    if int(left_trace[0]["block"]) != 0 or int(right_trace[0]["block"]) != 0:
        raise ValueError("V16 traces must start at block zero")
    common = min(len(left_trace), len(right_trace))
    matches = [int(left_trace[index]["selected_candidate_id"]) == int(right_trace[index]["selected_candidate_id"]) for index in range(common)]
    return {
        "first_candidate_match": float(matches[0]),
        "shared_blocks": int(common),
        # Paths differ after an action mismatch. This is a diagnostic proxy,
        # not an action-ranking accuracy claim on shared states.
        "trace_candidate_match_rate": float(np.mean(matches)),
    }


def aggregate(
    reports: list[Path],
    manifest_path: Path,
    *,
    samples: int = 4000,
    seed: int = 202610421,
) -> dict[str, Any]:
    """Aggregate independently scored V16 panels at the root level."""
    if len(reports) < 2:
        raise ValueError("V16 aggregate requires at least primary and replication reports")
    manifest = _read_json(manifest_path)
    root_to_cell = _cell_labels(manifest)
    pooled: dict[int, dict[str, dict[str, Any]]] = {}
    panel_summaries: list[dict[str, Any]] = []
    common_goal: list[float] | None = None
    common_design: str | None = None
    for report_path in reports:
        report = _read_json(report_path)
        by_root = _rows_by_root(report)
        roots = sorted(by_root)
        if set(roots) & set(pooled):
            raise ValueError("V16 outcome roots overlap across reports")
        if not set(roots).issubset(root_to_cell):
            raise ValueError("V16 report contains roots outside the obstacle challenge manifest")
        goal = [float(value) for value in report.get("selected_goal_qpos", [])]
        design = str(report.get("design_sha256", ""))
        if len(goal) != 2 or not design:
            raise ValueError("V16 report lacks sealed design binding")
        if common_goal is None:
            common_goal, common_design = goal, design
        elif goal != common_goal or design != common_design:
            raise ValueError("V16 reports are not bound to the same goal/design")
        pooled.update(by_root)
        panel_summaries.append({
            "report": str(Path(report_path).resolve()),
            "root_count": len(roots),
            "roots": roots,
            "start_cell_counts": {cell: sum(root_to_cell[root] == cell for root in roots) for cell in sorted(set(root_to_cell.values()))},
            "reported_position_anchored_over_rgb_greedy": report["paired_position_anchored_over_rgb_greedy"],
        })

    roots = sorted(pooled)
    def paired(method: str) -> np.ndarray:
        return np.asarray([float(pooled[root][BASELINE]["final_distance"]) - float(pooled[root][method]["final_distance"]) for root in roots])

    root_rows = []
    for root in roots:
        row: dict[str, Any] = {"episode_id": root, "static_start_cell": root_to_cell[root]}
        for method in (BASELINE, *METHODS, ORACLE):
            arm = pooled[root][method]
            row[method] = {"final_distance": float(arm["final_distance"]), "success": bool(arm["success"])}
        for method in METHODS:
            row[f"{method}_gain_over_rgb_greedy"] = float(row[BASELINE]["final_distance"] - row[method]["final_distance"])
        root_rows.append(row)

    strata: dict[str, Any] = {}
    for cell in sorted(set(root_to_cell[root] for root in roots)):
        cell_roots = [root for root in roots if root_to_cell[root] == cell]
        strata[cell] = {
            "root_count": len(cell_roots),
            "position_anchored_mean_gain_over_rgb_greedy": float(np.mean([pooled[root][BASELINE]["final_distance"] - pooled[root][METHODS[0]]["final_distance"] for root in cell_roots])),
            "position_anchored_win_count": int(sum(pooled[root][METHODS[0]]["final_distance"] < pooled[root][BASELINE]["final_distance"] for root in cell_roots)),
            "rgb_greedy_success_rate": float(np.mean([pooled[root][BASELINE]["success"] for root in cell_roots])),
            "position_anchored_success_rate": float(np.mean([pooled[root][METHODS[0]]["success"] for root in cell_roots])),
        }

    action_match: dict[str, Any] = {}
    for method in (BASELINE, *METHODS):
        diagnostics = [_trace_match(pooled[root][method], pooled[root][ORACLE]) for root in roots]
        first = [float(value["first_candidate_match"]) for value in diagnostics if value["first_candidate_match"] is not None]
        weighted_shared = sum(int(value["shared_blocks"]) for value in diagnostics)
        weighted_matches = sum(float(value["trace_candidate_match_rate"]) * int(value["shared_blocks"]) for value in diagnostics if value["trace_candidate_match_rate"] is not None)
        action_match[method] = {
            "first_decision_candidate_match_rate_to_oracle": float(np.mean(first)),
            "trace_aligned_candidate_match_rate_to_oracle": float(weighted_matches / weighted_shared) if weighted_shared else None,
            "shared_trace_block_count": int(weighted_shared),
            "interpretation": "first decision is compared under a common root; trace-aligned blocks are a post-divergence diagnostic proxy, not common-state ranking accuracy",
        }

    def success_delta(method: str) -> float:
        return float(np.mean([pooled[root][method]["success"] for root in roots]) - np.mean([pooled[root][BASELINE]["success"] for root in roots]))

    return {
        "schema": "pointmaze_v16_obstacle_challenge_aggregate.v1",
        "manifest": str(Path(manifest_path).resolve()),
        "report_count": len(reports),
        "design_sha256": common_design,
        "selected_goal_qpos": common_goal,
        "outcome_root_count": len(roots),
        "outcome_roots": roots,
        "panels": panel_summaries,
        "pooled_root_level_paired_position_anchored_over_rgb_greedy": _bootstrap(paired(METHODS[0]), samples=samples, seed=seed),
        "pooled_root_level_paired_full_v14_over_rgb_greedy": _bootstrap(paired(METHODS[1]), samples=samples, seed=seed + 1),
        "pooled_root_level_paired_oracle_over_rgb_greedy": _bootstrap(paired(ORACLE), samples=samples, seed=seed + 2),
        "pooled_success_rate_delta_over_rgb_greedy": {method: success_delta(method) for method in (*METHODS, ORACLE)},
        "static_start_cell_strata": strata,
        "oracle_action_match_diagnostics": action_match,
        "root_rows": root_rows,
        "claim_boundary": "Independent held-out PointMaze obstacle-challenge aggregation. Bootstrap resampling unit is the root. It tests one frozen RGB-position-plus-V14-velocity policy on one predeclared maze/goal, not frozen-latent-only or cross-environment validity.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=202610421)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite sealed V16 aggregate artifact")
    result = aggregate(args.reports, args.manifest, samples=args.bootstrap_samples, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(args.output), "outcome_root_count": result["outcome_root_count"], "pooled": result["pooled_root_level_paired_position_anchored_over_rgb_greedy"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
