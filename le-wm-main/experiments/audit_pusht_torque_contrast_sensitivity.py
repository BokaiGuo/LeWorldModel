"""Sweep privileged PushT torque-contrast sensitivity at fixed contact roots.

This standalone mechanism audit asks a deliberately narrow question: is there
a predeclared pulse amplitude/duration regime in which matched opposite torque
pulses have a measurable, low-noise angular contrast?  A root is selected once
before the sweep, then every ``+t``, ``-t`` and zero pulse is reset from that
same exact root state.  Simulator state and real successors are privileged and
may only be used offline to build this diagnostic.

The artifact is an *eligibility screen* for a later RGB ranker.  It neither
reads RGB nor selects an action at runtime, and it must never be reported as a
learned-controller, planning, or pose-closure result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch

import stable_worldmodel  # noqa: F401 - registers PushT

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_pusht_orientation_response import (  # noqa: E402
    collect_roots,
    edge_chart,
    matched_actions,
    probe,
)
from experiments.gpu_preflight import require_cuda_allocation  # noqa: E402
from experiments.run_pusht_receding_block_pose_planner import make_state_only_env  # noqa: E402
from experiments.run_pusht_signed_torque_audit import wrap_angle  # noqa: E402


ARTIFACT_NAME = "pusht_torque_contrast_sensitivity_audit"
ROOT_SPLIT_CONTRACT = "fixed_contact_root_split_v1"
RUNTIME_CONTRACT = {
    "runtime_action_selection": False,
    "runtime_successor_verification": False,
    "uses_privileged_simulator_state_offline_only": True,
    "is_learned_ranker": False,
    "is_planning_evaluation": False,
}


@dataclass(frozen=True)
class SweepCell:
    """A predeclared amplitude/duration condition with a stable identifier."""

    amplitude: float
    pulse_steps: int

    @property
    def key(self) -> str:
        return f"t{self.amplitude:.4f}__k{self.pulse_steps}"


@dataclass(frozen=True)
class ProbeSpec:
    """One reset-from-root probe in a same-state three-pulse triplet."""

    label: str
    state_digest: str
    action: tuple[float, float]
    pulse_steps: int
    reset_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--root-protocol", choices=("edge_boundary",), default="edge_boundary")
    parser.add_argument("--root-seed", type=int, default=2026102101)
    parser.add_argument("--states-per-bin", type=int, default=1)
    parser.add_argument("--fraction-bins", type=int, default=4)
    parser.add_argument("--max-roots", type=int, default=16)
    parser.add_argument("--max-contact-gap", type=float, default=30.0)
    parser.add_argument("--inward-magnitude", type=float, default=0.18)
    parser.add_argument("--amplitudes", nargs="+", type=float, default=(0.25, 0.40, 0.55, 0.70))
    parser.add_argument("--pulse-steps", nargs="+", type=int, default=(1, 2, 4))
    parser.add_argument("--min-effect", type=float, default=2.0e-4)
    parser.add_argument("--min-effect-to-noise", type=float, default=2.0)
    parser.add_argument("--min-eligible-fraction", type=float, default=0.50)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def root_digest(state: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(state, dtype=np.float64).tobytes()).hexdigest()


def predeclared_sweep(amplitudes: list[float] | tuple[float, ...], pulse_steps: list[int] | tuple[int, ...]) -> tuple[SweepCell, ...]:
    """Return the deterministic Cartesian sweep, rejecting ambiguous inputs."""
    cleaned_amplitudes = tuple(float(value) for value in amplitudes)
    cleaned_steps = tuple(int(value) for value in pulse_steps)
    if not cleaned_amplitudes or not cleaned_steps:
        raise ValueError("amplitudes and pulse-steps must both be non-empty")
    if any(not 0.0 < value <= 1.0 for value in cleaned_amplitudes):
        raise ValueError("all torque amplitudes must lie in (0, 1]")
    if any(value < 1 for value in cleaned_steps):
        raise ValueError("all pulse durations must be positive")
    if len(set(cleaned_amplitudes)) != len(cleaned_amplitudes):
        raise ValueError("duplicate amplitudes would make sweep cells ambiguous")
    if len(set(cleaned_steps)) != len(cleaned_steps):
        raise ValueError("duplicate pulse durations would make sweep cells ambiguous")
    return tuple(SweepCell(amplitude, duration) for amplitude in cleaned_amplitudes for duration in cleaned_steps)


def triplet_specs(state: np.ndarray, actions: dict[str, Any], cell: SweepCell, root_id: int, root_seed: int) -> tuple[ProbeSpec, ...]:
    """Bind all three actions to the same immutable source-state digest."""
    digest = root_digest(state)
    return tuple(
        ProbeSpec(
            label=label,
            state_digest=digest,
            action=tuple(float(value) for value in np.asarray(actions[label], dtype=np.float64)),
            pulse_steps=int(cell.pulse_steps),
            reset_seed=int(root_seed + root_id * 100_000 + cell.pulse_steps * 100 + index),
        )
        for index, label in enumerate(("zero", "plus", "minus"))
    )


def assert_same_state_pairing(specs: tuple[ProbeSpec, ...]) -> None:
    """Fail before simulation if the matched-pulse reset contract is broken."""
    if tuple(spec.label for spec in specs) != ("zero", "plus", "minus"):
        raise ValueError("triplet must contain zero, plus, minus in that fixed order")
    if len({spec.state_digest for spec in specs}) != 1:
        raise ValueError("every probe in a matched triplet must reset from one source state")
    if len({spec.pulse_steps for spec in specs}) != 1:
        raise ValueError("every probe in a matched triplet must have one duration")
    if not np.allclose(specs[0].action, (0.0, 0.0)):
        raise ValueError("the first matched probe must be the zero-pulse drift reference")


def run_matched_triplet(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    specs: tuple[ProbeSpec, ...],
) -> dict[str, dict[str, Any]]:
    """Execute offline real-successor probes, always resetting to ``state``."""
    assert_same_state_pairing(specs)
    state = np.asarray(state, dtype=np.float64).copy()
    if root_digest(state) != specs[0].state_digest:
        raise ValueError("state changed after triplet construction")
    return {
        spec.label: probe(env, state, goal, np.asarray(spec.action, dtype=np.float64), spec.reset_seed, spec.pulse_steps)
        for spec in specs
    }


def contrast_metrics(actions: dict[str, Any], probes: dict[str, dict[str, Any]], *, min_effect: float, min_effect_to_noise: float) -> dict[str, Any]:
    """Calculate zero-corrected torque contrast and later-ranker eligibility."""
    zero = float(probes["zero"]["signed_angle_delta"])
    effect_plus = float(wrap_angle(float(probes["plus"]["signed_angle_delta"]) - zero))
    effect_minus = float(wrap_angle(float(probes["minus"]["signed_angle_delta"]) - zero))
    contrast = float(wrap_angle(effect_plus - effect_minus) / 2.0)
    antisymmetry_rmse = float(math.sqrt(((effect_plus - contrast) ** 2 + (effect_minus + contrast) ** 2) / 2.0))
    noise = float(abs(zero) + antisymmetry_rmse + 1.0e-6)
    effect_to_noise = float(abs(contrast) / noise)
    moment_difference = float(actions["moment_plus"] - actions["moment_minus"])
    sign_consistent = bool(
        actions["pair_valid"] and abs(contrast) > 1.0e-8 and np.sign(contrast) == np.sign(moment_difference)
    )
    mode_agreement = bool(probes["plus"]["mode_after"] == probes["minus"]["mode_after"] == probes["zero"]["mode_after"])
    # Eligibility defines label quality only.  It is intentionally independent
    # of a task goal, visual features, a learned ranker, or a planning rollout.
    eligible_for_visual_ranker = bool(
        actions["pair_valid"]
        and mode_agreement
        and abs(contrast) >= float(min_effect)
        and effect_to_noise >= float(min_effect_to_noise)
    )
    return {
        "common_drift": zero,
        "effect_plus": effect_plus,
        "effect_minus": effect_minus,
        "paired_contrast": contrast,
        "paired_separation": float(abs(effect_plus - effect_minus)),
        "paired_antisymmetry_rmse": antisymmetry_rmse,
        "effect_to_noise": effect_to_noise,
        "sign_consistent": sign_consistent,
        "successor_mode_agreement": mode_agreement,
        "eligible_for_later_visual_ranker": eligible_for_visual_ranker,
    }


def gpu_mean_ci(values: list[float], device: torch.device, draws: int, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"estimate": None, "ci95_low": None, "ci95_high": None, "n": 0}
    tensor = torch.as_tensor(array, dtype=torch.float64, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    indices = torch.randint(array.size, (int(draws), array.size), generator=generator, device=device)
    means = tensor[indices].mean(dim=1)
    result = {
        "estimate": float(tensor.mean().item()),
        "ci95_low": float(torch.quantile(means, 0.025).item()),
        "ci95_high": float(torch.quantile(means, 0.975).item()),
        "n": int(array.size),
    }
    del tensor, indices, means
    torch.cuda.synchronize(device)
    return result


def cell_summary(rows: list[dict[str, Any]], device: torch.device, args: argparse.Namespace, cell_index: int) -> dict[str, Any]:
    metrics = ("sign_consistent", "paired_contrast", "paired_separation", "effect_to_noise", "eligible_for_later_visual_ranker", "successor_mode_agreement")
    summary = {
        metric: gpu_mean_ci([float(row[metric]) for row in rows], device, int(args.bootstrap), int(args.root_seed + cell_index * 101 + metric_index))
        for metric_index, metric in enumerate(metrics)
    }
    eligible_fraction = float(summary["eligible_for_later_visual_ranker"]["estimate"])
    summary["measurable_contrast_regime"] = bool(eligible_fraction >= float(args.min_eligible_fraction))
    summary["n_roots"] = len(rows)
    return summary


def protocol_manifest(args: argparse.Namespace, cells: tuple[SweepCell, ...], root_digests: list[str], cuda: dict[str, object]) -> dict[str, Any]:
    return {
        "artifact": ARTIFACT_NAME,
        "claim_boundary": (
            "Privileged real-successor sensitivity audit only. It reports whether a fixed matched-pulse regime produces "
            "measurable torque contrast eligible for future visual-ranker data collection; it makes no learned, runtime, planning, or pose-closure claim."
        ),
        "root_split": {
            "contract": ROOT_SPLIT_CONTRACT,
            "root_protocol": str(args.root_protocol),
            "root_seed": int(args.root_seed),
            "root_count": len(root_digests),
            "root_digests": root_digests,
            "selection_frozen_before_sweep": True,
        },
        "sweep": {
            "cells": [{"key": cell.key, "amplitude": cell.amplitude, "pulse_steps": cell.pulse_steps} for cell in cells],
            "inward_magnitude": float(args.inward_magnitude),
            "triplet": ("zero common-drift reference, +t, -t; each resets from exactly the same root state; " "real successor is offline measurement only"),
            "predeclared_before_results": True,
        },
        "eligibility_rule": {
            "min_effect": float(args.min_effect),
            "min_effect_to_noise": float(args.min_effect_to_noise),
            "requires_mode_agreement": True,
            "minimum_regime_eligible_fraction": float(args.min_eligible_fraction),
        },
        "runtime_contract": RUNTIME_CONTRACT,
        "analysis_device": cuda,
        "simulator_device": "cpu (Gymnasium/Pymunk); CUDA is required for analysis/bootstrap and preflight",
    }


def main() -> None:
    args = parse_args()
    # ``collect_roots`` is intentionally reused only as a fixed contact-root
    # constructor.  Its legacy helper names this value ``seed``; bind that
    # internal alias explicitly so no hidden default can affect the split.
    args.seed = int(args.root_seed)
    cells = predeclared_sweep(args.amplitudes, args.pulse_steps)
    cuda = require_cuda_allocation(args.device)
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    env = make_state_only_env()
    try:
        roots = collect_roots(env, args)
        root_digests = [root_digest(np.asarray(root["state"], dtype=np.float64)) for root in roots]
        rows_by_cell: dict[str, list[dict[str, Any]]] = {cell.key: [] for cell in cells}
        for root_id, root in enumerate(roots):
            state = np.asarray(root["state"], dtype=np.float64).copy()
            goal = np.asarray(root["goal"], dtype=np.float64).copy()
            chart = edge_chart(env, state)
            for cell in cells:
                actions = matched_actions(chart, float(args.inward_magnitude), float(cell.amplitude))
                specs = triplet_specs(state, actions, cell, root_id, int(args.root_seed))
                probes = run_matched_triplet(env, state, goal, specs)
                metrics = contrast_metrics(actions, probes, min_effect=float(args.min_effect), min_effect_to_noise=float(args.min_effect_to_noise))
                rows_by_cell[cell.key].append({
                    "root_id": int(root_id),
                    "root_state_digest": root_digests[root_id],
                    "edge_id": str(chart["edge_id"]),
                    "contact_gap": float(chart["contact_gap"]),
                    "cell": {"key": cell.key, "amplitude": cell.amplitude, "pulse_steps": cell.pulse_steps},
                    "action_norms": {label: float(np.linalg.norm(actions[label])) for label in ("zero", "plus", "minus")},
                    "moment_plus": float(actions["moment_plus"]),
                    "moment_minus": float(actions["moment_minus"]),
                    "pair_valid": bool(actions["pair_valid"]),
                    "same_state_pairing_verified": True,
                    "probe": probes,
                    **metrics,
                })
    finally:
        env.close()
    summaries = {cell.key: cell_summary(rows_by_cell[cell.key], device, args, index) for index, cell in enumerate(cells)}
    measurable = [key for key, summary in summaries.items() if summary["measurable_contrast_regime"]]
    manifest = protocol_manifest(args, cells, root_digests, cuda)
    report = {
        **manifest,
        "result": {
            "measurable_contrast_regimes": measurable,
            "measurable_contrast_regime_exists": bool(measurable),
            "interpretation": (
                "At least one predeclared cell cleared the offline label-quality rule; this authorizes only a later, separately "
                "split visual-ranker data study."
                if measurable
                else "No predeclared cell cleared the offline label-quality rule in this root panel; do not train or claim a visual torque ranker from this audit."
            ),
        },
        "cell_summary": summaries,
        "records": [row for cell in cells for row in rows_by_cell[cell.key]],
        "sanity_checks": {
            "all_cells_use_same_fixed_root_split": bool(all(row["root_state_digest"] in root_digests for rows in rows_by_cell.values() for row in rows)),
            "all_triplets_same_state_paired": bool(all(row["same_state_pairing_verified"] for rows in rows_by_cell.values() for row in rows)),
            "all_nonzero_pairs_equal_norm": bool(all(abs(row["action_norms"]["plus"] - row["action_norms"]["minus"]) < 1e-7 for rows in rows_by_cell.values() for row in rows)),
            "runtime_claim_absent": bool(not any(RUNTIME_CONTRACT[key] for key in ("runtime_action_selection", "runtime_successor_verification", "is_learned_ranker", "is_planning_evaluation"))),
            "wall_seconds": time.perf_counter() - started,
        },
    }
    (output / "torque_contrast_sensitivity.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# PushT torque-contrast sensitivity audit", "",
        f"- Fixed contact roots: `{len(root_digests)}`; sweep cells: `{len(cells)}`.",
        f"- CUDA analysis: `{cuda['name']}` (`{cuda['torch_cuda']}`).",
        f"- Measurable contrast regime exists: `{report['result']['measurable_contrast_regime_exists']}`.", "",
        "| cell | roots | sign consistency | effect-to-noise | eligibility | measurable |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, summary in summaries.items():
        lines.append(f"| {key} | {summary['n_roots']} | {summary['sign_consistent']['estimate']:.3f} | {summary['effect_to_noise']['estimate']:.3f} | {summary['eligible_for_later_visual_ranker']['estimate']:.3f} | {summary['measurable_contrast_regime']} |")
    lines += ["", report["claim_boundary"], ""]
    (output / "torque_contrast_sensitivity.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), "result": report["result"], "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
