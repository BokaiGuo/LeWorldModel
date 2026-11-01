"""Preseal an episode-disjoint all8 confirmation, then execute every stage.

No parameter search or outcome-dependent rerouting. All child neural runs
require CUDA and the same original frozen checkpoint; analysis is NumPy.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import audit_reacher_d2p_waypoint_tracking as tracker
from experiments.spectral_reachability import run_reacher_d2p_state_native_confirmation as planner
from experiments.spectral_reachability import analyze_reacher_d2p_state_native_selector as planning_analysis
from experiments.spectral_reachability import analyze_reacher_d2p_waypoint_tracking as tracking_analysis
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new
from experiments.spectral_reachability.run_reacher_d2p_suffix_confirmation import bind_exclusions
from experiments.spectral_reachability.run_reacher_d2p_program_dispatch_diagnostic import planned_source_roots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--alignment-reference", type=Path, required=True)
    parser.add_argument("--disjoint-reference", type=Path, required=True)
    parser.add_argument("--pilot-tracking", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202648000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("new immutable confirmation directory required")
    seeds = list(range(args.seed, args.seed + 8))
    excluded, bindings = bind_exclusions([args.disjoint_reference], seeds)
    pilot = json.loads(args.pilot_tracking.read_text())
    if (pilot["status"] != "completed" or not pilot.get("gates") or not all(v is True for v in pilot["gates"].values())
            or not {r["source_episode"] for r in pilot["rows"]}.issubset(excluded)):
        raise ValueError("exclude every pilot source episode before confirmation")
    tracking_analysis.load(args.pilot_tracking)
    planner.load_alignment(args.alignment_reference)
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    roots = planned_source_roots(dataset, seeds, excluded)
    model_sha = digest(args.policy / "weights.pt")
    if model_sha != pilot["sealed_protocol"]["weights_sha256"]:
        raise ValueError("same official frozen checkpoint required")
    args.output_dir.mkdir(parents=True)
    root_dir = args.output_dir.resolve()
    reference, tracking = root_dir / "planning.json", root_dir / "tracking.json"
    common = ["--policy", str(args.policy.resolve()), "--adapter-artifact", str(args.adapter_artifact.resolve())]
    commands = [
        [sys.executable, "-u", "-m", planner.__name__, *common, "--alignment-reference", str(args.alignment_reference.resolve()),
            "--disjoint-reference", str(args.disjoint_reference.resolve()), "--seed", str(args.seed), "--roots", "8", "--goal-contract", "joint-feasible-v1", "--output", str(reference)],
        [sys.executable, "-u", "-m", tracker.__name__, *common, "--reference", str(reference), "--output", str(tracking)],
        [sys.executable, "-m", planning_analysis.__name__, "--report", str(reference), "--output-dir", str(root_dir / "planning_figures")],
        [sys.executable, "-m", tracking_analysis.__name__, "--report", str(tracking), "--output-dir", str(root_dir / "tracking_figures")],
    ]
    modules = {name: Path(module.__file__) for name, module in (("planner", planner), ("tracker", tracker), ("planning_analysis", planning_analysis), ("tracking_analysis", tracking_analysis))}
    modules["pipeline"] = Path(__file__)
    protocol = {"root_seeds": seeds, "source_roots": roots, "excluded_source_episodes": sorted(excluded), "exclusion_bindings": bindings,
        "pilot_tracking_path": str(args.pilot_tracking.resolve()), "pilot_tracking_sha256": digest(args.pilot_tracking),
        "weights_sha256": model_sha, "adapter_sha256": digest(args.adapter_artifact), "source_sha256": source_sha,
        "feedback_policy": {"iterations": 4, "damping": .01, "first_packet_unchanged": True, "initial_plan_only": True,
            "waypoints_frozen_before_execution": True, "own_branch_observed_rgb_only": True},
        "no_outcome_parameter_selection": True, "all8_roots_and_all4_families_required": True, "commands": commands,
        "implementation_sha256": {k: digest(p) for k, p in modules.items()},
        "scope": "Fresh episode-disjoint fixed-feedback confirmation, not compute-matched mechanism efficacy or P1 admission."}
    write_new(root_dir / "pipeline.protocol.json", protocol)
    host = shared.require_cuda_allocation("cuda")
    print("Fresh all8 confirmation is PRESEALED before any new planning/outcome. All4 families retained, fixed four-iteration causal tracker, no CPU neural fallback.", flush=True)
    print(json.dumps({"roots": roots, "excluded_episodes": len(excluded)}), flush=True)
    for i, command in enumerate(commands, 1):
        print(json.dumps({"stage": i, "command": command}), flush=True)
        subprocess.run(command, check=True)
        if any(digest(p) != protocol["implementation_sha256"][k] for k, p in modules.items()):
            raise RuntimeError("presealed implementation changed during confirmation")
        write_new(root_dir / f"pipeline.stage_{i}.json", {"stage": i, "command": command, "status": "completed"})
    reports = {"planning": json.loads(reference.read_text()), "tracking": json.loads(tracking.read_text())}
    gates = {"cuda_only": True, "all_child_integrity_gates": all(r["status"] == "completed" and r["gates"] and all(v is True for v in r["gates"].values()) for r in reports.values()),
        "all8_episode_disjoint": all([r["seed"] for r in report["rows"]] == seeds and not ({r["source_episode"] for r in report["rows"]} & excluded) for report in reports.values()),
        "pilot_unchanged": digest(args.pilot_tracking) == protocol["pilot_tracking_sha256"], "original_weights_unchanged": digest(args.policy / "weights.pt") == model_sha,
        "implementation_unchanged": all(digest(p) == protocol["implementation_sha256"][k] for k, p in modules.items())}
    write_new(root_dir / "pipeline.json", {"schema": "reacher_d2p_waypoint_confirmation_pipeline.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "protocol_sha256": digest(root_dir / "pipeline.protocol.json"), "gates": gates,
        "reports": {"planning": digest(reference), "tracking": digest(tracking)}})
    print(json.dumps({"pipeline_gates": gates}), flush=True)


if __name__ == "__main__":
    main()
