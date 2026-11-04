"""Mixed-action, matched-distribution candidate-source development ablation."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .d2p_composition_witness_v2 import digest
from .d2p_local_compiler import CompilerConfig, LocalCompiler
from .pc_atlas_unicycle_smoke_20261103 import UnicycleBackend
from .pc_exact_unicycle_development import TURN, evaluate
from .pc_spectral_candidate_ablation_20261103 import root_context

SEEDS = (101, 202, 303, 404)
SOURCES = ("raw_isotropic", "spectral_isotropic", "spectral_anisotropic")


def disk_coefficients(seed, count=16):
    rng = np.random.default_rng(seed)
    radius = np.sqrt(rng.uniform(0., 1., count))
    angle = rng.uniform(-np.pi, np.pi, count)
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))


def run_arm(index, seed, source):
    if index not in range(8) or seed not in SEEDS or source not in SOURCES:
        raise ValueError("outside fixed development protocol")
    angle = 2 * np.pi * index / 64 + .031
    goal = np.array([np.cos(angle), np.sin(angle)])
    root = root_context()
    compiler = LocalCompiler(
        UnicycleBackend(), goal=[*goal, 0.], sqrt_q=np.diag([1., 1., 0.]),
        action_scale=[1., 1.], low=[-1., -TURN], high=[1., TURN],
        config=CompilerConfig(packet_steps=1, suffix_packets=1, spectral_modes=2,
                              max_rollout_operations=1000, max_jacobian_operations=100))
    start = time.perf_counter()
    greedy = compiler.progress(root, packets=1)
    coefficients = disk_coefficients(seed)
    basis = np.eye(2)
    if source != "raw_isotropic":
        jacobian = compiler._call("jacobian", root, np.zeros((1, 2)))
        _, _, basis = np.linalg.svd(compiler.q @ jacobian, full_matrices=False)
    weights = np.array([1., .25]) if source == "spectral_anisotropic" else np.ones(2)
    candidates = (coefficients * weights) @ basis
    bank = [("zero", np.zeros(2)), ("greedy", np.asarray(greedy["actions"])[0])]
    bank.extend((f"proposal_{i}", action) for i, action in enumerate(candidates))
    rows = []
    for name, prefix in bank:
        prefix = np.asarray(prefix, dtype=float).reshape(1, 2)
        successor = compiler._call("rollout", root, prefix)
        suffix = compiler.progress(successor, packets=1)
        actions = np.concatenate((prefix, np.asarray(suffix["actions"])))
        direct = compiler._call("rollout", root, actions)
        if digest(direct) != digest(suffix["output_context"]):
            raise ValueError("direct and split rollout differ")
        rows.append({"name": name, "actions": actions.tolist(),
                     "prefix_cost": compiler._cost(successor),
                     "terminal_cost": compiler._cost(direct),
                     "terminal_position_error": evaluate(actions, goal)[0]})
    selected = min(range(len(rows)), key=lambda i: (rows[i]["terminal_cost"], i))
    return {"source": source, "seed": seed, "basis": basis.tolist(),
            "coefficient_weights": weights.tolist(), "selected_index": selected,
            "candidates": rows, "operation_counts": dict(compiler.counts),
            "wall_seconds": time.perf_counter() - start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    print("Development diagnostic: matched mixed-action sampling, same suffix optimizer.", flush=True)
    rows = [{"index": index, "seed": seed,
             "arms": [run_arm(index, seed, source) for source in SOURCES]}
            for index in range(8) for seed in SEEDS]
    source_path = Path(__file__)
    report = {"scope": "exposed analytical cases; mixed action development, not independent/equal compute",
              "protocol_sha256": hashlib.sha256(
                  (source_path.parents[3] / "docs/PC_SPECTRAL_MIXED_BASIS_DEV_PROTOCOL_20261103.md")
                  .read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    for index, source in enumerate(SOURCES):
        errors = [r["arms"][index]["candidates"][r["arms"][index]["selected_index"]]
                  ["terminal_position_error"] for r in rows]
        print(f"{source}: mean position error {np.mean(errors):.9f}", flush=True)


if __name__ == "__main__":
    main()
