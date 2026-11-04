"""Fixed-bank action candidate-source diagnostic on known unicycle dynamics."""
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


def root_context():
    return {"layout": "states_with_between_state_actions_v1", "context_length": 2,
            "latent_dim": 3, "action_dim": 2, "latent_history": [[0., 0., 0.]] * 2,
            "action_history": [[0., 0.]]}


def run_arm(index, source):
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
    if source == "spectral":
        zero = np.zeros((1, 2))
        # Generate both observable and null directions from a single Jacobian.
        jacobian = compiler._call("jacobian", root, zero)
        _, _, vh = np.linalg.svd(compiler.q @ jacobian, full_matrices=False)
        directions = [row / max(float(np.max(np.abs(row))), 1e-12) for row in vh]
        candidates = [sign * row for row in directions for sign in (-1, 1)]
    elif source == "random":
        candidates = list(np.random.default_rng(100 + index).uniform(-1., 1., (4, 2)))
    elif source == "coordinate_lookahead":
        candidates = [np.array([x, y]) for x, y in [(-1., 0.), (1., 0.), (0., -1.), (0., 1.)]]
    else:
        raise ValueError("unknown source")
    bank = [("zero", np.zeros(2)), ("greedy", np.asarray(greedy["actions"])[0])]
    bank.extend((f"proposal_{i}", np.asarray(action)) for i, action in enumerate(candidates))
    rows = []
    for name, prefix in bank:
        prefix = np.asarray(prefix, dtype=float).reshape(1, 2)
        successor = compiler._call("rollout", root, prefix)
        suffix = compiler.progress(successor, packets=1)
        actions = np.concatenate((prefix, np.asarray(suffix["actions"])))
        direct = compiler._call("rollout", root, actions)
        if digest(direct) != digest(suffix["output_context"]):
            raise ValueError("nonlinear complete action replay mismatch")
        rows.append({"name": name, "actions": actions.tolist(),
                     "prefix_cost": compiler._cost(successor),
                     "terminal_cost": compiler._cost(direct),
                     "terminal_position_error": evaluate(actions, goal)[0]})
    selected = min(range(len(rows)), key=lambda i: (rows[i]["terminal_cost"], i))
    return {"source": source, "selected_index": selected, "candidates": rows,
            "operation_counts": dict(compiler.counts), "wall_seconds": time.perf_counter() - start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    print("Diagnostic: compare equally sized prefix banks with the same nonlinear suffix optimizer.")
    rows = [{"index": i, "arms": [run_arm(i, source) for source in
             ("spectral", "random", "coordinate_lookahead")]} for i in range(8)]
    report = {"scope": "known-dynamics development; fixed candidates and suffix policy, not equal compute",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
    for i in range(3):
        errors = [row["arms"][i]["candidates"][row["arms"][i]["selected_index"]]
                  ["terminal_position_error"] for row in rows]
        print(f"{rows[0]['arms'][i]['source']}: mean position error {np.mean(errors):.9f}")


if __name__ == "__main__":
    main()
