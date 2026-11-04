"""Two-step finite-horizon basis ablation on exposed exact-unicycle goals."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .d2p_composition_witness_v2 import digest
from .d2p_local_compiler import CompilerConfig, LocalCompiler
from .pc_atlas_unicycle_smoke_20261103 import UnicycleBackend
from .pc_exact_unicycle_development import TURN
from .pc_spectral_candidate_ablation_20261103 import root_context

REFERENCE = np.array([[0., .4], [.6, 0.]])
SEEDS = (101, 202, 303, 404)
SOURCES = ("raw_isotropic", "spectral_isotropic", "spectral_anisotropic")


def samples(seed, count=16):
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(count, 4))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return .35 * direction * rng.uniform(size=(count, 1)) ** .25


def run_arm(index, seed, source):
    if index not in range(8) or seed not in SEEDS or source not in SOURCES:
        raise ValueError("outside fixed protocol")
    angle = 2 * np.pi * index / 64 + .031
    goal = np.array([np.cos(angle), np.sin(angle), 0.])
    root = root_context()
    compiler = LocalCompiler(
        UnicycleBackend(), goal=goal, sqrt_q=np.diag([1., 1., 0.]),
        action_scale=[1., 1.], low=[-1., -TURN], high=[1., TURN],
        config=CompilerConfig(packet_steps=1, suffix_packets=1, spectral_modes=2,
                              max_rollout_operations=1000, max_jacobian_operations=100))
    start = time.perf_counter()
    jacobian = compiler._call("jacobian", root, REFERENCE)
    _, singular, basis = np.linalg.svd(compiler.q @ jacobian, full_matrices=True)
    coeff = samples(seed)
    if source == "raw_isotropic":
        offsets = coeff
    else:
        weights = np.array([1., 1., .25, .25]) if source == "spectral_anisotropic" else np.ones(4)
        offsets = (coeff * weights) @ basis
    proposals = REFERENCE + offsets.reshape(-1, 2, 2)
    if np.any(proposals < compiler.low) or np.any(proposals > compiler.high):
        raise ValueError("candidate violates physical action bounds")
    greedy = np.asarray(compiler.progress(root, packets=2)["actions"])
    bank = [("zero", np.zeros((2, 2))), ("greedy", greedy)]
    bank.extend((f"proposal_{i}", action) for i, action in enumerate(proposals))
    rows = []
    for name, prefix in bank:
        successor = compiler._call("rollout", root, prefix)
        suffix = compiler.progress(successor, packets=1)
        actions = np.concatenate((prefix, np.asarray(suffix["actions"])))
        direct = compiler._call("rollout", root, actions)
        if digest(direct) != digest(suffix["output_context"]):
            raise ValueError("split/direct replay mismatch")
        position = np.asarray(direct["latent_history"][-1])[:2]
        rows.append({"name": name, "actions": actions.tolist(),
                     "prefix_cost": compiler._cost(successor),
                     "terminal_cost": compiler._cost(direct),
                     "terminal_error": float(np.linalg.norm(position - goal[:2]))})
    selected = min(range(len(rows)), key=lambda i: (rows[i]["terminal_cost"], i))
    return {"source": source, "basis": basis.tolist(), "singular_values": singular.tolist(),
            "selected_index": selected, "candidates": rows,
            "operation_counts": dict(compiler.counts),
            "wall_seconds": time.perf_counter() - start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    print("Diagnostic: two-step rotated action basis and matched suffix planning.", flush=True)
    rows = [{"index": index, "seed": seed,
             "arms": [run_arm(index, seed, source) for source in SOURCES]}
            for index in range(8) for seed in SEEDS]
    path = Path(__file__)
    files = [path, path.with_name("d2p_local_compiler.py"),
             path.with_name("pc_atlas_unicycle_smoke_20261103.py"),
             path.with_name("pc_exact_unicycle_development.py"),
             path.parents[3] / "docs/PC_ROTATED_SPECTRAL_BASIS_DEV_PROTOCOL_20261103.md"]
    report = {"scope": "exposed exact dynamics; not independent, learned, or real physics",
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    for i, source in enumerate(SOURCES):
        errors = [r["arms"][i]["candidates"][r["arms"][i]["selected_index"]]
                  ["terminal_error"] for r in rows]
        print(f"{source}: mean error {np.mean(errors):.9f}", flush=True)


if __name__ == "__main__":
    main()
