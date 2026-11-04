"""Fixed turn/translation grid on exposed exact-dynamics development goals."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .d2p_composition_witness_v2 import digest
from .d2p_local_compiler import CompilerConfig, LocalCompiler
from .pc_atlas_matched_exact_dev_20261103_exclusive import ROOT, MeteredBackend
from .pc_exact_unicycle_development import TURN

BASE = Path(__file__).resolve().parents[2]
PROTOCOL = BASE.parent / "docs/PC_ATLAS_TURN_AMPLITUDE_GRID_PROTOCOL_20261103.md"
TURNS = (-TURN, -.5, 0., .5, TURN)


def run_goal(index):
    if type(index) is not int or not 0 <= index < 8:
        raise ValueError("invalid development goal index")
    angle = 2 * np.pi * index / 64 + .031
    goal = np.array([np.cos(angle), np.sin(angle), 0.])
    backend = MeteredBackend()
    c = LocalCompiler(backend, goal=goal, sqrt_q=np.diag([1., 1., 0.]),
                      action_scale=[1., 1.], low=[-1., -TURN], high=[1., TURN],
                      config=CompilerConfig(packet_steps=1, spectral_modes=2,
                                            max_rollout_operations=64,
                                            max_jacobian_operations=11))
    progress = c.progress(ROOT, packets=1)
    v = float(progress["actions"][0][0])
    rows = []
    for kind, velocity in (("stationary", 0.), ("progress_velocity", v)):
        for turn in TURNS:
            prefix = np.array([[velocity, turn]])
            successor = c._call("rollout", ROOT, prefix)
            suffix = c.progress(successor, packets=1)
            actions = np.concatenate((prefix, np.asarray(suffix["actions"])))
            direct = c._call("rollout", ROOT, actions)
            if digest(direct) != digest(suffix["output_context"]):
                raise ValueError("split/full witness mismatch")
            rows.append({"velocity_kind": kind, "turn": turn, "actions": actions.tolist(),
                         "prefix_cost": c._cost(successor), "terminal_cost": c._cost(direct),
                         "terminal_error": float(np.linalg.norm(
                             np.asarray(direct["latent_history"][-1])[:2] - goal[:2])),
                         "witness_sha256": digest(direct)})
    if c.counts != {"rollout_attempted": 64, "rollout_returned": 64,
                    "jacobian_attempted": 11, "jacobian_returned": 11}:
        raise RuntimeError("grid budget mismatch")
    if backend.steps != {"rollout": 74, "jacobian": 11}:
        raise RuntimeError("grid action-step budget mismatch")
    return {"index": index, "goal": goal.tolist(), "progress_velocity": v,
            "candidates": rows, "counts": c.counts, "backend_action_steps": backend.steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    print("Compare fixed prefix turn amplitudes and translation on exposed goals.", flush=True)
    rows = [run_goal(i) for i in range(8)]
    summary = {}
    for kind in ("stationary", "progress_velocity"):
        for amplitude, allowed in (("half", [-.5, .5]), ("max", [-TURN, TURN]),
                                   ("both", [-TURN, -.5, .5, TURN])):
            chosen = [min((c["terminal_error"] for c in r["candidates"]
                           if c["velocity_kind"] == kind and c["turn"] in allowed))
                      for r in rows]
            summary[f"{kind}_{amplitude}"] = float(np.mean(chosen))
    report = {"schema": "pc_atlas_turn_amplitude_grid.v1",
              "scope": "exposed exact-dynamics diagnostic, not independent or real control",
              "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "summary_best_sign_error": summary, "rows": rows}
    with args.output.open("x") as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
