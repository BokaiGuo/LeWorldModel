"""Matched-call Atlas-style candidate ablation on exposed exact unicycle goals."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .d2p_composition_witness_v2 import digest
from .d2p_local_compiler import CompilerConfig, LocalCompiler
from .pc_atlas_unicycle_smoke_20261103 import UnicycleBackend
from .pc_exact_unicycle_development import TURN


ARMS = ("random", "spectral", "ordinary")
ROOT = {"layout": "states_with_between_state_actions_v1", "context_length": 2,
        "latent_dim": 3, "action_dim": 2, "latent_history": [[0., 0., 0.]] * 2,
        "action_history": [[0., 0.]]}


class MeteredBackend(UnicycleBackend):
    def __init__(self):
        self.steps = {"rollout": 0, "jacobian": 0}

    def rollout(self, context, actions):
        self.steps["rollout"] += len(actions)
        return super().rollout(context, actions)

    def jacobian(self, context, actions):
        self.steps["jacobian"] += len(actions)
        return super().jacobian(context, actions)


def run_arm(index, seed, arm):
    if arm not in ARMS or type(index) is not int or not 0 <= index < 64 or type(seed) is not int or seed < 0:
        raise ValueError("invalid arm, panel index, or seed")
    angle = 2 * np.pi * index / 64 + .031
    goal = np.array([np.cos(angle), np.sin(angle), 0.])
    backend = MeteredBackend()
    compiler = LocalCompiler(backend, goal=goal, sqrt_q=np.diag([1., 1., 0.]),
                             action_scale=[1., 1.], low=[-1., -TURN], high=[1., TURN],
                             config=CompilerConfig(packet_steps=1, spectral_modes=2,
                                                   max_rollout_operations=28,
                                                   max_jacobian_operations=5))
    progress = compiler.progress(ROOT, packets=1)
    p = np.asarray(progress["actions"])
    if arm == "random":
        rng = np.random.default_rng(seed)
        extra = [rng.uniform(compiler.low, compiler.high, size=(1, 2)) for _ in range(2)]
    elif arm == "spectral":
        directions = progress["geometry"]["raw_spectral_directions"]
        if not directions:
            raise ValueError("spectral rank zero; matched design is undefined")
        d = np.asarray(directions[0]).reshape(1, 2)
        d = d / np.max(np.abs(d))
        extra = [-d, d]
    else:
        # Ordinary lookahead tests fixed steering alternatives with the same
        # velocity as the shared greedy candidate; no spectral proposal.
        extra = [np.array([[p[0, 0], sign * TURN]]) for sign in (-1, 1)]
    bank = [np.zeros((1, 2)), p, *extra]
    candidates = []
    for prefix in bank:
        successor = compiler._call("rollout", ROOT, prefix)
        suffix = compiler.progress(successor, packets=1)
        actions = np.concatenate((prefix, np.asarray(suffix["actions"])))
        direct = compiler._call("rollout", ROOT, actions)
        if digest(direct) != digest(suffix["output_context"]):
            raise ValueError("split/direct mismatch")
        candidates.append({"actions": actions.tolist(), "prefix_cost": compiler._cost(successor),
                           "terminal_cost": compiler._cost(direct),
                           "terminal_error": float(np.linalg.norm(
                               np.asarray(direct["latent_history"][-1])[:2] - goal[:2])),
                           "direct_sha256": digest(direct)})
    selected = min(range(4), key=lambda j: (candidates[j]["terminal_cost"], j))
    return {"index": index, "seed": seed, "arm": arm, "goal": goal.tolist(),
            "selected": selected, "candidates": candidates, "counts": compiler.counts,
            "backend_action_steps": backend.steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    print("Exposed exact-dynamics development: matched-call candidate ablation, not independent confirmation.")
    rows = [run_arm(i, seed, arm) for i in range(8) for seed in (42, 43, 44) for arm in ARMS]
    expected = {"rollout_attempted": 28, "rollout_returned": 28,
                "jacobian_attempted": 5, "jacobian_returned": 5}
    if any(r["counts"] != expected or r["backend_action_steps"] != {"rollout": 32, "jacobian": 5} for r in rows):
        raise RuntimeError("matched budget or action-step invariant violated")
    summary = {arm: {"mean_terminal_error": float(np.mean([
        r["candidates"][r["selected"]]["terminal_error"] for r in rows if r["arm"] == arm])),
        "selected_extra": sum(r["selected"] >= 2 for r in rows if r["arm"] == arm)}
        for arm in ARMS}
    files = [Path(__file__), Path(__file__).with_name("d2p_local_compiler.py"),
             Path(__file__).with_name("pc_atlas_unicycle_smoke_20261103.py"),
             Path(__file__).with_name("pc_exact_unicycle_development.py"),
             Path(__file__).with_name("d2p_composition_witness_v2.py")]
    report = {"schema": "atlas_matched_exact_dev_exclusive.v1",
              "scope": "exposed analytical development; not independent, physical, neural, or full Atlas",
              "design": {"indices": list(range(8)), "seeds": [42, 43, 44],
                         "arms": list(ARMS), "prefixes": 4, "suffix_packets": 1,
                         "counts_per_arm": expected, "steps_per_arm": {"rollout": 32, "jacobian": 5}},
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              "numpy_version": np.__version__, "summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
