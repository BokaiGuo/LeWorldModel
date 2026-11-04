"""Matched-call nullspace extension of the exposed exact-unicycle ablation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .d2p_atlas_nullspace_v3 import task_nullspace
from .d2p_composition_witness_v2 import digest
from .d2p_local_compiler import CompilerConfig, LocalCompiler
from .pc_atlas_matched_exact_dev_20261103_exclusive import ROOT, MeteredBackend
from .pc_exact_unicycle_development import TURN

BASE = Path(__file__).resolve().parents[2]
SOURCE = BASE / "results/pc_atlas_matched_exact_dev_20261103_exclusive.json"
SOURCE_SHA = "8d4a990529598e882f79db312106f178126e081f7dffc18c41bf93baad172c86"
PREVIOUS_SCRIPT = Path(__file__).with_name("pc_atlas_matched_exact_dev_20261103_exclusive.py")
PREVIOUS_SCRIPT_SHA = "1941819bd3e84fdda7bf074ffa7f5557394f0f97378b83b895eb3755c6e04cce"
PROTOCOL = BASE.parent / "docs/PC_ATLAS_MATCHED_NULLSPACE_DEV_PROTOCOL_20261103.md"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CachedBackend(MeteredBackend):
    def __init__(self):
        super().__init__()
        self.latest_jacobian = None

    def jacobian(self, context, actions):
        value = super().jacobian(context, actions)
        self.latest_jacobian = value.copy()
        return value


def run_arm(index, seed):
    if type(index) is not int or not 0 <= index < 8 or seed not in (42, 43, 44):
        raise ValueError("invalid frozen development roster")
    angle = 2 * np.pi * index / 64 + .031
    goal = np.array([np.cos(angle), np.sin(angle), 0.])
    backend = CachedBackend()
    compiler = LocalCompiler(backend, goal=goal, sqrt_q=np.diag([1., 1., 0.]),
                             action_scale=[1., 1.], low=[-1., -TURN], high=[1., TURN],
                             config=CompilerConfig(packet_steps=1, spectral_modes=2,
                                                   max_rollout_operations=28,
                                                   max_jacobian_operations=5))
    progress = compiler.progress(ROOT, packets=1)
    direction = task_nullspace(backend.latest_jacobian, compiler.q, compiler.scale)
    if direction.shape != (1, 2):
        raise ValueError("expected one-dimensional root nullspace")
    direction = direction[0] / np.max(np.abs(direction[0]))
    extra = [sign * compiler.config.prefix_radius * direction[None] for sign in (-1, 1)]
    bank = [np.zeros((1, 2)), np.asarray(progress["actions"]), *extra]
    candidates = []
    for prefix in bank:
        successor = compiler._call("rollout", ROOT, prefix)
        suffix = compiler.progress(successor, packets=1)
        actions = np.concatenate((prefix, np.asarray(suffix["actions"])))
        direct = compiler._call("rollout", ROOT, actions)
        if digest(direct) != digest(suffix["output_context"]):
            raise ValueError("split/direct witness mismatch")
        candidates.append({"actions": actions.tolist(), "prefix_cost": compiler._cost(successor),
                           "terminal_cost": compiler._cost(direct),
                           "terminal_error": float(np.linalg.norm(
                               np.asarray(direct["latent_history"][-1])[:2] - goal[:2])),
                           "direct_sha256": digest(direct)})
    selected = min(range(4), key=lambda j: (candidates[j]["terminal_cost"], j))
    return {"index": index, "seed": seed, "arm": "nullspace", "goal": goal.tolist(),
            "selected": selected, "candidates": candidates, "counts": compiler.counts,
            "backend_action_steps": backend.steps, "root_null_direction": direction.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha(SOURCE) != SOURCE_SHA or sha(PREVIOUS_SCRIPT) != PREVIOUS_SCRIPT_SHA:
        raise ValueError("previous ablation changed")
    source = json.loads(SOURCE.read_text())
    old = {(r["index"], r["seed"], r["arm"]): r for r in source["rows"]}
    print("Compare nullspace and existing candidates on exposed paired exact goals.", flush=True)
    rows = [run_arm(i, seed) for i in range(8) for seed in (42, 43, 44)]
    expected = {"rollout_attempted": 28, "rollout_returned": 28,
                "jacobian_attempted": 5, "jacobian_returned": 5}
    if any(r["counts"] != expected or r["backend_action_steps"] != {"rollout": 32, "jacobian": 5}
           for r in rows):
        raise RuntimeError("backend call mismatch")
    comparisons = {}
    for arm in ("random", "spectral", "ordinary"):
        differences = []
        for r in rows:
            reference = old[(r["index"], r["seed"], arm)]
            value = r["candidates"][r["selected"]]["terminal_error"]
            other = reference["candidates"][reference["selected"]]["terminal_error"]
            differences.append(value - other)
        comparisons[arm] = {"wins": sum(d < -1e-12 for d in differences),
                            "ties": sum(abs(d) <= 1e-12 for d in differences),
                            "losses": sum(d > 1e-12 for d in differences),
                            "mean_error_difference": float(np.mean(differences))}
    report = {"schema": "pc_atlas_matched_nullspace_dev.v1",
              "scope": "exposed exact-dynamics development; not total-compute matched or independent",
              "source_sha256": SOURCE_SHA, "source_script_sha256": PREVIOUS_SCRIPT_SHA,
              "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
              "nullspace_sha256": sha(Path(__file__).with_name("d2p_atlas_nullspace_v3.py")),
              "numpy_version": np.__version__,
              "mean_terminal_error": float(np.mean([r["candidates"][r["selected"]]["terminal_error"] for r in rows])),
              "selected_extra": sum(r["selected"] >= 2 for r in rows),
              "comparisons": comparisons, "rows": rows}
    with args.output.open("x") as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps({k: report[k] for k in ("mean_terminal_error", "selected_extra", "comparisons")}))


if __name__ == "__main__":
    main()
