"""Run the predeclared v3 nullspace search on exposed analytical goals."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .d2p_atlas_nullspace_v3 import NullspaceAtlasSearch
from .d2p_atlas_search_v2 import AtlasConfig
from .d2p_local_compiler import CompilerConfig, LocalCompiler
from .pc_atlas_unicycle_smoke_20261103 import UnicycleBackend
from .pc_exact_unicycle_development import TURN, evaluate
from .pc_exact_unicycle_strong_baseline import multistart_gradient

ROOT = {"layout": "states_with_between_state_actions_v1", "context_length": 2,
        "latent_dim": 3, "action_dim": 2, "latent_history": [[0., 0., 0.]] * 2,
        "action_history": [[0., 0.]]}
BASE = Path(__file__).resolve().parents[2]
PROTOCOL = BASE.parent / "docs/PC_ATLAS_NULLSPACE_DEV_PROTOCOL_20261103.md"


def run_goal(index):
    if type(index) is not int or not 0 <= index < 8:
        raise ValueError("only eight exposed goals in the development protocol")
    angle = 2 * np.pi * index / 64 + .031
    goal = np.array([np.cos(angle), np.sin(angle)])
    compiler = LocalCompiler(UnicycleBackend(), goal=[*goal, 0.], sqrt_q=np.diag([1., 1., 0.]),
                             action_scale=[1., 1.], low=[-1., -TURN], high=[1., TURN],
                             config=CompilerConfig(packet_steps=1, spectral_modes=2,
                                                   max_rollout_operations=1000,
                                                   max_jacobian_operations=100))
    atlas = NullspaceAtlasSearch(compiler, AtlasConfig(seed=42 + index)).search(ROOT)
    selected = atlas["programs"][atlas["selected_index"]]
    error, trajectory = evaluate(np.asarray(selected["actions"]), goal)
    gradient, ledger = multistart_gradient(goal)
    baseline_error, _ = evaluate(gradient, goal)
    return {"index": index, "goal": goal.tolist(), "atlas_error": error,
            "selected_trajectory": trajectory.tolist(), "gradient_error": baseline_error,
            "gradient_ledger": ledger, "atlas": atlas}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    print("Run v3 on eight exposed exact-dynamics goals; no equal-budget or physical claim.", flush=True)
    rows = [run_goal(i) for i in range(8)]
    dependencies = [Path(__file__), PROTOCOL, Path(__file__).with_name("d2p_atlas_nullspace_v3.py"),
                    Path(__file__).with_name("d2p_atlas_search_v2.py"),
                    Path(__file__).with_name("d2p_local_compiler.py"),
                    Path(__file__).with_name("pc_atlas_unicycle_smoke_20261103.py"),
                    Path(__file__).with_name("pc_exact_unicycle_development.py"),
                    Path(__file__).with_name("pc_exact_unicycle_strong_baseline.py")]
    report = {"schema": "pc_atlas_nullspace_unicycle_dev.v1",
              "scope": "exposed exact-dynamics development, not equal-budget or physical",
              "dependency_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies},
              "rows": rows}
    with args.output.open("x") as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps({"roots": len(rows),
                      "atlas_mean_error": float(np.mean([r["atlas_error"] for r in rows])),
                      "gradient_mean_error": float(np.mean([r["gradient_error"] for r in rows])),
                      "qualified": sum(r["atlas"]["qualified_reconfiguration_index"] is not None for r in rows)}))


if __name__ == "__main__":
    main()
