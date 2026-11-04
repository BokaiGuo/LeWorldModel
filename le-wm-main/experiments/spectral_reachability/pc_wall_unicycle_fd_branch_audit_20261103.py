"""Replay Atlas and trace finite-difference collision branches."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import pc_wall_unicycle_atlas_cem_20261103 as wall

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/pc_wall_unicycle_atlas_cem_20261103_DEV01.json"
EXPECTED = "7f7136d1f4192f79e680d10ef79fb06ee76dd5390daef87fc5227a40a85a122d"
PROTOCOL = ROOT.parent / "docs/PC_WALL_UNICYCLE_FD_BRANCH_AUDIT_PROTOCOL_20261103.md"


class TracingBackend(wall.Backend):
    def __init__(self):
        super().__init__()
        self.in_fd = False
        self.current = None
        self.fd_paths = []
        self.fd_audit = []

    def advance(self, state, action):
        result, blocked = super().advance(state, action)
        if self.in_fd:
            self.current.append(blocked)
        return result, blocked

    def _roll(self, context, actions):
        if not self.in_fd:
            return super()._roll(context, actions)
        self.current = []
        result = super()._roll(context, actions)
        self.fd_paths.append(self.current)
        self.current = None
        return result

    def jacobian(self, context, actions):
        self.in_fd = True
        self.fd_paths = []
        try:
            result = super().jacobian(context, actions)
            if len(self.fd_paths) != 2 * actions.size:
                raise ValueError("finite-difference path count")
            differences = [self.fd_paths[2*i] != self.fd_paths[2*i+1] for i in range(actions.size)]
            self.fd_audit.append({"pair_count": actions.size, "branch_different": sum(differences),
                                  "different_coordinates": [i for i, x in enumerate(differences) if x],
                                  "max_abs_jacobian": float(np.max(np.abs(result)))})
            return result
        finally:
            self.in_fd = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(SOURCE) != EXPECTED:
        raise ValueError("source receipt changed")
    source = json.loads(SOURCE.read_text())
    old_backend = wall.Backend
    rows = []
    print("Trace collision-branch differences in the saved Atlas finite-difference searches.", flush=True)
    try:
        wall.Backend = TracingBackend
        for reference in source["rows"]:
            result = wall.run_atlas(reference["seed"])
            expected = reference["atlas"]
            if (result["selected"] != expected["selected"]
                    or result["compiler_counts"] != expected["compiler_counts"]):
                raise ValueError("saved search did not replay")
            backend = result["backend"]
            for name in ("transitions", "rollouts", "fd_rollouts", "jacobians"):
                if backend[name] != expected["backend"][name]:
                    raise ValueError("saved cost did not replay")
            details = backend["fd_audit"]
            if len(details) != backend["jacobians"]:
                raise ValueError("incomplete Jacobian trace")
            rows.append({"seed": reference["seed"], "jacobians": len(details),
                         "coordinate_pairs": sum(r["pair_count"] for r in details),
                         "differing_pairs": sum(r["branch_different"] for r in details),
                         "jacobians_with_difference": sum(r["branch_different"] > 0 for r in details),
                         "max_abs_jacobian": max(r["max_abs_jacobian"] for r in details),
                         "details": details, "replay_backend_transitions": backend["transitions"]})
            print(f"seed={reference['seed']} differing pairs={rows[-1]['differing_pairs']}", flush=True)
    finally:
        wall.Backend = old_backend
    report = {"schema": "wall_unicycle_fd_branch_audit.v1", "source_sha256": sha(SOURCE),
              "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
              "backend_sha256": sha(Path(wall.__file__)), "rows": rows,
              "scope": "replayed single analytical development task; collision branch audit only"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
