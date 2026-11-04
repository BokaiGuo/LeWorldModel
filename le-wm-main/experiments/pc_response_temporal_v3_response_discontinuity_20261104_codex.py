"""Post-hoc audit of finite-difference blocker branches in exact Wall."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_response_temporal_v3_mechanism_attribution_20261104_codex as prior
from experiments import pc_response_temporal_v3_prospective_mechanism_20261104_codex as source

PROTOCOL = source.ROOT.parent / "docs/PC_RESPONSE_TEMPORAL_V3_RESPONSE_DISCONTINUITY_DIAG_PROTOCOL_20261104_CODEX.md"
OUTPUT = source.PREFIX.with_name(source.PREFIX.name + "_response_discontinuity.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    attribution, intervention = json.loads(prior.OUTPUT.read_text()), json.loads(source.INTERVENTION.read_text())
    if attribution["source_sha256"] != source.base.sha(source.INTERVENTION):
        raise ValueError("attribution source mismatch")
    old = {row["root"]: row for row in intervention["rows"]}
    axes = np.eye(10).reshape(10, 5, 2)
    wall = source.base.exact.wall
    rows = []
    for attribution_row in attribution["rows"]:
        root = attribution_row["root"]
        row, task = old[root], old[root]["task"]
        nominal = np.clip(np.asarray(row["selected_chain_float32"])[5:10], -.9, .9)
        controls = {}
        for name in ("selected", "zero", "greedy"):
            prefix = np.asarray(row["controls"][name]["prefix"])
            scales = {}
            for epsilon in (.01, .05, .1):
                actions = np.concatenate((nominal[None], nominal[None] + epsilon * axes,
                                          nominal[None] - epsilon * axes))
                tails = np.concatenate((actions, np.zeros((21, 5, 2))), axis=1)
                chains = np.concatenate((np.broadcast_to(prefix, (21, 5, 2)), tails), axis=1).astype(np.float32)
                padded = np.pad(chains, ((0, 0), (0, 1), (0, 0))) * wall.HIGH
                trajectory, blocked = wall.rollout(task["start"], padded)
                endpoints, branches = trajectory[:, 15, :2], np.asarray(blocked)[:, :15]
                matrix = ((endpoints[1:11] - endpoints[11:21]) / (2 * epsilon)).T
                np.testing.assert_allclose(matrix,
                    attribution_row["controls"][name]["matrices"][f"selected_suffix_clipped/{epsilon}"],
                    rtol=0, atol=1e-12)
                scales[str(epsilon)] = {"center_endpoint": endpoints[0].tolist(),
                    "blocked": branches.tolist(),
                    "pairs_with_different_blocked_sequence": int(np.any(branches[1:11] != branches[11:21], axis=1).sum()),
                    "perturbations_different_from_center": int(np.any(branches[1:] != branches[0], axis=1).sum()),
                    "matrix": matrix.tolist()}
            controls[name] = scales
        rows.append({"root": root, "controls": controls})
    source.base.write(OUTPUT, {"protocol_sha256": source.base.sha(PROTOCOL),
                              "attribution_sha256": source.base.sha(prior.OUTPUT),
                              "script_sha256": source.base.sha(Path(__file__)), "rows": rows})
    print("Saved blocker-branch traces for all six nonprogress roots.", flush=True)


if __name__ == "__main__":
    main()
