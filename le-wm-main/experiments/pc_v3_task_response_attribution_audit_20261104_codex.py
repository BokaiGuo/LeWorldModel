"""Recompute saved tail-local task-attribution probes with exact dynamics."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v3_task_response_attribution_20261104_codex as analysis

source = analysis.source
OUTPUT = analysis.OUTPUT.with_name(analysis.OUTPUT.stem + "_audit.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = source.prior.base.sha
    stored = json.loads(source.INTERVENTION.read_text())
    result = json.loads(analysis.OUTPUT.read_text())
    if (result["source_sha256"] != sha(source.INTERVENTION)
            or result["protocol_sha256"] != sha(analysis.PROTOCOL)
            or result["script_sha256"] != sha(Path(analysis.__file__))):
        raise ValueError("attribution seal mismatch")
    axes = np.eye(10).reshape(10, 5, 2)
    wall = source.prior.base.exact.wall
    max_error = 0.0
    count = 0
    for row, output in zip(stored["rows"], result["rows"], strict=True):
        if row["root"] != output["root"]:
            raise ValueError("root mismatch")
        for eps in (.01, .05):
            for name in analysis.NAMES:
                control = row["controls"][name]
                tail = np.broadcast_to(np.asarray(row["nominal_tail"]), (21, 10, 2)).copy()
                tail[1:11, :5] += eps * axes
                tail[11:21, :5] -= eps * axes
                joined = np.concatenate((np.broadcast_to(control["prefix"], (21, 5, 2)), tail), axis=1).astype(np.float32)
                endpoints = source.prior.base.exact.endpoints(row["task"], joined)
                recorded = np.asarray(control["tail_local_response"][str(eps)]["center_and_probe_endpoints"])
                max_error = max(max_error, float(np.max(np.abs(endpoints - recorded))))
                np.testing.assert_allclose(endpoints, recorded, rtol=0, atol=1e-12)
                trajectory, blocked = wall.rollout(row["task"]["start"],
                    np.pad(joined, ((0, 0), (0, 1), (0, 0))) * wall.HIGH)
                np.testing.assert_allclose(trajectory[:, 15, :2], endpoints, rtol=0, atol=1e-12)
                expected = control["tail_local_response"][str(eps)]
                if np.asarray(blocked)[:, :15].tolist() != expected["blocked"]:
                    raise ValueError("blocked branch mismatch")
                d = np.linalg.norm(endpoints - np.asarray(row["task"]["goal"]), axis=1)
                fd = (d[1:11] - d[11:21]) / (2 * eps)
                entry = output["scales"][str(eps)]["controls"][name]
                np.testing.assert_allclose(fd, entry["distance_probe_fd"], rtol=0, atol=1e-12)
                count += len(joined)
    if len(result["rows"]) != 24 or count != 5040:
        raise ValueError("incomplete audit")
    source.prior.base.write(OUTPUT, {"result_sha256": sha(analysis.OUTPUT), "recomputed_chains": count,
                                     "max_endpoint_abs_error": max_error,
                                     "scope": "same exact dynamics and wall backend"})
    print(f"Recomputed {count} exact trajectories and blocked branches.", flush=True)


if __name__ == "__main__":
    main()
