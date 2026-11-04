"""Privileged post-hoc injection into frozen v23 candidate prefixes."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_e0_wall_independent_branch_replay_dev_20261104_codex as geometry
from experiments import pc_v23_shared_cem_e0_dev_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = source.ROOT / "results/pc_v23_baseline_prefix_injection_posthoc_20261104_codex.json"
PROTOCOL = source.ROOT.parent / "docs/PC_V23_BASELINE_PREFIX_INJECTION_POSTHOC_PROTOCOL_20261104_CODEX.md"


def measure(start, goal, reference, layer):
    reference = np.asarray(reference, np.float32)
    length = (int(layer["depth"]) + 1) * 5
    prefixes = np.asarray([item["chain"] for item in layer["candidates"]], np.float32)
    retained = np.asarray(layer["retained"], dtype=np.int64)
    if (reference.shape != (15, 2) or prefixes.ndim != 3 or prefixes.shape[1:] != (length, 2)
            or len(prefixes) == 0 or len(retained) == 0
            or len(set(retained.tolist())) != len(retained)
            or np.any(retained < 0) or np.any(retained >= len(prefixes))
            or not np.isfinite(prefixes).all() or np.abs(prefixes).max() > 1):
        raise ValueError("invalid saved prefix bank")
    suffix = np.broadcast_to(reference[length:], (len(prefixes), 15 - length, 2))
    chains = np.concatenate((prefixes, suffix), axis=1)
    path, _ = geometry.independent(start, chains)
    reference_path, _ = geometry.independent(start, reference[None])
    scores = np.linalg.norm(path[:, -1, :2] - goal, axis=1)
    action_rms = np.sqrt(np.mean((prefixes - reference[:length]) ** 2, axis=(1, 2)))
    successor_distance = np.linalg.norm(path[:, length, :2] - reference_path[0, length, :2], axis=1)
    heading_delta = np.abs((path[:, length, 2] - reference_path[0, length, 2] + np.pi)
                           % (2 * np.pi) - np.pi)

    def summarize(indices):
        return {"count": len(indices), "best_injected_score": float(scores[indices].min()),
                "injected_success_count": int(np.sum(scores[indices] < 0.05)),
                "nearest_action_rms": float(action_rms[indices].min()),
                "nearest_successor_position": float(successor_distance[indices].min()),
                "nearest_successor_heading_rad": float(heading_delta[indices].min())}

    return {"prefix_steps": length, "all": summarize(np.arange(len(prefixes))),
            "retained": summarize(retained)}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest_path, plans_path, audit_path = (source.OUT / name for name in
                                             ("manifest.json", "plans.json", "audit.json"))
    manifest, plans, audit = (json.loads(path.read_text()) for path in
                              (manifest_path, plans_path, audit_path))
    seal.verify(manifest["dependencies"])
    if (audit["status"] != "passed" or audit["plans_sha256"] != seal.digest(plans_path)
            or plans["manifest_sha256"] != seal.digest(manifest_path)
            or len(plans["rows"]) != 16):
        raise ValueError("source audit or manifest binding mismatch")
    dependencies = seal.capture([Path(__file__), PROTOCOL, manifest_path, plans_path, audit_path,
                                 Path(geometry.__file__), *[source.OUT / row["trace"]
                                                            for row in plans["rows"]]])
    rows = []
    for root in range(8):
        records = {row["arm"]: row for row in plans["rows"] if row["root"] == root}
        if set(records) != set(source.ARMS):
            raise ValueError("missing or duplicated arm")
        response, control = records["v23"], records["shared_cem_v23"]
        if response["task"] != control["task"]:
            raise ValueError("unpaired tasks")
        for row in (response, control):
            if seal.digest(source.OUT / row["trace"]) != row["trace_sha256"]:
                raise ValueError("source trace drift")
        if response["status"] != "completed" or control["status"] != "completed":
            rows.append({"root": root, "status": "source_failure"})
            continue
        task = response["task"]
        reference = np.asarray(control["result"]["result"]["selected"], np.float32)
        path, _ = geometry.independent(task["start"], reference[None])
        reference_score = float(np.linalg.norm(path[0, -1, :2] - task["goal"]))
        np.testing.assert_allclose(reference_score, control["result"]["result"]["selected_score"],
                                   atol=1e-12, rtol=0)
        layers = response["result"]["result"]["child"]["layers"]
        if len(layers) != 3:
            raise ValueError("unexpected search depth")
        rows.append({"root": root, "status": "completed", "reference_score": reference_score,
                     "reference_success": reference_score < 0.05,
                     "layers": [measure(task["start"], task["goal"], reference, layer)
                                for layer in layers[:2]]})
    seal.verify(manifest["dependencies"])
    seal.verify(dependencies)
    output = {"scope": "posthoc privileged diagnostic, not planner or gate evidence",
              "dependencies": dependencies, "rows": rows}
    with OUT.open("x") as handle:
        json.dump(output, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
