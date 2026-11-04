"""Check actual saved FD probes against nominal wall-blocking branches."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_e0_wall_independent_branch_replay_dev_20261104_codex as geometry
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = geometry.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V8_V7_E0_FD_BRANCH_STABILITY_DEV_PROTOCOL_20261104_CODEX.md"
OUT = ROOT / "results/pc_v8_v7_e0_fd_branch_stability_dev_20261104_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    sha = seal.digest
    independent_audit = geometry.PREFIX.with_name(geometry.PREFIX.name + "_audit.json")
    if (json.loads(independent_audit.read_text())["result_sha256"] != sha(geometry.OUTPUT)
            or json.loads(geometry.v8.AUDIT.read_text())["result_sha256"] != sha(geometry.v8.RESULT)
            or json.loads(geometry.panel.AUDIT.read_text())["plans_sha256"] != sha(geometry.panel.OUT)):
        raise ValueError("source audits mismatch")
    deps = seal.capture([Path(__file__), PROTOCOL, geometry.OUTPUT, independent_audit,
                         geometry.v8.RESULT, geometry.v8.AUDIT, geometry.panel.OUT,
                         geometry.panel.AUDIT, Path(geometry.__file__),
                         Path(geometry.panel.source.exact.__file__),
                         Path(geometry.panel.source.exact.wall.__file__),
                         Path(geometry.panel.source.exact.wall.physics.__file__)])
    versions = (("v8", json.loads(geometry.v8.RESULT.read_text())["rows"], 5),
                ("v7", json.loads(geometry.panel.OUT.read_text())["rows"], 0))
    rows = []
    axes = np.eye(10).reshape(10, 5, 2)
    print("Replaying the actual first and successor FD probes with independent wall geometry.", flush=True)
    for arm, plans, offset in versions:
        for row in plans:
            result = row["result"]["v7"] if arm == "v8" else row["result"]
            first, later, bank = result["first_response"], result["successor_responses"], result["first_bank"]
            if len(first) != 2 or len(later) != len(bank):
                raise ValueError("unexpected response bank")
            combined = [(info, np.empty((0, 2)), "first", i) for i, info in enumerate(first)]
            combined += [(info, np.asarray(bank[i]["prefix"]), "successor", i) for i, info in enumerate(later)]
            for position, (info, prefix, layer, index) in enumerate(combined):
                trace_index = offset + position + int(layer == "successor")
                batch = row["trace"][trace_index]
                nominal = np.asarray(info["nominal"], dtype=float)
                probes = np.concatenate((nominal[None], nominal[None]+.05*axes, nominal[None]-.05*axes))
                expected = np.concatenate((np.broadcast_to(prefix, (21, len(prefix), 2)), probes), axis=1)
                np.testing.assert_array_equal(np.asarray(batch["requested"]), expected)
                raw = np.asarray(batch["raw"], np.float32)
                np.testing.assert_array_equal(expected.astype(np.float32), raw)
                path, events = geometry.independent(row["task"]["start"], raw)
                np.testing.assert_allclose(path[:, -1, :2], batch["positions"], atol=1e-12, rtol=0)
                matrix = ((path[1:11, -1, :2] - path[11:21, -1, :2]) / .1).T
                np.testing.assert_allclose(matrix, info["matrix"], atol=1e-12, rtol=0)
                positive = np.any(events[1:11] != events[0], axis=1)
                negative = np.any(events[11:21] != events[0], axis=1)
                rows.append({"arm": arm, "root": row["root"], "layer": layer, "index": index,
                             "trace_index": trace_index, "nominal_blocked": bool(np.any(events[0])),
                             "unstable_coordinates": np.flatnonzero(positive | negative).tolist(),
                             "positive_branch_changed": int(positive.sum()),
                             "negative_branch_changed": int(negative.sum()),
                             "matrix_frobenius": float(np.linalg.norm(matrix)),
                             "rank": info["rank"], "mode_order": info["mode_order"]})
    seal.verify(deps)
    summary = {}
    for arm in ("v8", "v7"):
        for layer in ("first", "successor"):
            subset = [r for r in rows if r["arm"] == arm and r["layer"] == layer]
            summary[f"{arm}_{layer}"] = {"batches": len(subset),
                "batches_with_branch_change": sum(bool(r["unstable_coordinates"]) for r in subset),
                "coordinate_total": 10*len(subset),
                "coordinates_with_branch_change": sum(len(r["unstable_coordinates"]) for r in subset),
                "roots_with_branch_change": len({r["root"] for r in subset if r["unstable_coordinates"]}),
                "nominals_with_blocking": sum(r["nominal_blocked"] for r in subset)}
    geometry.panel.source.write(OUT, {"dependencies": deps, "rows": rows, "summary": summary,
        "scope": "actual saved .05 FD probes; exposed E0; same analytic wall family"})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
