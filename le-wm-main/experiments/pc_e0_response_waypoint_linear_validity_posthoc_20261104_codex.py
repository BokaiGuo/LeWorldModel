"""Evaluate saved finite-difference predictions on actual optimized prefixes."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_e0_response_waypoint_direction_dev_20261104_codex as source
from experiments.pc_e0_wall_independent_branch_replay_dev_20261104_codex import independent
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

PROTOCOL = source.ROOT.parent / "docs/PC_E0_RESPONSE_WAYPOINT_LINEAR_VALIDITY_POSTHOC_PROTOCOL_20261104_CODEX.md"
OUT = source.OUT / "linear_validity_posthoc.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    result_path, audit_path = source.OUT / "result.json", source.OUT / "audit.json"
    structure_path = source.OUT / "direction_structure_audit.json"
    geometry_path = source.OUT / "independent_geometry_replay.json"
    result = json.loads(result_path.read_text())
    for path in (audit_path, structure_path, geometry_path):
        check = json.loads(path.read_text())
        if check["status"] != "passed" or check["result_sha256"] != source.prior.previous.sha(result_path):
            raise ValueError("source audit mismatch")
    deps = seal.capture([Path(__file__), PROTOCOL, result_path, audit_path, structure_path, geometry_path,
                         Path(independent.__code__.co_filename), Path(source.prior.previous.exact.__file__)])
    rows = []
    for row in result["rows"]:
        if not row["arm"].endswith("_waypoint"):
            continue
        if row["status"] != "completed":
            raise ValueError("source plan failed")
        goal = np.asarray(row["task"]["goal"])
        with np.load(source.OUT / row["trace_file"], allow_pickle=False) as data:
            probes, nominal_end = data["actions_2"], data["positions_2"][0]
        nominal = probes[0]
        matrix = np.asarray(row["result"]["response"]["matrix"])
        _, nominal_blocked = independent(row["task"]["start"], nominal[None])
        for index, branch in enumerate(row["result"]["waypoints"]):
            prefix = np.asarray(branch["prefix"], np.float32)
            actual = source.prior.previous.exact.endpoints(row["task"], prefix[None])[0]
            path, blocked = independent(row["task"]["start"], prefix[None])
            np.testing.assert_allclose(actual, path[0, -1, :2], atol=1e-12, rtol=0)
            predicted = nominal_end + matrix @ (prefix.astype(float) - nominal.astype(float)).reshape(-1)
            error = float(np.linalg.norm(predicted - actual))
            displacement = float(np.linalg.norm(actual - nominal_end))
            actual_delta = float(np.linalg.norm(actual - goal) - np.linalg.norm(nominal_end - goal))
            predicted_delta = float(np.linalg.norm(predicted - goal) - np.linalg.norm(nominal_end - goal))
            sign_match = None if abs(actual_delta) <= 1e-10 or abs(predicted_delta) <= 1e-10 else bool(
                np.sign(actual_delta) == np.sign(predicted_delta))
            rows.append({"root": row["root"], "arm": row["arm"], "branch": index,
                         "prefix": prefix.tolist(), "nominal_endpoint": nominal_end.tolist(),
                         "actual_endpoint": actual.tolist(), "linear_endpoint": predicted.tolist(),
                         "endpoint_error": error, "actual_displacement": displacement,
                         "relative_error": error / displacement if displacement > 1e-10 else None,
                         "actual_goal_cost_change": actual_delta, "predicted_goal_cost_change": predicted_delta,
                         "goal_change_sign_match": sign_match,
                         "same_blocked_sequence": bool(np.array_equal(blocked, nominal_blocked))})
    if len(rows) != 48:
        raise ValueError("unexpected comparison count")
    summaries = {}
    for arm in ("response_waypoint", "random_waypoint", "geometric_waypoint"):
        selected = [r for r in rows if r["arm"] == arm]
        defined = [r["goal_change_sign_match"] for r in selected if r["goal_change_sign_match"] is not None]
        summaries[arm] = {"count": len(selected),
            "median_endpoint_error": float(np.median([r["endpoint_error"] for r in selected])),
            "median_relative_error": float(np.median([r["relative_error"] for r in selected if r["relative_error"] is not None])),
            "goal_sign_agree": sum(defined), "goal_sign_defined": len(defined),
            "different_blocked_sequence": sum(not r["same_blocked_sequence"] for r in selected)}
    seal.verify(deps)
    source.prior.previous.write(OUT, {"status": "complete", "dependencies": deps, "rows": rows,
        "summary": summaries, "scope": "post-hoc actual-prefix linearity, same exact system"})
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
