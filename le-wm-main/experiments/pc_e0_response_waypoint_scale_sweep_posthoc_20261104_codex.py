"""Fixed multi-scale validity sweep on sealed optimized action directions."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_e0_response_waypoint_direction_dev_20261104_codex as source
from experiments.pc_e0_wall_independent_branch_replay_dev_20261104_codex import independent
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

PROTOCOL = source.ROOT.parent / "docs/PC_E0_RESPONSE_WAYPOINT_SCALE_SWEEP_POSTHOC_PROTOCOL_20261104_CODEX.md"
OUT = source.OUT / "scale_sweep_posthoc.json"
SCALES = (0., .05, .1, .25, .5, 1.)


def fixed_branch_jacobian(nominal, path, blocked):
    jac = np.zeros((3, nominal.size), dtype=np.float64)
    for t, action in enumerate(nominal):
        angle, speed = path[t, 2], .3 * action[0]
        if not blocked[t]:
            heading_gradient = jac[2].copy()
            jac[0] += -speed * np.sin(angle) * heading_gradient
            jac[1] += speed * np.cos(angle) * heading_gradient
            jac[0, 2*t] += .3 * np.cos(angle)
            jac[1, 2*t] += .3 * np.sin(angle)
        jac[2, 2*t+1] += np.pi / 2
    return jac[:2]


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    result_path = source.OUT / "result.json"
    result = json.loads(result_path.read_text())
    audits = [source.OUT / name for name in ("audit.json", "direction_structure_audit.json",
                                               "independent_geometry_replay.json")]
    for path in audits:
        record = json.loads(path.read_text())
        if record["status"] != "passed" or record["result_sha256"] != source.prior.previous.sha(result_path):
            raise ValueError("source audit binding")
    records = [row for row in result["rows"] if row["arm"].endswith("_waypoint")]
    dependencies = seal.capture([Path(__file__), PROTOCOL, result_path, *audits,
        Path(independent.__code__.co_filename), Path(source.prior.previous.exact.__file__),
        *(source.OUT / row["trace_file"] for row in records)])
    rows = []
    for row in records:
        if row["status"] != "completed":
            raise ValueError("failed source plan")
        task, goal = row["task"], np.asarray(row["task"]["goal"])
        with np.load(source.OUT / row["trace_file"], allow_pickle=False) as data:
            probes = data["actions_2"]
            nominal_endpoint = data["positions_2"][0]
        nominal = probes[0]
        probe_path, probe_blocked = independent(task["start"], probes)
        np.testing.assert_allclose(probe_path[0, -1, :2], nominal_endpoint, atol=1e-12, rtol=0)
        matrix = np.asarray(row["result"]["response"]["matrix"])
        jac = fixed_branch_jacobian(nominal, probe_path[0], probe_blocked[0])
        stable = bool(np.all(probe_blocked == probe_blocked[0]))
        for branch, saved in enumerate(row["result"]["waypoints"]):
            prefix = np.asarray(saved["prefix"], np.float32)
            direction = prefix.astype(float) - nominal.astype(float)
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-10:
                raise ValueError("zero saved direction")
            unit = direction / norm
            fd_actions = np.stack((nominal.astype(float) + .001 * unit,
                                   nominal.astype(float) - .001 * unit)).astype(np.float32)
            fd_path, fd_blocked = independent(task["start"], fd_actions)
            fd_exact = source.prior.previous.exact.endpoints(task, fd_actions)
            np.testing.assert_allclose(fd_path[:, -1, :2], fd_exact, atol=1e-12, rtol=0)
            derivative = (fd_exact[0] - fd_exact[1]) / .002
            analytic_directional = jac @ unit.reshape(-1)
            scale_actions = np.asarray([nominal.astype(float) + alpha * direction for alpha in SCALES], np.float32)
            path, blocked = independent(task["start"], scale_actions)
            endpoints = source.prior.previous.exact.endpoints(task, scale_actions)
            np.testing.assert_allclose(path[:, -1, :2], endpoints, atol=1e-12, rtol=0)
            scales = []
            for alpha, action, endpoint, events in zip(SCALES, scale_actions, endpoints, blocked, strict=True):
                difference = action.astype(float) - nominal.astype(float)
                predicted = nominal_endpoint + matrix @ difference.reshape(-1)
                abs_error = float(np.linalg.norm(predicted - endpoint))
                displacement = float(np.linalg.norm(endpoint - nominal_endpoint))
                physical_change = float(np.linalg.norm(endpoint-goal) - np.linalg.norm(nominal_endpoint-goal))
                predicted_change = float(np.linalg.norm(predicted-goal) - np.linalg.norm(nominal_endpoint-goal))
                valid = abs(physical_change) > 1e-10 and abs(predicted_change) > 1e-10
                scales.append({"alpha": alpha, "max_action_delta": float(np.max(np.abs(difference))),
                    "endpoint": endpoint.tolist(), "predicted_endpoint": predicted.tolist(),
                    "absolute_error": abs_error, "displacement": displacement,
                    "relative_error": abs_error / displacement if displacement > 1e-10 else None,
                    "goal_change": physical_change, "predicted_goal_change": predicted_change,
                    "sign_agreement": bool(np.sign(physical_change) == np.sign(predicted_change)) if valid else None,
                    "branch_changed": bool(not np.array_equal(events, probe_blocked[0]))})
            rows.append({"root": row["root"], "arm": row["arm"], "branch": branch,
                "nominal": nominal.tolist(), "prefix": prefix.tolist(),
                "probe_branch_stable": stable, "analytic_jacobian": jac.tolist(),
                "coordinate_jacobian_absolute_error": float(np.linalg.norm(matrix-jac)),
                "directional_fd": derivative.tolist(), "analytic_directional": analytic_directional.tolist(),
                "directional_absolute_error": float(np.linalg.norm(derivative - analytic_directional)),
                "directional_branch_stable": bool(np.all(fd_blocked == probe_blocked[0])),
                "scales": scales})
    if len(rows) != 48:
        raise ValueError("unexpected direction count")
    summary = []
    for index, alpha in enumerate(SCALES):
        values = [row["scales"][index] for row in rows]
        ratios = [item["relative_error"] for item in values if item["relative_error"] is not None]
        signs = [item["sign_agreement"] for item in values if item["sign_agreement"] is not None]
        summary.append({"alpha": alpha, "directions": len(values),
            "median_max_action_delta": float(np.median([item["max_action_delta"] for item in values])),
            "median_absolute_error": float(np.median([item["absolute_error"] for item in values])),
            "median_relative_error": float(np.median(ratios)) if ratios else None,
            "relative_defined": len(ratios), "goal_sign_agree": sum(signs), "goal_sign_defined": len(signs),
            "changed_branches": sum(item["branch_changed"] for item in values)})
    seal.verify(dependencies)
    source.prior.previous.write(OUT, {"status": "complete", "dependencies": dependencies, "rows": rows,
        "summary": summary, "scope": "48 optimizer-selected directions on exposed abstract E0 roots"})
    print(json.dumps({"summary": summary,
        "stable_coordinate_probes": sum(row["probe_branch_stable"] for row in rows),
        "stable_directional_probes": sum(row["directional_branch_stable"] for row in rows),
        "max_coordinate_jacobian_error": max(row["coordinate_jacobian_absolute_error"] for row in rows),
        "max_directional_error": max(row["directional_absolute_error"] for row in rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
