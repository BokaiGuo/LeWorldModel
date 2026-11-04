"""Reconstruct direction selection and matched model probes from saved traces."""
import json

import numpy as np

from experiments import pc_e0_response_waypoint_direction_dev_20261104_codex as source


def main():
    output = source.OUT / "direction_structure_audit.json"
    if output.exists():
        raise FileExistsError(output)
    result = json.loads((source.OUT / "result.json").read_text())
    audit = json.loads((source.OUT / "audit.json").read_text())
    if audit["result_sha256"] != source.prior.previous.sha(source.OUT / "result.json"):
        raise ValueError("source audit mismatch")
    rows = []
    for root in range(8):
        records = [row for row in result["rows"] if row["root"] == root and row["arm"].endswith("_waypoint")]
        if len(records) != 3 or any(row["status"] != "completed" for row in records):
            raise ValueError("incomplete direction arms")
        probe_actions, record_by_mode = [], {}
        for row in records:
            mode = row["arm"].removesuffix("_waypoint")
            with np.load(source.OUT / row["trace_file"], allow_pickle=False) as trace:
                probes, positions = trace["actions_2"], trace["positions_2"]
                origin = trace["positions_1"][0]
            probe_actions.append(probes)
            matrix = ((positions[1:11] - positions[11:21]) / .1).T
            left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
            rank = int(np.count_nonzero(singular > 1e-10))
            delta = np.asarray(row["task"]["goal"]) - origin
            selected_mode = int(np.argmin(np.abs(left[:, :rank].T @ (delta / np.linalg.norm(delta)))))
            if mode == "response":
                direction = left[:, selected_mode].copy()
            elif mode == "random":
                direction = np.random.default_rng(2028120100 + root * 100 + 30).normal(size=2)
                direction /= np.linalg.norm(direction)
            else:
                direction = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
            if direction[np.flatnonzero(np.abs(direction) > 1e-12)[0]] < 0:
                direction = -direction
            saved = row["result"]["response"]
            np.testing.assert_allclose(saved["matrix"], matrix, atol=1e-12, rtol=0)
            np.testing.assert_allclose(saved["singular"], singular, atol=1e-12, rtol=0)
            np.testing.assert_allclose(saved["direction"], direction, atol=1e-12, rtol=0)
            if saved["rank"] != rank or saved["mode"] != selected_mode:
                raise ValueError("rank or mode mismatch")
            for index, sign in enumerate((-1, 1)):
                expected = origin + .5 * delta + sign * 1.2 * np.linalg.norm(delta) * direction
                np.testing.assert_allclose(row["result"]["waypoints"][index]["waypoint"], expected, atol=1e-12, rtol=0)
            record_by_mode[mode] = {"direction": direction.tolist(), "score": row["score"]}
        for probe in probe_actions[1:]:
            np.testing.assert_array_equal(probe, probe_actions[0])
        rows.append({"root": root, "singular": singular.tolist(), "matched_probe_samples": len(probe_actions[0]),
                     "modes": record_by_mode})
    source.prior.previous.write(output, {"status": "passed", "result_sha256": source.prior.previous.sha(source.OUT / "result.json"),
        "roots": 8, "rows": rows, "scope": "same-SVD deterministic reconstruction, not numerical independence"})
    print("Reconstructed directions, waypoint targets and matched probes on 8 roots.", flush=True)


if __name__ == "__main__":
    main()
