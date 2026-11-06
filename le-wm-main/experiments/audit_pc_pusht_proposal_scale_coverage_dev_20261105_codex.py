"""Independent arithmetic audit of saved proposal coverage records."""
import json

import numpy as np

from experiments import pc_pusht_proposal_scale_coverage_dev_20261105_codex as run


def main():
    report = json.loads((run.OUT / "report.json").read_text())
    if report["status"] != "completed" or len(report["rows"]) != 6:
        raise ValueError("incomplete report")
    if any(run.sha(path) != digest for path, digest in report["bindings"].items()):
        raise ValueError("source binding mismatch")
    expected = run.banks()
    seen = set()
    for row in report["rows"]:
        key = (row["sequence"], row["arm"])
        if key in seen:
            raise ValueError("duplicate row")
        seen.add(key)
        path = run.OUT / f"seq{key[0]}_{key[1]}.npz"
        if run.sha(path) != row["data_sha256"]:
            raise ValueError("data binding mismatch")
        with np.load(path, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["actions"], expected[key[1]])
            s, goal, hist, crossed = [data[k] for k in ("states", "goal", "history", "crossed")]
        position = np.sqrt(np.sum((s[:, :4] - goal[:4]) ** 2, axis=1))
        angle = np.abs(np.angle(np.exp(1j * (s[:, 4] - goal[4]))))
        checks = {"count": len(s),
                  "success_count": int(((position < 20) & (angle < np.pi/9)).sum()),
                  "best_joint_position_error": float(position.min()),
                  "median_joint_position_error": float(np.median(position)),
                  "median_object_displacement": float(np.median(np.sqrt(np.sum((s[:, 2:4] - hist[2:4])**2, axis=1)))),
                  "terminal_center_outside_count": int(((s[:, :4] < 40) | (s[:, :4] > 472)).any(axis=1).sum()),
                  "trajectory_center_outside_count": int(crossed.sum())}
        for name, value in checks.items():
            if not np.isclose(row[name], value, rtol=1e-6, atol=1e-6):
                raise ValueError(f"{key} {name}: {row[name]} != {value}")
    if seen != {(seq, arm) for seq in (9, 11) for arm in expected}:
        raise ValueError("missing row")
    result = {"status": "passed", "rows": len(seen), "report_sha256": run.sha(run.OUT / "report.json"),
              "limitations": "arithmetic and action regeneration only; no physical or model rerun"}
    with (run.OUT / "audit.json").open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
