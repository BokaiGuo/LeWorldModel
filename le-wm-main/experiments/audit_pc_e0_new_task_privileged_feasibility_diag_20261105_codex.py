"""Independent scalar geometry replay of privileged waypoint witnesses."""
import json
from pathlib import Path
import numpy as np
from experiments import pc_e0_new_task_privileged_feasibility_diag_20261105_codex as run


def scalar_replay(task, actions):
    state = np.array(task["start"], dtype=np.float64)
    trace, hits = [state.copy()], []
    for a in np.asarray(actions, dtype=np.float32):
        velocity = float(a[0])*.3
        turn = float(a[1])*(np.pi/2)
        dx, dy = velocity*np.cos(state[2]), velocity*np.sin(state[2])
        nx, ny = state[0]+dx, state[1]+dy
        crosses = (((state[0] <= 0 and nx >= 0) or (state[0] >= 0 and nx <= 0)) and abs(dx) > 1e-15)
        blocked = bool(crosses and abs(state[1] + (-state[0]/dx)*dy) < 1.)
        if not blocked:
            state[0], state[1] = nx, ny
        state[2] += turn
        trace.append(state.copy())
        hits.append(blocked)
    return np.array(trace), hits


def main():
    out = run.OUT
    report = json.loads((out / "report.json").read_text())
    seal = run.source.fixed.prior.prior.prior.seal
    seal.verify(report["dependencies"])
    seal.verify(report["source_dependencies"])
    if report["status"] != "completed" or len(report["rows"]) != 16:
        raise ValueError("incomplete report")
    successes, within, maxerr = 0, 0, 0.
    for row in report["rows"]:
        root_success = False
        for item in row["attempts"]:
            if not item["within_horizon"]:
                if item["required_steps"] <= 15 or item["success"]:
                    raise ValueError("invalid overhorizon flag")
                continue
            within += 1
            a = np.asarray(item["actions"], np.float32)
            if a.shape != (15, 2) or np.any(np.abs(a) > 1):
                raise ValueError("bad action")
            trace, hits = scalar_replay(row["task"], a)
            maxerr = max(maxerr, float(np.max(np.abs(trace-np.asarray(item["trajectory"])))))
            np.testing.assert_allclose(trace, item["trajectory"], rtol=0, atol=1e-12)
            if hits != item["blocked"]:
                raise ValueError("collision mismatch")
            score = float(np.linalg.norm(trace[-1, :2]-row["task"]["goal"]))
            if abs(score-item["score"]) > 1e-12 or (score < .05) != item["success"]:
                raise ValueError("score mismatch")
            root_success |= item["success"]
        successes += int(root_success)
    digest = run.source.prior.old.prior.digest
    result = {"status": "passed", "roots": len(report["rows"]), "within_horizon_attempts": within,
              "roots_with_success": successes, "max_trace_abs_error": maxerr,
              "report_sha256": digest(out / "report.json"), "auditor_sha256": digest(Path(__file__)),
              "scope": "independent scalar implementation of same geometry; privileged assessment only"}
    with (out / "audit.json").open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
