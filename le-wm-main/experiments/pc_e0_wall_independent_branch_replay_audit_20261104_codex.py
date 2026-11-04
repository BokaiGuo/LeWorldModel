"""Compare every saved wall action's independent and original state/event traces."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_e0_wall_independent_branch_replay_dev_20261104_codex as run
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

AUDIT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")


def main():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    sha = seal.digest
    manifest, output = (json.loads(path.read_text()) for path in (run.MANIFEST, run.OUTPUT))
    seal.verify(manifest["dependencies"])
    if output["manifest_sha256"] != sha(run.MANIFEST):
        raise ValueError("source manifest mismatch")
    rows8, rows7, baselines = (json.loads(p.read_text())["rows"] for p in
                               (run.v8.RESULT, run.panel.OUT, run.baseline.OUT))
    source = [(r, "v8") for r in rows8] + [(r, "v7") for r in rows7]
    source += [(r, r["arm"]) for r in baselines]
    indexed = {(r["root"], r["arm"]): r for r in output["rows"]}
    if len(indexed) != 72 or set(indexed) != {(r["root"], a) for r, a in source}:
        raise ValueError("panel mismatch")
    max_state_error, action_count, event_count, v8_crossed = 0., 0, 0, []
    for row, arm in source:
        totals = {"full_scored": 0, "full_ever_right": 0, "full_terminal_right": 0,
                  "full_any_blocked": 0, "full_success": 0}
        best_crossed = None
        for batch in row["trace"]:
            raw = np.asarray(batch["raw"], np.float32)
            length = raw.shape[1]
            test_path, test_blocked = run.independent(row["task"]["start"], raw)
            native = np.pad(raw.astype(np.float64), ((0, 0), (0, 16-length), (0, 0))) * run.panel.source.exact.wall.HIGH
            ref_path, ref_blocked = run.panel.source.exact.wall.rollout(row["task"]["start"], native)
            delta = np.abs(test_path-ref_path[:, :length+1])
            max_state_error = max(max_state_error, float(np.max(delta)))
            np.testing.assert_allclose(test_path, ref_path[:, :length+1], rtol=0, atol=1e-12)
            np.testing.assert_array_equal(test_blocked, ref_blocked[:, :length])
            action_count += len(raw)
            event_count += len(raw)*length
            if length == 15:
                scores = np.asarray(batch["scores"])
                right = np.any(test_path[:, 1:, 0] > 0, axis=1)
                totals["full_scored"] += len(raw)
                totals["full_ever_right"] += int(np.sum(right))
                totals["full_terminal_right"] += int(np.sum(test_path[:, -1, 0] > 0))
                totals["full_any_blocked"] += int(np.sum(np.any(test_blocked, axis=1)))
                totals["full_success"] += int(np.sum(scores < .05))
                if np.any(right):
                    candidate = float(np.min(scores[right]))
                    best_crossed = candidate if best_crossed is None else min(candidate, best_crossed)
        record = indexed[(row["root"], arm)]
        if any(record[k] != val for k, val in totals.items()):
            raise ValueError("branch counter mismatch")
        if arm == "v8" and best_crossed is not None:
            selected = row["result"]["selected"]["terminal_score"]
            v8_crossed.append({"root": row["root"], "best_crossed_score": best_crossed,
                               "selected_score": selected, "margin": best_crossed-selected})
    seal.verify(manifest["dependencies"])
    run.panel.source.write(AUDIT, {"result_sha256": sha(run.OUTPUT),
        "audit_script_sha256": sha(Path(__file__)), "trajectory_samples": action_count,
        "action_steps": event_count, "max_state_error": max_state_error,
        "v8_crossed_roots": v8_crossed,
        "scope": "compares independent reference and original backend on the same tasks and actions"})
    print(f"Audited {action_count} trajectories and {event_count} state/action steps.", flush=True)


if __name__ == "__main__":
    main()
