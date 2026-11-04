"""Independent line-segment replay of sealed E0 planning traces."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v7_e0_exposed_panel_dev_20261104_codex as panel
from experiments import pc_v7_e0_nonzero_cem_dev_20261104_codex as baseline
from experiments import pc_v8_search_induced_nominal_e0_dev_20261104_codex as v8
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = panel.source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_E0_WALL_INDEPENDENT_BRANCH_REPLAY_DEV_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_e0_wall_independent_branch_replay_dev_20261104_codex"
MANIFEST, OUTPUT = [PREFIX.with_name(PREFIX.name + s) for s in ("_manifest.json", "_result.json")]


def independent(start, actions):
    """Scalar reference on each step, vectorized only across independent chains."""
    actions = np.asarray(actions, dtype=np.float64)
    state = np.broadcast_to(np.asarray(start, dtype=np.float64), (len(actions), 3)).copy()
    path, events = [state.copy()], []
    for step in range(actions.shape[1]):
        move = actions[:, step, 0] * .3
        end_x = state[:, 0] + move*np.cos(state[:, 2])
        end_y = state[:, 1] + move*np.sin(state[:, 2])
        delta = end_x - state[:, 0]
        hits_x = (np.minimum(state[:, 0], end_x) <= 0) & (np.maximum(state[:, 0], end_x) >= 0)
        defined = np.abs(delta) > 1e-15
        # Compute line intersection from endpoints rather than using dynamics helper.
        ratio = np.divide(-state[:, 0], delta, out=np.zeros_like(delta), where=defined)
        hit_y = state[:, 1] + ratio*(end_y-state[:, 1])
        blocked = hits_x & defined & (np.abs(hit_y) < 1.0)
        state[:, 0] = np.where(blocked, state[:, 0], end_x)
        state[:, 1] = np.where(blocked, state[:, 1], end_y)
        state[:, 2] += actions[:, step, 1] * (np.pi / 2)
        path.append(state.copy())
        events.append(blocked.copy())
    return np.stack(path, axis=1), np.stack(events, axis=1)


def main():
    if MANIFEST.exists() or OUTPUT.exists():
        raise FileExistsError("prior or partial result")
    sha = seal.digest
    if (json.loads(v8.AUDIT.read_text())["result_sha256"] != sha(v8.RESULT)
            or json.loads(panel.AUDIT.read_text())["plans_sha256"] != sha(panel.OUT)
            or json.loads(baseline.AUDIT.read_text())["result_sha256"] != sha(baseline.OUT)):
        raise ValueError("source audit mismatch")
    doc8, doc7, docs = (json.loads(p.read_text()) for p in (v8.RESULT, panel.OUT, baseline.OUT))
    plans = [(r, "v8") for r in doc8["rows"]] + [(r, "v7") for r in doc7["rows"]]
    plans += [(r, r["arm"]) for r in docs["rows"]]
    if len(plans) != 72 or len({(r["root"], arm) for r, arm in plans}) != 72:
        raise ValueError("source panel mismatch")
    deps = seal.capture([Path(__file__), PROTOCOL, v8.MANIFEST, v8.RESULT, v8.AUDIT,
                         panel.OUT, panel.AUDIT, baseline.OUT, baseline.AUDIT,
                         Path(panel.source.exact.__file__), Path(panel.source.exact.wall.__file__),
                         Path(panel.source.exact.wall.physics.__file__)])
    panel.source.write(MANIFEST, {"dependencies": deps, "scope": "posthoc independent geometry"})
    records = []
    max_error = 0.
    print("Checking sealed plans against independent wall-segment geometry.", flush=True)
    for row, arm in plans:
        goal = np.asarray(row["task"]["goal"])
        selected = row["result"]["selected"]["chain"] if arm in ("v7", "v8") else row["selected"]
        selected = np.asarray(selected, np.float32).astype(np.float64)
        metrics = {"full_scored": 0, "full_ever_right": 0, "full_terminal_right": 0,
                   "full_any_blocked": 0, "full_success": 0,
                   "warm_full_scored": 0, "warm_full_ever_right": 0, "warm_full_success": 0}
        for batch_index, batch in enumerate(row["trace"]):
            requested, actions = np.asarray(batch["requested"]), np.asarray(batch["raw"], np.float32)
            np.testing.assert_array_equal(requested.astype(np.float32), actions)
            if np.any(np.abs(actions) > 1):
                raise ValueError("out-of-bounds action")
            path, blocked = independent(row["task"]["start"], actions)
            pos, scores = path[:, -1, :2], np.linalg.norm(path[:, -1, :2] - goal, axis=1)
            max_error = max(max_error, float(np.max(np.abs(pos - np.asarray(batch["positions"])))))
            np.testing.assert_allclose(pos, batch["positions"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(scores, batch["scores"], atol=1e-12, rtol=0)
            if actions.shape[1] == 15:
                right = np.any(path[:, 1:, 0] > 0, axis=1)
                success = scores < .05
                metrics["full_scored"] += len(actions)
                metrics["full_ever_right"] += int(np.sum(right))
                metrics["full_terminal_right"] += int(np.sum(path[:, -1, 0] > 0))
                metrics["full_any_blocked"] += int(np.sum(np.any(blocked, axis=1)))
                metrics["full_success"] += int(np.sum(success))
                if arm == "v8" and batch_index < 5:
                    metrics["warm_full_scored"] += len(actions)
                    metrics["warm_full_ever_right"] += int(np.sum(right))
                    metrics["warm_full_success"] += int(np.sum(success))
        trajectory, blocked = independent(row["task"]["start"], selected[None])
        padded = np.pad(selected[None], ((0, 0), (0, 1), (0, 0))) * panel.source.exact.wall.HIGH
        original_path, original_blocked = panel.source.exact.wall.rollout(row["task"]["start"], padded)
        np.testing.assert_allclose(trajectory[0], original_path[0, :16], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(blocked[0], original_blocked[0, :15])
        metrics["selected_ever_right"] = bool(np.any(trajectory[0, 1:, 0] > 0))
        metrics["selected_any_blocked"] = bool(np.any(blocked))
        metrics["selected_terminal_distance"] = float(np.linalg.norm(trajectory[0, -1, :2]-goal))
        if arm == "v8" and metrics["warm_full_scored"] != 49:
            raise ValueError("warm search phase changed")
        records.append({"root": row["root"], "arm": arm, **metrics})
    seal.verify(deps)
    summary = {}
    for arm in ("v8", "v7", "smooth", "action"):
        entries = [r for r in records if r["arm"] == arm]
        keys = ("full_scored", "full_ever_right", "full_terminal_right", "full_any_blocked", "full_success",
                "warm_full_scored", "warm_full_ever_right", "warm_full_success",
                "selected_ever_right", "selected_any_blocked")
        summary[arm] = {key: sum(r[key] for r in entries) for key in keys}
    panel.source.write(OUTPUT, {"manifest_sha256": sha(MANIFEST), "max_endpoint_error": max_error,
                                 "rows": records, "summary": summary,
                                 "scope": "independent analytic implementation, same exposed tasks; repeated samples counted"})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
