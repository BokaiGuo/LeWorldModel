"""Capture actual state-assisted feedback observations without changing historical runner."""
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import pc_pointmaze_packet_cem_feedback_20261107_codex as source
from experiments.pc_input_permissions import file_hash

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_LIVE_OBSERVATION_CAPTURE_PROTOCOL_20261107_CODEX.md"
BASE = ROOT / "results/pc_pointmaze_live_observation_capture_20261107_codex"
RESULT = Path(str(BASE) + "_run.json")
OBS = Path(str(BASE) + "_observations.npz")
AUDIT = Path(str(BASE) + "_audit.json")


def array_hash(value):
    arr = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(json.dumps(arr.shape).encode("ascii"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def compare(old, new, frames, states):
    if len(old["rows"]) != 8 or len(new["rows"]) != 8:
        raise ValueError("expected eight roots")
    if frames.shape != (8, 5, 224, 224, 3) or states.shape != (8, 5, 4):
        raise ValueError("unexpected captured observation shapes")
    decisions, rows = [], []
    for i, (oldrow, row) in enumerate(zip(old["rows"], new["rows"], strict=True)):
        if oldrow["root"] != row["root"] or len(row["decisions"]) != 4:
            raise ValueError("root or decision mismatch")
        per = []
        for k, (previous, decision) in enumerate(zip(oldrow["decisions"], row["decisions"], strict=True)):
            np.testing.assert_allclose(states[i, k], decision["observation_proprio_before"], atol=1e-6, rtol=0)
            scores = np.asarray([r["costs"] for r in decision["rounds"]])
            old_scores = np.asarray([r["costs"] for r in previous["rounds"]])
            if scores.shape != (3, 16) or old_scores.shape != scores.shape or not np.isfinite(scores).all():
                raise ValueError("invalid scores")
            entry = {"step": decision["step"], "rgb_sha256": array_hash(frames[i, k]),
                     "proprio_sha256": array_hash(states[i, k]),
                     "score_max_abs_difference": float(np.max(np.abs(scores - old_scores))),
                     "chosen_matches_history": decision["chosen"] == previous["chosen"]}
            per.append(entry)
            decisions.append(entry)
        np.testing.assert_allclose(states[i, 4], row["final_proprio"], atol=1e-6, rtol=0)
        rows.append({"root": row["root"], "decisions": per,
                     "terminal_rgb_sha256": array_hash(frames[i, 4]),
                     "actions_match_history": row["executed_actions"] == oldrow["executed_actions"],
                     "final_distance_difference": row["feedback_final_distance"] - oldrow["feedback_final_distance"]})
    return {"rows": rows, "summary": {
        "observations": len(decisions),
        "chosen_matches_history": sum(x["chosen_matches_history"] for x in decisions),
        "max_score_abs_difference": max(x["score_max_abs_difference"] for x in decisions),
        "actions_match_history": sum(x["actions_match_history"] for x in rows),
        "max_final_distance_abs_difference": max(abs(x["final_distance_difference"]) for x in rows)}}


def main():
    for path in (RESULT, OBS, AUDIT):
        if path.exists():
            raise FileExistsError(path)
    original_bridge = source.NativeRGBBridge
    original_output = source.OUTPUT
    captures = []

    class CaptureBridge:
        def __init__(self, initial):
            self.env = original_bridge(initial)
            self.frames, self.states = [], []
            captures.append((self.frames, self.states))

        def __enter__(self):
            self.env.__enter__()
            return self

        def __exit__(self, *args):
            return self.env.__exit__(*args)

        def __getattr__(self, key):
            return getattr(self.env, key)

        def step(self, actions):
            rgb, proprio = self.env.step(actions)
            self.frames.append(np.array(rgb, copy=True))
            self.states.append(np.array(proprio, copy=True))
            return rgb, proprio

    try:
        source.NativeRGBBridge = CaptureBridge
        source.OUTPUT = RESULT
        source.main()
    finally:
        source.NativeRGBBridge = original_bridge
        source.OUTPUT = original_output
    frames = np.stack([np.stack(row[0]) for row in captures])
    states = np.stack([np.stack(row[1]) for row in captures])
    result = compare(json.loads(original_output.read_text()), json.loads(RESULT.read_text()), frames, states)
    with OBS.open("xb") as stream:
        np.savez_compressed(stream, rgb=frames, proprio=states)
    result.update({"track": "state_assisted", "protocol": file_hash(PROTOCOL),
                   "capture_runner": file_hash(__file__), "historical_run": file_hash(original_output),
                   "new_run": file_hash(RESULT), "observations": file_hash(OBS),
                   "scope": "live observations, exposed roots; not RGB-only or independent"})
    with AUDIT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
