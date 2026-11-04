"""Independently audit unified-interface result bindings and exact dispatch."""
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import pc_unified_multidirection_interface_verify_20261103_codex as verify

OUTPUT = verify.OUTPUT.with_name(verify.OUTPUT.stem + "_audit.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = json.loads(verify.OUTPUT.read_text())
    for name, digest in data["dependencies"].items():
        if verify.sha(name) != digest:
            raise ValueError("dependency drift")
    if len(data["rgb_trace_replay"]) != 12 or len(data["exact_feedback"]) != 3:
        raise ValueError("incomplete feedback panel")
    for row in data["rgb_trace_replay"]:
        verify.rgb.runner.PREFIX = verify.rgb.PREFIX
        path = verify.rgb.runner.paths(row["root"], "multidirection")[0][row["step"]]
        original = json.loads(path.read_text())
        if row["source_sha256"] != verify.sha(path) or row["selected"] != original["result"]["selected"]:
            raise ValueError("RGB source or selection mismatch")
        if row["tokens"] != original["counts"]["prediction_token_samples"] or row["calls"] != original["counts"]["prediction_calls"]:
            raise ValueError("RGB budget mismatch")
    current, carry = data["exact_task"]["start"], None
    for step, row in enumerate(data["exact_feedback"]):
        if row["step"] != step or row["start"] != current:
            raise ValueError("state clock mismatch")
        task = {"start": current, "goal": data["exact_task"]["goal"]}
        context = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
        if row["context_id"] != context:
            raise ValueError("context hash mismatch")
        chain = np.asarray(row["selected"]["chain"])
        if chain.shape != (15 - 5 * step, 2):
            raise ValueError("remaining horizon mismatch")
        np.testing.assert_array_equal(chain[:5].astype(np.float32), np.asarray(row["executed"], dtype=np.float32))
        np.testing.assert_array_equal(chain[5:], np.asarray(row["carry_out"]).reshape(-1, 2))
        position = verify.exact.endpoints(task, chain[None].astype(np.float32))[0]
        score = float(np.linalg.norm(position - task["goal"]))
        if abs(score - row["selected"]["terminal_score"]) > 1e-12:
            raise ValueError("selected chain score mismatch")
        actions = np.asarray(row["executed"], dtype=np.float32)
        padded = np.pad(actions[None], ((0, 0), (0, 11), (0, 0))) * verify.exact.wall.HIGH
        trajectory, blocked = verify.exact.wall.rollout(current, padded)
        np.testing.assert_allclose(trajectory[0, 5], row["next_state"], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(blocked[0, :5], row["blocked"])
        if row["counts"]["prediction_token_samples"] > 480:
            raise ValueError("budget exceeded")
        current, carry = row["next_state"], row["carry_out"]
    if carry != []:
        raise ValueError("tail remains after endpoint")
    with OUTPUT.open("x") as stream:
        json.dump({"verification_sha256": verify.sha(verify.OUTPUT), "audit_sha256": verify.sha(Path(__file__)),
                   "rgb_bindings": 12, "exact_feedback_steps": 3, "status": "passed"}, stream, indent=2)
    print("Audited 12 RGB bindings and 3 exact dispatch/state/score steps.")


if __name__ == "__main__":
    main()
