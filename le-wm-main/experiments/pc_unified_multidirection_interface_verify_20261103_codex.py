"""Replay sealed RGB decisions and exercise the unified entry with exact feedback."""
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import pc_multidirection_rgb_fixed_terminal_dev_20261103_codex as rgb
from experiments import pc_successor_response_composition_e0_dev_20261103_codex as exact
from experiments import pc_block_program_wall_prospective_dev_20261103_codex as tasks
from experiments.spectral_reachability import pc_unified_multidirection_dev_v1 as unified

ROOT = rgb.runner.ROOT
PROTOCOL = ROOT.parent / "docs/PC_UNIFIED_MULTIDIRECTION_DEV_INTERFACE_20261103_CODEX.md"
OUTPUT = ROOT / "results/pc_unified_multidirection_interface_verify_20261103_codex.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    rgb.runner.PREFIX = rgb.PREFIX
    paths = [rgb.runner.paths(root, "multidirection")[0][step] for root in range(4) for step in range(3)]
    files = [Path(__file__), PROTOCOL, Path(unified.__file__), Path(exact.__file__),
             Path(exact.wall.__file__), Path(exact.wall.physics.__file__), Path(tasks.__file__), *paths]
    hashes = {str(p): sha(p) for p in files}
    rows = []
    print("Replay 12 sealed RGB decisions; zero target is a dimension-only placeholder.", flush=True)
    for root in range(4):
        previous_tail = None
        for step in range(3):
            path = rgb.runner.paths(root, "multidirection")[0][step]
            decision = json.loads(path.read_text())
            saved = decision["result"]
            for name, digest in saved["actual_source_sha256"].items():
                if sha(name) != digest:
                    raise ValueError("sealed RGB compiler source drift")
            trace = saved["prediction_trace"]
            cursor = [0]

            def predict(requested):
                if cursor[0] >= len(trace):
                    raise ValueError("more predictions than sealed trace")
                batch = trace[cursor[0]]
                cursor[0] += 1
                np.testing.assert_array_equal(requested, np.asarray(batch["requested_actions"]))
                np.testing.assert_array_equal(requested.astype(np.float32), np.asarray(batch["model_raw_float32_actions"], dtype=np.float32))
                return np.asarray(batch["positions"], dtype=np.float32), np.asarray(batch["scores"], dtype=np.float32)

            binding = sha(path)
            output = unified.compile_decision(root=root, step=step, target=[0., 0.], predict=predict,
                                              context_id=binding, carry=previous_tail)
            if cursor[0] != len(trace):
                raise ValueError("sealed predictions not fully consumed")
            np.testing.assert_array_equal(output["core_result"]["selected"]["chain"], saved["selected"]["chain"])
            if output["core_result"]["abstained"] != saved["abstained"]:
                raise ValueError("abstention mismatch")
            if output["counts"]["prediction_token_samples"] != decision["counts"]["prediction_token_samples"]:
                raise ValueError("token mismatch")
            if output["counts"]["prediction_calls"] != decision["counts"]["prediction_calls"]:
                raise ValueError("call mismatch")
            if step and saved["carry"] != previous_tail:
                raise ValueError("carry mismatch")
            previous_tail = output["remaining_tail"]
            rows.append({"root": root, "step": step, "source_sha256": binding, "calls": cursor[0],
                         "tokens": output["counts"]["prediction_token_samples"],
                         "selected": output["core_result"]["selected"], "tail": previous_tail})
    print("Exercise 3 exact-dynamics feedback steps on one exposed development task.", flush=True)
    task = tasks.tasks()[0]
    current = {"start": task["start"], "goal": task["goal"]}
    carry, exact_steps = None, []
    for step in range(3):
        def actual(actions):
            model = np.asarray(actions, dtype=np.float32)
            positions = exact.endpoints(current, model)
            return positions, np.linalg.norm(positions - current["goal"], axis=1)

        context = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()
        output = unified.compile_decision(root=0, step=step, target=current["goal"], predict=actual,
                                          context_id=context, carry=carry)
        executed = np.asarray(output["dispatched_actions"], dtype=np.float32)
        padded = np.pad(executed[None], ((0, 0), (0, 11), (0, 0))) * exact.wall.HIGH
        trajectory, blocked = exact.wall.rollout(current["start"], padded)
        next_state = trajectory[0, 5].tolist()
        exact_steps.append({"step": step, "start": current["start"], "context_id": context,
                            "selected": output["core_result"]["selected"], "executed": executed.tolist(),
                            "next_state": next_state, "blocked": blocked[0, :5].tolist(),
                            "counts": output["counts"], "carry_out": output["remaining_tail"]})
        current = {"start": next_state, "goal": task["goal"]}
        carry = output["remaining_tail"]
    if carry != [] or hashes != {str(p): sha(p) for p in files}:
        raise ValueError("final tail or dependency drift")
    with OUTPUT.open("x") as stream:
        json.dump({"scope": "interface parity and one exposed exact execution, not scientific confirmation",
                   "dependencies": hashes, "rgb_trace_replay": rows, "exact_task": task,
                   "exact_feedback": exact_steps}, stream, allow_nan=False, indent=2)
    print("Verified 12 RGB trace decisions and 3 exact feedback decisions.", flush=True)


if __name__ == "__main__":
    main()
