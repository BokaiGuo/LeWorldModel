"""Reconstruct v22 decision, dispatch, target and normalized action history."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pc_jepa_v22_contact_feedback_dev_20261104_codex"
OUT = ROOT / "results/pc_jepa_v22_action_dispatch_audit_20261105_codex.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_history(actions, mean, std):
    """Match float32 pixel data projection, then per-coordinate normalization."""
    u = (100 * np.asarray(actions, dtype=np.float64)).astype(np.float32) / np.float32(100)
    return ((u - mean) / std).reshape(2, 10)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    journal = [json.loads(line)["event"] for line in (SOURCE / "journal.jsonl").open()]
    rows = []
    bindings = {"journal.jsonl": sha(SOURCE / "journal.jsonl")}
    for arm in ("response", "cem"):
        intents = [e for e in journal if e.get("role") == arm and e.get("operation") == "step"
                   and e.get("phase") == "intent"]
        if len(intents) != 25:
            raise ValueError(f"{arm}: unexpected dispatch count {len(intents)}")
        intents_by_frame = {e["details"]["frame"]: e["details"] for e in intents}
        if sorted(intents_by_frame) != list(range(1, 26)):
            raise ValueError("missing or duplicate frames")
        actions = [intents_by_frame[i]["action"] for i in range(1, 26)]
        for step in range(3):
            path = SOURCE / f"{arm}_{step}"
            decision_path, context_path = path / "output/decision.json", path / "input/context.npz"
            decision, decision_hash = json.loads(decision_path.read_text()), sha(decision_path)
            if decision["input_sha256"] != sha(context_path):
                raise ValueError("decision does not bind context")
            if decision["arm"] != arm or decision["step"] != step:
                raise ValueError("decision identity")
            bindings[str(decision_path.relative_to(SOURCE))] = decision_hash
            bindings[str(context_path.relative_to(SOURCE))] = sha(context_path)
            selected = np.asarray(decision["decision"]["selected"]["chain"], dtype=np.float32)
            if selected.shape != (15 - 5 * step, 2):
                raise ValueError("chain shape")
            with np.load(context_path, allow_pickle=False) as context:
                mean, std = context["action_mean"], context["action_std"]
                norm = context["normalized_history_packets"]
                np.testing.assert_array_equal(norm, expected_history(actions[step * 5:step * 5 + 10], mean, std))
            outside_canvas, negative_actions, targets_outside = 0, 0, 0
            for j in range(5):
                frame = 11 + 5 * step + j
                command_path = SOURCE / f"{arm}_physical/command_{frame:02d}.json"
                command = json.loads(command_path.read_text())
                bindings[str(command_path.relative_to(SOURCE))] = sha(command_path)
                if command != intents_by_frame[frame] or command["decision_sha256"] != decision_hash:
                    raise ValueError("command, journal, or decision binding mismatch")
                np.testing.assert_array_equal(np.asarray(command["action"], dtype=np.float32), selected[j])
                previous_transition = SOURCE / f"{arm}_physical/frame_{frame - 1:02d}/transition.json"
                previous = json.loads(previous_transition.read_text())
                before = np.asarray(previous["info"]["pos_agent"]["values"], dtype=np.float64)
                np.testing.assert_array_equal(before, np.asarray(command["agent_xy_before"], dtype=np.float64))
                np.testing.assert_array_equal(before + 100 * np.asarray(command["action"]),
                                              np.asarray(command["absolute_target"]))
                outside_canvas += int(np.any((before < 0) | (before > 512)))
                targets_outside += int(np.any((np.asarray(command["absolute_target"]) < 0)
                                              | (np.asarray(command["absolute_target"]) > 512)))
                negative_actions += int(np.any(np.asarray(command["action"]) < 0))
            rows.append({"arm": arm, "step": step, "history_frames": [step * 5 + 1, step * 5 + 10],
                         "dispatch_frames": [11 + 5 * step, 15 + 5 * step],
                         "dispatches_checked": 5, "negative_relative_action_count": negative_actions,
                         "pre_step_agent_outside_canvas": outside_canvas,
                         "absolute_target_outside_canvas": targets_outside})
            print(f"{arm} decision {step}: history and 5 dispatches verified", flush=True)
    output = {"status": "passed", "scope": "saved v22 records only; journaled call intent, not independent live interception; no provenance claim",
              "auditor_sha256": sha(Path(__file__)), "bindings": bindings, "rows": rows}
    with OUT.open("x") as f:
        json.dump(output, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
