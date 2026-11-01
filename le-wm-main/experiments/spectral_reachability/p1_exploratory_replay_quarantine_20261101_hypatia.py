"""Validate two fixed saved replays and write format-only quarantined NPZs."""
import hashlib
import json
import os

import numpy as np

from experiments.spectral_reachability import p1_exploratory_rgb_replay_20261101_hypatia as replay
from experiments.spectral_reachability import p1_exploratory_rgb_v4_verify_20261101_hypatia as verify

NAME = "p1_exploratory_replay_quarantine_20261101_hypatia"
REPLAY_PREP_SHA = "b64cffe104724d1289b28ce46ed224341ddf31d2d4c13d59fc51d10efe461f4b"
REPLAY_JOURNAL_SHA = "5f694ba86a2b342aab15c5931152aecf99812068c0aad52f2fd54d565653a2e3"
REPLAY_RECEIPT_SHA = "6dc231bc05ddd831271c8c1442c54fc3ed1571fb61eef70e18bd056d0f113a66"
OUTPUT = replay.ROOT / f"results/{NAME}_LOCAL01"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def frame(directory, event):
    path = directory / event["file"]
    need(path.parent == directory and path.is_file() and not path.is_symlink()
         and replay.sha(path) == event["file_sha256"], "frame file binding")
    array = np.load(path, allow_pickle=False)
    need(array.dtype == np.uint8 and array.shape == (224, 224, 3)
         and hashlib.sha256(array.tobytes()).hexdigest() == event["pixel_sha256"], "frame pixel binding")
    return array


def evaluate():
    need(replay.sha(replay.PREPARED) == REPLAY_PREP_SHA
         and replay.sha(replay.OUTPUT / "receipt.json") == REPLAY_RECEIPT_SHA, "replay metadata SHA")
    prepared = json.loads(replay.PREPARED.read_bytes())
    for path, digest in prepared["files"].items():
        need(replay.sha(path) == digest, "prepared input drift")
    receipt = json.loads((replay.OUTPUT / "receipt.json").read_bytes())
    need(receipt["prepare_sha256"] == REPLAY_PREP_SHA and receipt["journal_sha256"] == REPLAY_JOURNAL_SHA
         and receipt["freshness"] == "unknown" and receipt["P1_admission"] is False,
         "replay receipt binding")
    old = verify.read_journal(verify.v4.OUTPUT / "journal.jsonl", verify.JOURNAL_SHA)
    new = verify.read_journal(replay.OUTPUT / "journal.jsonl", REPLAY_JOURNAL_SHA)
    need(new[0]["kind"] == "run_started" and new[0]["contract"] == prepared
         and new[-1]["kind"] == "run_finished" and new[-1]["slots"] == receipt["slots"], "run boundaries")
    expected_actions = np.asarray(prepared["actions_float32"], dtype=np.float32)
    outputs = []
    for slot in (0, 1):
        a = [e for e in old if e.get("slot") == slot]
        b = [e for e in new if e.get("slot") == slot]
        require_status = [r for r in receipt["slots"] if r["slot"] == slot]
        need(len(require_status) == 1 and require_status[0]["status"] == "completed"
             and require_status[0]["completed_steps"] == 10, "replay stopped or incomplete")
        for phase in ("original_seed", "placement"):
            aa = [e for e in a if e["kind"] == "reset_started" and e["phase"] == phase]
            bb = [e for e in b if e["kind"] == "reset_started" and e["phase"] == phase]
            need(len(aa) == len(bb) == 1 and aa[0]["seed"] == bb[0]["seed"], "reset schedule")
            if phase == "placement":
                need(aa[0]["goal_state"] == bb[0]["goal_state"]
                     and aa[0]["requested_state"] == bb[0]["requested_state"], "placement request")
        pa = [e for e in a if e["kind"] == "reset_completed" and e["phase"] == "placement"]
        pb = [e for e in b if e["kind"] == "reset_completed" and e["phase"] == "placement"]
        need(len(pa) == len(pb) == 1 and pa[0]["actual_state"] == pb[0]["actual_state"]
             and not pa[0]["target_visible_flag"] and not pb[0]["target_visible_flag"], "initial state")
        steps_a = [e for e in a if e["kind"] == "step_completed"]
        steps_b = [e for e in b if e["kind"] == "step_completed"]
        starts_b = [e for e in b if e["kind"] == "step_started"]
        need([e["step"] for e in steps_a] == [e["step"] for e in steps_b]
             == [e["step"] for e in starts_b] == list(range(1, 11)), "step census")
        for i, (x, y, start) in enumerate(zip(steps_a, steps_b, starts_b)):
            need(all(x[k] == y[k] for k in ("state", "raw_float32_bits", "terminated", "truncated",
                                             "angle_diagnostics"))
                 and y["raw_float32_bits"] == start["raw_float32_bits"] == expected_actions[i].tobytes().hex()
                 and not y["terminated"] and not y["truncated"], "step mismatch or stopped")
        frames = []
        for step in (0, 5, 10):
            aa = [e for e in a if e["kind"] == "render_completed" and e["step"] == step]
            bb = [e for e in b if e["kind"] == "render_completed" and e["step"] == step]
            need(len(aa) == len(bb) == 1 and aa[0]["file"] == bb[0]["file"]
                 == f"slot{slot}_rgb{step:02}.npy", "frame schedule")
            x, y = frame(verify.v4.OUTPUT, aa[0]), frame(replay.OUTPUT, bb[0])
            need(np.array_equal(x, y), "replay RGB mismatch")
            frames.append(y)
        ga = [e for e in a if e["kind"] == "goal_frame_saved"]
        gb = [e for e in b if e["kind"] == "goal_frame_saved"]
        need(len(ga) == len(gb) == 1 and ga[0]["file"] == gb[0]["file"]
             == f"slot{slot}_goal_rgb.npy", "goal frame schedule")
        goal_a, goal_b = frame(verify.v4.OUTPUT, ga[0]), frame(replay.OUTPUT, gb[0])
        need(np.array_equal(goal_a, goal_b), "goal RGB mismatch")
        outputs.append({"slot": slot, "rgb_history": np.stack(frames),
                        "raw_action_history": np.frombuffer(b"".join(bytes.fromhex(e["raw_float32_bits"])
                                                                      for e in steps_b), dtype=np.float32).copy().reshape(10, 2),
                        "goal_rgb": goal_b})
    return outputs


def write_quarantine(outputs):
    need([x["slot"] for x in outputs] == [0, 1], "fixed denominator")
    OUTPUT.mkdir()
    rows = []
    for slot in outputs:
        path = OUTPUT / f"slot{slot['slot']}_FORMAT_ONLY_NOT_FOR_PLANNER.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, rgb_history=slot["rgb_history"],
                                raw_action_history=slot["raw_action_history"], goal_rgb=slot["goal_rgb"])
            stream.flush()
            os.fsync(stream.fileno())
        with np.load(path, allow_pickle=False) as saved:
            need(set(saved.files) == {"rgb_history", "raw_action_history", "goal_rgb"}
                 and all(np.array_equal(saved[k], slot[k]) for k in saved.files), "payload verification")
        rows.append({"slot": slot["slot"], "file": path.name, "sha256": replay.sha(path)})
    result = {"schema": NAME + ".v1", "slots": rows, "replay_journal_sha256": REPLAY_JOURNAL_SHA,
              "v4_journal_sha256": verify.JOURNAL_SHA, "scope": "format-only quarantined; no planner",
              "replay_scope": "new instances; same code family; observable state/actions/RGB/goal/stop",
              "moving_fresh_witness_goal": "not_established", "hidden_solver_equivalence": "unknown",
              "historical_coverage": "unknown", "training_disjoint": "unknown", "freshness": "unknown",
              "planner_authorized": False, "P1_admission": False}
    replay.write(OUTPUT / "receipt.json", result)
    return result


def main():
    result = write_quarantine(evaluate())
    print(json.dumps({"directory": str(OUTPUT), "receipt_sha256": replay.sha(OUTPUT / "receipt.json"),
                      "slots": len(result["slots"]), "planner_authorized": False}))


if __name__ == "__main__":
    main()
