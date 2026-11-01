"""Export both sealed dev303 cp5 RGB/action histories for development only.

No model, CUDA, simulator, action selection, or outcome-dependent filtering.
The exported history uses warmup steps 5/10 and actual control step 5; its two
action packets are actual warmup steps 6--10 and actual control steps 1--5.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_dev303_rgb_chunk_diagnostic_20261102_exclusive as run

NAME = "p1_dev303_cp5_rgb_action_export_20261102_exclusive"
ROOT = run.ROOT
RUN_DIR = ROOT / "results/p1_dev303_rgb_chunk_diagnostic_20261102_exclusive_RUN_DEV01"
PREP = ROOT / "results/p1_dev303_rgb_chunk_diagnostic_20261102_exclusive_PREP_DEV01.json"
REPORT_SHA = "694faea5854ee2d5b558fb9b10530aa5ecb20865cf5d2ce27473ec5c2c417625"
PREP_SHA = "668232ba7891bd447fd5ebfb222627c29c587b0664642cbd83957a85670abeb5"
ARMS = run.ARMS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_regular(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"regular source: {path}")
    return path.read_bytes()


def decode(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    def invalid(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def verify_journal(raw, report):
    require(sha(raw) == report["journal_sha256"] and raw.endswith(b"\n"),
            "journal SHA or incomplete")
    rows = [decode(line) for line in raw.splitlines()]
    require(bool(rows), "empty journal")
    previous = None
    for index, row in enumerate(rows):
        body = {k: v for k, v in row.items() if k != "sha256"}
        require(row.get("index") == index and row.get("previous") == previous
                and row.get("sha256") == sha(canonical(body)), "journal hash chain")
        previous = row["sha256"]
    require(previous == report["journal_tail"], "journal tail")
    return rows


def event(events, arm, kind, phase=None, step=None):
    matches = [e for e in events if e.get("arm") == arm and e["kind"] == kind
               and (phase is None or e.get("phase") == phase)
               and (step is None or e.get("step") == step)]
    require(len(matches) == 1, f"{arm}: unique {kind}/{phase}/{step}")
    return matches[0]


def _action(value):
    raw = np.asarray(value, dtype=np.float64)
    require(raw.shape == (2,) and np.isfinite(raw).all() and (np.abs(raw) <= 1).all(),
            "bounded action")
    action = raw.astype("<f4")
    require(np.array_equal(raw, action.astype(np.float64)), "exact float32 action")
    return action


def _frame(directory, receipt):
    name = receipt["file"]
    require(isinstance(name, str) and Path(name).name == name, "frame basename")
    raw = read_regular(directory / name)
    require(sha(raw) == receipt["file_sha256"], "frame file SHA")
    with Image.open(io.BytesIO(raw)) as image:
        require(image.mode == "RGB", "frame RGB mode")
        pixels = np.asarray(image, dtype=np.uint8).copy()
    require(pixels.shape == (224, 224, 3) and sha(pixels.tobytes()) == receipt["pixel_sha256"],
            "frame pixel SHA")
    return pixels


def verify_arm(directory, arm, row, events, warmup_sha):
    require(row["arm"] == arm and row["status"] == "fixed_horizon_completed"
            and row["warmup_returned"] == 10 and row["control_returned"] == 5,
            f"{arm}: complete five-step execution")
    require(decode(read_regular(directory / (arm + ".row.json"))) == row, "row/report match")
    obs = row["observations"]
    require([(o["phase"], o["step"]) for o in obs] ==
            [("warmup", 0), ("warmup", 5), ("warmup", 10)]
            + [("control", n) for n in range(1, 6)], "complete observation roster")
    frames = []
    for receipt in obs:
        frame = _frame(directory, receipt)
        ev = event(events, arm, "observation", receipt["phase"], receipt["step"])
        expected = {k: v for k, v in ev.items()
                    if k not in ("index", "previous", "kind", "arm", "sha256")}
        require(expected == receipt, "observation journal/report mismatch")
        frames.append(frame)
    require([o["pixel_sha256"] for o in obs[:3]] == warmup_sha, "sealed warmup RGB")
    for step in range(1, 11):
        dispatch = event(events, arm, "action_dispatch", "warmup", step)
        observed = event(events, arm, "action_observed", "warmup", step)
        returned = event(events, arm, "action_return", "warmup", step)
        require(dispatch["index"] < observed["index"] < returned["index"]
                and all(np.array_equal(_action(e["action"]), np.zeros(2, "<f4"))
                        for e in (dispatch, observed, returned))
                and observed["terminated"] is observed["truncated"] is False
                and returned["terminated"] is returned["truncated"] is False,
                "actual warmup action chain")
    require(len(row["transitions"]) == 5
            and [t["step"] for t in row["transitions"]] == list(range(1, 6)),
            "actual action roster")
    actions = []
    for step, transition in enumerate(row["transitions"], 1):
        dispatch = event(events, arm, "action_dispatch", "control", step)
        observed = event(events, arm, "action_observed", "control", step)
        returned = event(events, arm, "action_return", "control", step)
        observation = event(events, arm, "observation", "control", step)
        action = _action(transition["action"])
        action_sha = sha(action.tobytes())
        require(dispatch["index"] < observed["index"] < returned["index"] < observation["index"]
                and all(np.array_equal(_action(e["action"]), action)
                        for e in (dispatch, observed, returned))
                and dispatch["action_sha256"] == returned["action_sha256"]
                == transition["action_sha256"] == action_sha,
                "actual action chain")
        returned_body = {k: v for k, v in returned.items()
                         if k not in ("index", "previous", "kind", "arm", "phase", "sha256")}
        require(returned_body == transition and not transition["terminated"]
                and not transition["truncated"], "transition journal/report mismatch")
        if arm == "rgb_chunk":
            decision = event(events, arm, "controller_decision", step=step)
            proposal = decision["proposal"]
            previous_obs = event(events, arm, "observation",
                                 "warmup" if step == 1 else "control",
                                 10 if step == 1 else step - 1)
            require(previous_obs["index"] < decision["index"] < dispatch["index"]
                    and proposal["frame_id"] == step - 1
                    and proposal["rgb_sha256"] == obs[step + 1]["pixel_sha256"]
                    and proposal["action_sha256"] == action_sha
                    and np.array_equal(_action(proposal["action"]), action)
                    and proposal["P1_admission"] is False, "RGB decision binding")
        else:
            require(not action.any(), "zero arm actual action")
        actions.append(action)
    return {
        "rgb_history": np.stack([frames[1], frames[2], frames[-1]]),
        "raw_action_history": np.vstack([np.zeros((5, 2), "<f4"), actions]),
        "cp5_observation": obs[-1],
    }


def build():
    prep_raw = read_regular(PREP)
    report_raw = read_regular(RUN_DIR / "report.json")
    require(sha(prep_raw) == PREP_SHA and sha(report_raw) == REPORT_SHA,
            "fixed preparation/report SHA")
    prep, report = decode(prep_raw), decode(report_raw)
    require(prep == run.build() and prep["P1_admission"] is False
            and prep["independent_confirmation"] is False
            and prep["arms_in_fixed_order"] == list(ARMS), "sealed preparation")
    require(report["schema"] == run.NAME + ".report.v1" and report["status"] == "recorded"
            and report["prepared_sha256"] == PREP_SHA and report["errors"] == []
            and report["P1_admission"] is False and report["independent_confirmation"] is False
            and report["classification"] == "development_only_not_original_holdout_confirmation"
            and [r["arm"] for r in report["rows"]] == list(ARMS), "development report")
    binding = decode(read_regular(RUN_DIR / "execution_binding.json"))
    require(binding == {"prepared": str(PREP.resolve()), "prepare_sha256": PREP_SHA},
            "execution/preparation binding")
    events = verify_journal(read_regular(RUN_DIR / "journal.jsonl"), report)
    require(events[0]["kind"] == "run_started" and events[0]["prepare_sha256"] == PREP_SHA
            and events[-1]["kind"] == "run_finished" and events[-1]["errors"] == [],
            "run journal boundary")
    source_arrays, _, _ = run.source_data()
    require(sha(source_arrays["goal_rgb"].tobytes()) == prep["goal_rgb_sha256"],
            "sealed goal pixels")
    arrays = {"goal_rgb": source_arrays["goal_rgb"]}
    roles = []
    for arm, row in zip(ARMS, report["rows"]):
        verified = verify_arm(RUN_DIR, arm, row, events, prep["warmup_rgb_sha256"])
        arrays[arm + "_rgb_history"] = verified["rgb_history"]
        arrays[arm + "_raw_action_history"] = verified["raw_action_history"]
        cp5 = verified["cp5_observation"]
        roles.append({"arm": arm, "cp5_file": cp5["file"],
                      "cp5_file_sha256": cp5["file_sha256"],
                      "cp5_pixel_sha256": cp5["pixel_sha256"],
                      "action_history_sha256": sha(verified["raw_action_history"].tobytes())})
    metadata = {
        "schema": NAME + ".manifest.v1",
        "classification": "development_root_not_original_holdout_confirmation",
        "seed": run.SEED, "P1_admission": False, "independent_confirmation": False,
        "model_or_cuda_executed": False, "physical_execution": False,
        "source_report_sha256": REPORT_SHA, "source_preparation_sha256": PREP_SHA,
        "source_journal_sha256": report["journal_sha256"],
        "source_journal_tail": report["journal_tail"],
        "source_receipt_sha256": prep["source_receipt_sha256"],
        "source_goal_rgb_pixel_sha256": prep["goal_rgb_sha256"],
        "history_steps": ["warmup_5", "warmup_10", "control_5"],
        "action_packets": ["actual_warmup_6_to_10", "actual_control_1_to_5"],
        "roles": roles,
        "no_arm_selection": True,
        "limitation": "saved RGB/actions only; no encoded cp5 latent or replanning has been run",
    }
    return metadata, arrays


def export(directory):
    metadata, arrays = build()
    directory = Path(directory).absolute()
    require(not os.path.lexists(directory), "exclusive output directory")
    directory.mkdir()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    raw = buffer.getvalue()
    with (directory / "rgb_action_history.npz").open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    metadata["archive_sha256"] = sha(raw)
    manifest_raw = canonical(metadata) + b"\n"
    with (directory / "manifest.json").open("xb") as stream:
        stream.write(manifest_raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha(manifest_raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    require(args.tag.isascii() and args.tag.isalnum(), "ASCII alphanumeric tag")
    output = ROOT / "results" / f"{NAME}_EXPORT_{args.tag}"
    manifest_sha = export(output)
    print(json.dumps({"output": str(output), "manifest_sha256": manifest_sha,
                      "P1_admission": False}, sort_keys=True))


if __name__ == "__main__":
    main()
