"""Verify one queue-bound development root by exact replay and export blind inputs.

This module cannot consume confirmation slots and never grants P1 admission.
"""
import argparse
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_constructive_blind_encode_cuda_20261027_v1 as blind
from . import p1_dev_reset_bound_20261101_codex as native
from . import p1_queue_goal_history_20261101_codex as producer
from . import p1_queue_goal_saved_20261101_codex as saved

NAME = "p1_source_replay_blind_20261101_exclusive"
ROOT = producer.q.ROOT
require = producer.q.require


def read_image(directory, binding):
    path = (Path(directory) / binding["file"]).resolve(strict=True)
    require(path.parent == Path(directory).resolve(), "image path escape")
    require(producer.sha(path) == binding["file_sha256"], "image file digest")
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    require(pixels.shape == (224, 224, 3), "image shape")
    require(hashlib.sha256(pixels.tobytes()).hexdigest() == binding["pixel_sha256"],
            "image pixel digest")
    return pixels


def source_paths(queue_path, slot_id, attempt):
    queue_path = Path(queue_path).resolve(strict=True)
    directory = (queue_path.parent / (queue_path.name + ".native_reset")
                 / f"{slot_id}_{attempt:04d}.warmup")
    return directory, directory / "reset_receipt.json", directory / "source_receipt.json"


def replay(queue_path, queue_sha, slot_id, attempt, report_sha, output, *, factory=None):
    """Replay a completed development source in a new environment instance."""
    audit = saved.verify(queue_path, queue_sha, slot_id, attempt, report_sha)
    require(audit["independent_confirmation"] is False and audit["P1_admission"] is False,
            "development audit boundary")
    directory, reset_path, source_path = source_paths(queue_path, slot_id, attempt)
    reset = producer.q.read_json(reset_path)
    source = producer.q.read_json(source_path)
    expected = source["history"]
    require(expected["reset_receipt_sha256"] == producer.sha(reset_path), "reset binding")
    require(expected["actions"] == [[0.0, 0.0]] * 10, "fixed zero warmup")

    output = Path(output).resolve()
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    journal_path = output / "journal.jsonl"
    events = []
    native.emit(journal_path, events, "replay_started", queue_sha256=queue_sha,
                report_sha256=report_sha, slot_id=slot_id, attempt=attempt)
    env = None
    exact = False
    error = None
    frames_checked = 0
    steps_returned = 0
    try:
        cls = __import__("stable_worldmodel.envs.pusht.env", fromlist=["PushT"]).PushT
        env = cls(render_mode="rgb_array", resolution=224, with_target=False) if factory is None else factory()
        native.emit(journal_path, events, "reset_dispatch", seed=audit["seed"])
        obs, info = env.reset(seed=audit["seed"])
        state = np.asarray(obs["state"], dtype=np.float64)
        goal = np.asarray(info["goal_state"], dtype=np.float64)
        goal_rgb = np.asarray(info["goal"], dtype=np.uint8)
        require(state.tolist() == reset["state"] and goal.tolist() == reset["goal"],
                "reset state or goal replay")
        require(native.snapshot(env) == reset["snapshot"], "reset snapshot replay")
        require(np.array_equal(goal_rgb, read_image(directory, reset["goal_rgb"])),
                "goal RGB replay")
        native.emit(journal_path, events, "reset_return", state_sha256=producer.q.digest(state.tolist()),
                    goal_sha256=producer.q.digest(goal.tolist()))

        frame_by_step = {row["step"]: row for row in expected["frames"]}

        def check_frame(step):
            nonlocal frames_checked
            before = native.snapshot(env)
            pixels = np.asarray(env.render(), dtype=np.uint8)
            after = native.snapshot(env)
            row = frame_by_step[step]
            require(before == after == row["snapshot"], "frame snapshot replay")
            require(before["state"] == row["state"], "frame state replay")
            require(np.array_equal(pixels, read_image(directory, row)), "frame pixel replay")
            frames_checked += 1
            native.emit(journal_path, events, "frame_verified", step=step,
                        pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())

        check_frame(0)
        for step in range(1, 11):
            action = np.zeros(2, dtype=np.float32)
            native.emit(journal_path, events, "step_dispatch", step=step,
                        raw_float32_bits=action.tobytes().hex())
            obs, _, terminated, truncated, info = env.step(action.copy())
            steps_returned += 1
            actual = np.asarray(env.latest_action)
            require(actual.dtype == np.float32 and actual.tobytes() == action.tobytes(),
                    "executed action replay")
            require(not terminated and not truncated, "unexpected replay stop")
            if step == 10:
                require(np.asarray(obs["state"], np.float64).tolist()
                        == expected["frames"][-1]["state"], "terminal state replay")
            require(np.array_equal(np.asarray(info["goal_state"], np.float64), goal)
                    and np.array_equal(np.asarray(info["goal"], np.uint8), goal_rgb),
                    "goal drift during replay")
            native.emit(journal_path, events, "step_return", step=step,
                        state=np.asarray(obs["state"], np.float64).tolist())
            if step in frame_by_step:
                check_frame(step)
        require(native.snapshot(env) == source["final_snapshot"], "final snapshot replay")
        require(frames_checked == 3 and steps_returned == 10, "complete replay")
        exact = True
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)[:500]}
        native.emit(journal_path, events, "replay_failed", **error)
    finally:
        if env is not None:
            env.close()
    native.emit(journal_path, events, "replay_finished", exact=exact)
    result = {
        "schema": NAME + ".replay.v1",
        "implementation_sha256": producer.sha(Path(__file__)),
        "status": "exact_development_replay" if exact else "replay_failed",
        "exact": exact,
        "error": error,
        "queue_sha256": queue_sha,
        "source_report_sha256": report_sha,
        "source_receipt_sha256": producer.sha(source_path),
        "slot_id": slot_id,
        "attempt": attempt,
        "seed": audit["seed"],
        "steps_returned": steps_returned,
        "frames_verified": frames_checked,
        "journal_sha256": producer.sha(journal_path),
        "journal_tail": events[-1]["sha256"],
        "historical_coverage": "unknown",
        "training_membership": "unknown",
        "freshness": "unknown",
        "confirmation_authorized": False,
        "independent_confirmation": False,
        "P1_admission": False,
    }
    producer.q.write_new(output / "report.json", result)
    return result


def export_blind(queue_path, slot_id, attempt, replay_dir, replay_sha, output, manifest):
    replay_path = Path(replay_dir).resolve(strict=True) / "report.json"
    require(producer.sha(replay_path) == replay_sha, "external replay report SHA")
    replay_report = producer.q.read_json(replay_path)
    required = {"status", "exact", "implementation_sha256", "queue_sha256", "slot_id", "attempt",
                "source_receipt_sha256", "independent_confirmation", "P1_admission"}
    require(required <= set(replay_report), "exact development replay required")
    require(replay_report["status"] == "exact_development_replay"
            and replay_report["exact"] is True
            and replay_report["queue_sha256"] == producer.sha(queue_path)
            and replay_report["slot_id"] == slot_id
            and replay_report["attempt"] == attempt
            and replay_report["implementation_sha256"] == producer.sha(Path(__file__))
            and replay_report["independent_confirmation"] is False
            and replay_report["P1_admission"] is False, "exact development replay required")
    directory, _, source_path = source_paths(queue_path, slot_id, attempt)
    source = producer.q.read_json(source_path)
    require(producer.sha(source_path) == replay_report["source_receipt_sha256"],
            "source changed after replay")
    history_value = source["history"]
    frames = history_value["frames"]
    require([row["step"] for row in frames] == [0, 5, 10], "frame roster")
    arrays = {
        "rgb_history": np.stack([read_image(directory, row) for row in frames]),
        "raw_action_history": np.asarray(history_value["actions"], dtype=np.float32),
        "goal_rgb": read_image(directory, history_value["goal_rgb"]),
    }
    require(arrays["rgb_history"].shape == (3, 224, 224, 3)
            and arrays["rgb_history"].dtype == np.uint8, "blind RGB history")
    require(arrays["goal_rgb"].shape == (224, 224, 3)
            and arrays["goal_rgb"].dtype == np.uint8, "blind goal RGB")
    require(arrays["raw_action_history"].shape == (10, 2)
            and arrays["raw_action_history"].dtype == np.float32
            and not arrays["raw_action_history"].any(), "blind zero action history")
    payload = BytesIO()
    np.savez_compressed(payload, **arrays)
    raw = payload.getvalue()
    output = Path(output).resolve()
    manifest = Path(manifest).resolve()
    require(output != manifest and not output.exists() and not manifest.exists(), "exclusive outputs")
    with output.open("xb") as stream:
        require(stream.write(raw) == len(raw), "short blind write")
        stream.flush()
        os.fsync(stream.fileno())
    payload_sha = producer.sha(output)
    decoded = blind.load_blind(output, payload_sha)
    require(set(decoded) == set(arrays)
            and all(np.array_equal(decoded[key], value) for key, value in arrays.items()),
            "blind payload readback")
    result = {
        "schema": NAME + ".blind.v1",
        "implementation_sha256": replay_report["implementation_sha256"],
        "status": "blind_inputs_verified_development_only",
        "blind_path": str(output),
        "blind_sha256": payload_sha,
        "queue_sha256": replay_report["queue_sha256"],
        "slot_id": slot_id,
        "attempt": attempt,
        "replay_report_sha256": replay_sha,
        "source_receipt_sha256": replay_report["source_receipt_sha256"],
        "members": sorted(arrays),
        "observed_steps": [0, 5, 10],
        "actual_action_count": 10,
        "contains_seed": False,
        "contains_physical_state": False,
        "contains_snapshot": False,
        "planner_authorized": False,
        "confirmation_authorized": False,
        "independent_confirmation": False,
        "P1_admission": False,
    }
    producer.q.write_new(manifest, result)
    return result


def qualify(blind_manifest, blind_sha, history_inventory, history_sha,
            authority_snapshot, authority_sha):
    blind_path = Path(blind_manifest).resolve(strict=True)
    history_path = Path(history_inventory).resolve(strict=True)
    authority_path = Path(authority_snapshot).resolve(strict=True)
    require(producer.sha(blind_path) == blind_sha, "blind manifest SHA")
    require(producer.sha(history_path) == history_sha, "history inventory SHA")
    require(producer.sha(authority_path) == authority_sha, "authority snapshot SHA")
    blind_value = producer.q.read_json(blind_path)
    history_value = producer.q.read_json(history_path)
    authority = producer.q.read_json(authority_path)
    require(blind_value["status"] == "blind_inputs_verified_development_only"
            and blind_value["members"] == ["goal_rgb", "raw_action_history", "rgb_history"]
            and blind_value["implementation_sha256"] == producer.sha(Path(__file__))
            and blind_value["confirmation_authorized"] is False, "blind boundary")
    payload_path = Path(blind_value["blind_path"]).resolve(strict=True)
    require(producer.sha(payload_path) == blind_value["blind_sha256"], "blind payload SHA")
    payload = blind.load_blind(payload_path, blind_value["blind_sha256"])
    require(set(payload) == {"goal_rgb", "raw_action_history", "rgb_history"},
            "blind payload fields")
    require(history_value["schema"] == "codex_p1_independent_root_inputs_20261101_exclusive.v1"
            and history_value["independent_roots_certified"] is False
            and history_value["global_history_coverage"] == "unknown", "history boundary")
    assessment = authority["inventory_assessment"]
    require(authority["schema"] == "codex_p1_source_authority_gap_20261030.snapshot.v1"
            and assessment["history_complete"] is False
            and assessment["training_lineage_verified"] is False
            and assessment["source_run_authorized"] is False, "authority boundary")
    blockers = [
        "checkpoint_to_training_run_and_dataset_lineage_unverified",
        "training_episode_and_normalizer_membership_unknown",
        "historical_reset_coverage_not_exhaustive",
        "independent_origin_and_external_custody_not_verified",
        "confirmation_slot_execution_not_authorized",
    ]
    return {
        "schema": NAME + ".qualification.v1",
        "implementation_sha256": blind_value["implementation_sha256"],
        "status": "development_source_replay_blind_ready_confirmation_blocked",
        "engineering_ready": True,
        "blind_manifest_sha256": blind_sha,
        "history_inventory_sha256": history_sha,
        "authority_snapshot_sha256": authority_sha,
        "history_reset_records": history_value["baseline"]["reset_records"],
        "history_overlay_records": len(history_value["additional_overlay"]["events"]),
        "training_missing": assessment["training_missing"],
        "historical_calls_missing": assessment["calls_missing"],
        "scope_missing": assessment["scope_missing"],
        "blockers": blockers,
        "unknown_training_source_is_not_independence": True,
        "development_seed_is_not_independence": True,
        "confirmation_authorized": False,
        "independent_confirmation": False,
        "P1_admission": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    replay_cmd = sub.add_parser("replay")
    for name in ("queue", "queue-sha256", "slot", "report-sha256", "output"):
        replay_cmd.add_argument("--" + name, required=True)
    replay_cmd.add_argument("--attempt", required=True, type=int)
    replay_cmd.add_argument("--authorize-development-replay", action="store_true", required=True)
    blind_cmd = sub.add_parser("blind")
    for name in ("queue", "slot", "replay-dir", "replay-sha256", "output", "manifest"):
        blind_cmd.add_argument("--" + name, required=True)
    blind_cmd.add_argument("--attempt", required=True, type=int)
    qualify_cmd = sub.add_parser("qualify")
    for name in ("blind-manifest", "blind-sha256", "history-inventory", "history-sha256",
                 "authority-snapshot", "authority-sha256", "output"):
        qualify_cmd.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.command == "replay":
        with native.locking.gpu_lock(native.LOCK):
            result = replay(args.queue, args.queue_sha256, args.slot, args.attempt,
                            args.report_sha256, args.output)
    elif args.command == "blind":
        result = export_blind(args.queue, args.slot, args.attempt, args.replay_dir,
                              args.replay_sha256, args.output, args.manifest)
    else:
        result = qualify(args.blind_manifest, args.blind_sha256, args.history_inventory,
                         args.history_sha256, args.authority_snapshot, args.authority_sha256)
        producer.q.write_new(args.output, result)
    print(json.dumps(result, sort_keys=True))
    if result.get("status") == "replay_failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
