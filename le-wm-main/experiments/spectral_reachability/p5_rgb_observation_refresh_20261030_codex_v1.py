"""Fixed164 saved-RGB one-step refresh preparation and explicitly gated CUDA runner."""
import argparse
import hashlib
import os
import re
import time
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p5_rgb_cuda_review_gate_20261030_codex_v1 as old_gate

runner, abi = old_gate.runner, old_gate.abi
bridge = abi.bridge
NAME = "p5_rgb_observation_refresh_20261030_codex_v1"
ROOT = bridge.ROOT
AUTH = "MAIN_THREAD_APPROVED_FIXED164_OBSERVATION_REFRESH_CUDA"
OLD_GATE_SHA = "e557fd97b980c2f68da25733419c1be9225cc0096aca18864999147f5eb30d16"


def check(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_path(path, expected, limit):
    check(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit, "file")
    raw = path.read_bytes()
    check(digest(raw) == expected, "file SHA")
    return raw


def sources():
    check(digest(Path(old_gate.__file__).read_bytes()) == OLD_GATE_SHA, "old gate source SHA")
    old_gate.pinned_sources()
    prepared = runner.load_prepared()
    physical = bridge.read_json(bridge.read_fixed("physical_receipt"))
    panel = bridge.read_json(bridge.read_fixed("panel"))
    return prepared, physical, panel


def frame_at(row, step):
    frames = [f for f in row["trajectory"]
              if (f["phase"], f["step"], f["condition"], f["overlay_index"]) ==
              ("plan", step, row["condition"], int(row["condition"] == "on"))]
    check(len(frames) == 1, "frame time/condition identity")
    return frames[0]


def prepare(old, physical, panel, read_image=bridge.input_image):
    check(old["branch_denominator"] == physical["fixed_denominator"] == panel["branch_denominator"] == 164
          and old["pair_denominator"] == panel["pair_denominator"] == 80
          and len(old["rows"]) == len(physical["rows"]) == len(panel["rows"]) == 164
          and len(panel["pairs"]) == 80 and old["pairs"] == panel["pairs"]
          and physical["status"] == "observed_raw164_physical_recorded_not_admitted"
          and panel["status"] == "saved_input_physical_identity_bound_not_scientific_admission",
          "fixed source cohort")
    rows = []
    for i, (p, physical_row, binding) in enumerate(zip(old["rows"], physical["rows"], panel["rows"], strict=True)):
        keys = ("index", "root_slot", "condition", "task_id", "selection_digest")
        check(all(p[k] == physical_row[k] == binding[k] == physical_row["plan"][k] for k in keys)
              and (p["index"], p["root_slot"], p["condition"]) ==
              (i, i // 82, ("off", "on")[(i // 41) % 2])
              and physical_row["status"] == "horizon_completed" and physical_row["own_returned"] == 10
              and physical_row["initial_exact"] is True, "branch identity/completeness")
        actions = np.asarray(physical_row["plan"]["commands"], dtype="<f4")
        check(actions.shape == (10, 2) and np.isfinite(actions).all()
              and digest(actions.tobytes()) == binding["raw10_sha256"] == physical_row["plan"]["raw_float32_sha256"]
              and digest(actions[5:].tobytes()) == p["actions_sha256"], "actions binding")
        frames = [read_image(frame_at(physical_row, step)) for step in range(4, 10)]
        check(frames[0] == p["previous_rgb"] and frames[1] == p["current_rgb"], "MAIN02 history binding")
        steps = []
        for h in range(1, 6):
            act = actions[4 + h].copy()
            steps.append({"horizon": h, "previous_plan_step": h + 3, "current_plan_step": h + 4,
                          "target_plan_step": h + 5, "previous_rgb": frames[h - 1], "current_rgb": frames[h],
                          "action": act.tolist(), "action_sha256": digest(act.tobytes())})
        rows.append({**{k: p[k] for k in keys}, "steps": steps})
    for n, pair in enumerate(panel["pairs"]):
        slot, local = divmod(n, 20)
        check((pair["slot"], pair["root_slot"], pair["condition"], pair["coord"], pair["h"],
               pair["minus_index"], pair["plus_index"]) ==
              (slot, slot // 2, ("off", "on")[slot % 2], local % 10, (0.01, 0.001)[local // 10],
               slot * 41 + 1 + 2 * local, slot * 41 + 2 + 2 * local), "pair schedule")
    return {"schema": NAME + ".prepared", "status": "offline_inputs_only_not_authorized",
            "source_sha256": {"previous_prepared": runner.PREP_SHA, "physical_receipt": bridge.FILES["physical_receipt"][1],
                              "panel": bridge.FILES["panel"][1], "checkpoint": bridge.FILES["checkpoint"][1],
                              "model_source": bridge.FILES["model_source"][1],
                              "consumer_source": digest(Path(__file__).read_bytes()), "old_gate": OLD_GATE_SHA},
            "branch_denominator": 164, "pair_denominator": 80, "steps_per_branch": 5,
            "planned_calls": {"encode_state": 820, "predict_next": 820},
            "actual_calls": {"neural": 0, "cuda": 0, "physics": 0, "training": 0},
            "normalization_from_prior_strict_load": old["normalization_from_prior_strict_load"],
            "rows": rows, "pairs": panel["pairs"], "P5_admission": False}


def materialize(step):
    def image(record):
        path = Path(record["path"])
        check(path.parent == bridge.PHYSICAL_DIR and path.name not in ("", ".", ".."), "image path")
        current, array = bridge.input_image({"png": path.name, "file_sha256": record["file_sha256"],
                                             "pixel_sha256": record["pixel_sha256"]}, return_array=True)
        check(current == record, "image identity")
        return array[None, ...]
    action = np.asarray(step["action"], dtype="<f4")
    check(action.shape == (2,) and np.isfinite(action).all()
          and digest(action.tobytes()) == step["action_sha256"], "step action")
    return {"previous_image": image(step["previous_rgb"]), "image": image(step["current_rgb"]),
            "actions": action.reshape(1, 1, 2)}


def execute(prepared, prepared_sha, backend, journal, max_seconds, clock=time.monotonic, load=materialize):
    start = clock()
    journal.emit("start", schema=NAME + ".journal", prepared_sha256=prepared_sha,
                 source_sha256=digest(Path(__file__).read_bytes()), max_seconds=max_seconds,
                 backend=backend.metadata(), branch_denominator=164, pair_denominator=80)
    try:
        for row in prepared["rows"]:
            for step in row["steps"]:
                check(clock() - start < max_seconds, "between-call timeout")
                values = load(step)
                identity = {"branch": row["index"], "root_slot": row["root_slot"],
                            "condition": row["condition"], "task_id": row["task_id"],
                            "selection_digest": row["selection_digest"], "horizon": step["horizon"],
                            "previous_plan_step": step["previous_plan_step"],
                            "current_plan_step": step["current_plan_step"], "target_plan_step": step["target_plan_step"],
                            "previous_sha256": step["previous_rgb"]["model_input_chw_float32_sha256"],
                            "image_sha256": step["current_rgb"]["model_input_chw_float32_sha256"],
                            "action_sha256": step["action_sha256"]}
                for op in ("encode_state", "predict_next"):
                    check(clock() - start < max_seconds, "between-call timeout")
                    journal.emit("attempt", **identity, op=op)
                    try:
                        output = runner.vector(backend.encode(values) if op == "encode_state"
                                               else backend.step(values, 0))
                    except Exception as exc:
                        journal.emit("failed", **identity, op=op, error_type=type(exc).__name__, error=str(exc)[:500])
                        raise
                    journal.emit("returned", **identity, op=op, output=output)
        journal.emit("end", status="complete", elapsed_seconds=clock() - start)
        return "complete"
    except Exception as exc:
        journal.emit("end", status="partial", elapsed_seconds=clock() - start,
                     error_type=type(exc).__name__, error=str(exc)[:500])
        return "partial"


def target(name, suffix):
    check(re.fullmatch(NAME + r"_[A-Za-z0-9][A-Za-z0-9_-]*" + re.escape(suffix), name) is not None,
          "exclusive name")
    directory = ROOT / "results"
    check(directory.resolve() == directory and not directory.is_symlink(), "results path")
    return directory / name


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--name", required=True)
    p = sub.add_parser("run")
    p.add_argument("--prepared-name", required=True)
    p.add_argument("--prepared-sha256", required=True)
    p.add_argument("--output-name", required=True)
    p.add_argument("--authorization", required=True)
    p.add_argument("--device-index", type=int, required=True)
    p.add_argument("--max-seconds", type=int, required=True)
    args = ap.parse_args()
    if args.command == "prepare":
        path = target(args.name, ".json")
        data = bridge.canonical(prepare(*sources()))
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        runner.sync_dir(path.parent)
        print(digest(data))
        return
    check(args.authorization == AUTH and 0 <= args.device_index and 1 <= args.max_seconds <= 3600,
          "explicit main-thread authorization and resource limits")
    check(re.fullmatch(r"[0-9a-f]{64}", args.prepared_sha256) is not None, "expected prepared SHA")
    prepared_path = target(args.prepared_name, ".json")
    output = target(args.output_name, "")
    check(not output.exists() and not output.is_symlink(), "output exists")
    data = checked_path(prepared_path, args.prepared_sha256, 8 * 1024 * 1024)
    prepared = bridge.read_json(data)
    check(data == bridge.canonical(prepare(*sources())), "prepared exact rebuild")
    for row in prepared["rows"]:
        for step in row["steps"]:
            materialize(step)
    checkpoint, _ = runner.verify_sources()
    backend = runner.CudaBackend(checkpoint, prepared, args.device_index)
    journal = runner.Journal(output)
    try:
        status = execute(prepared, args.prepared_sha256, backend, journal, args.max_seconds)
    finally:
        journal.close()
    print("status:", status)
    if status != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
