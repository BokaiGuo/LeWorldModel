"""Offline input/source preflight and strict journal envelope review; never executes CUDA."""
import argparse
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p5_rgb_cuda_consumer_20261030_codex_v1 as consumer
from experiments.spectral_reachability import p5_rgb_cuda_runner_20261030_codex_v1 as runner

abi = runner.abi
NAME = "p5_rgb_cuda_review_gate_20261030_codex_v1"
PINS = {
    "experiments/spectral_reachability/p5_rgb_cuda_runner_20261030_codex_v1.py":
        "895a9d2af43ddfef4946cd08eaf5d6d963bbddf90522e8e39d560311b486ffdc",
    "experiments/spectral_reachability/p5_rgb_cuda_consumer_20261030_codex_v1.py":
        "6948764ced979945a3a9dc9d32fc1c825720203b12738aefb15a14f31ccb4ad2",
}
BASE = {"seq", "prev", "kind", "hash"}
IDENTITY = {"branch", "condition", "previous_sha256", "image_sha256", "actions_sha256", "op", "step"}
BACKEND = {"device", "device_name", "torch_version", "cuda_version", "strict_load",
           "frozen", "eval", "checkpoint_sha256"}


def pinned_sources():
    for name, expected in PINS.items():
        path = abi.bridge.ROOT / name
        abi.check(path.is_file() and not path.is_symlink() and
                  abi.bridge.digest(path.read_bytes()) == expected, "review source SHA: " + name)
    runner.verify_sources()


def preflight():
    pinned_sources()
    p = runner.load_prepared()
    per_condition = {c: {"branches": 0, "action_components": 0, "outside_unit_box": 0}
                     for c in ("off", "on")}
    for index, row in enumerate(p["rows"]):
        abi.check(row["index"] == index and row["root_slot"] == index // 82 and
                  row["condition"] == ("off", "on")[(index // 41) % 2], "fixed order")
        inputs = abi.materialize(row)
        a = inputs["actions"]
        group = per_condition[row["condition"]]
        group["branches"] += 1
        group["action_components"] += int(a.size)
        group["outside_unit_box"] += int((np.abs(a) > 1).sum())
    abi.check(all(group["branches"] == 82 and group["action_components"] == 820
                  for group in per_condition.values()), "condition denominators")
    comparisons = {}
    for key in ("previous_rgb", "current_rgb"):
        comparisons[key] = sum(
            p["rows"][base + offset][key]["model_input_chw_float32_sha256"] !=
            p["rows"][base + 41 + offset][key]["model_input_chw_float32_sha256"]
            for base in (0, 82) for offset in range(41))
    return {"schema": NAME + ".preflight", "status": "offline_review_only_not_execution_authorization",
            "prepared_sha256": runner.PREP_SHA, "source_pins": PINS,
            "branch_denominator": 164, "pair_denominator": 80,
            "condition_inputs": per_condition,
            "off_on_corresponding_input_hashes_differ": comparisons,
            "actual_neural_calls": 0, "actual_cuda_calls": 0, "actual_physics_calls": 0,
            "P5_admission": False,
            "boundary": "input checks and raw component counts do not establish training support or distribution equivalence"}


def strict_review(prepared, data, expected_sha):
    abi.check(isinstance(expected_sha, str) and len(expected_sha) == 64 and
              abi.bridge.digest(data) == expected_sha, "external journal SHA")
    abi.check(len(data) <= 16 * 1024 * 1024 and data.endswith(b"\n"), "journal bounds")
    lines = data.splitlines()
    abi.check(lines, "empty journal")
    for i, line in enumerate(lines):
        event = abi.bridge.read_json(line)
        abi.check(isinstance(event, dict) and abi.bridge.canonical(event).rstrip(b"\n") == line,
                  "canonical journal event")
        kind = event.get("kind")
        if kind == "start":
            fields = BASE | {"prepared_sha256", "max_seconds", "runner_source_sha256", "backend"}
            abi.check(i == 0 and isinstance(event.get("backend"), dict) and
                      set(event["backend"]) == BACKEND, "start envelope")
        elif kind == "attempt":
            fields = BASE | IDENTITY
        elif kind == "returned":
            fields = BASE | IDENTITY | {"output"}
        elif kind == "failed":
            fields = BASE | IDENTITY | {"error_type", "error"}
        elif kind == "end":
            fields = BASE | {"status", "elapsed_seconds"}
            if event.get("status") == "partial":
                fields |= {"error_type", "error"}
        else:
            raise ValueError("unknown event kind")
        abi.check(set(event) == fields, "unexpected or missing event field")
    result = consumer.consume(prepared, data)
    result["strict_envelope_verified"] = True
    result["review_gate_schema"] = NAME
    return result


def write_new(path, result):
    data = abi.bridge.canonical(result)
    with path.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    runner.sync_dir(path.parent)
    return abi.bridge.digest(data)


def main():
    ap = argparse.ArgumentParser()
    subs = ap.add_subparsers(dest="cmd", required=True)
    pre = subs.add_parser("preflight")
    pre.add_argument("--output-name", required=True)
    rev = subs.add_parser("review-journal")
    rev.add_argument("--run-name", required=True)
    rev.add_argument("--journal-sha256", required=True)
    rev.add_argument("--output-name", required=True)
    args = ap.parse_args()
    name = args.output_name
    abi.check(Path(name).name == name and name.startswith(NAME + "_") and name.endswith(".json"),
              "exclusive output name")
    output = abi.bridge.ROOT / "results" / name
    if args.cmd == "preflight":
        result = preflight()
    else:
        pinned_sources()
        name = args.run_name
        abi.check(Path(name).name == name and name.startswith(runner.NAME + "_"), "run name")
        directory = abi.bridge.ROOT / "results" / name
        path = directory / "events.jsonl"
        abi.check(directory.is_dir() and not directory.is_symlink() and path.is_file() and
                  not path.is_symlink() and path.stat().st_size <= 16 * 1024 * 1024, "journal path")
        result = strict_review(runner.load_prepared(), path.read_bytes(), args.journal_sha256)
    print(write_new(output, result))


if __name__ == "__main__":
    main()
