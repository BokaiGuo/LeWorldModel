"""Run fixed E3 cp10-goal root1 four-arm development experiment."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from . import e3_cp10_target_cuda_20261031 as targets
from . import p1_four_arm_extra32_root1_20261031 as extra

runner = extra.runner
ROOT = runner.ROOT
NAME = "e3_cp10_four_arm_root1_20261031"
PROTOCOL = ROOT.parent / "docs/D2P_E3_CP10_FOUR_ARM_ROOT1_PROTOCOL_20261031.md"
TARGET = ROOT / "results/e3_cp10_target_cuda_20261031_RUN01.json"
TARGET_SHA = "a0a524ef256dc8737995ce2f0daac5c15ad7578e2234be42c19c19cfbc864aef"
original_execute, original_build, original_protocol = runner.execute, extra.build, extra.protocol


def inputs():
    entries, sources = targets.collect()
    runner.require(runner.sha(TARGET) == TARGET_SHA, "target receipt SHA")
    receipt = json.loads(TARGET.read_bytes())
    row, entry = receipt["rows"][1], entries[1]
    runner.require(receipt["schema"] == targets.NAME and receipt["status"] == "complete"
                   and receipt["E3_admission"] is False and row["root"] == entry["root"] == 1
                   and row["pixel_sha256"] == entry["pixel_sha256"]
                   and row["goal_state"] == entry["goal_state"]
                   and row["root_state"] == entry["root_state"], "target identity")
    latent = np.asarray(row["latent"], dtype=np.float32)
    runner.require(latent.shape == (192,) and np.isfinite(latent).all(), "target latent")
    sources.update({str(TARGET): TARGET_SHA, str(PROTOCOL): runner.sha(PROTOCOL),
                    str(Path(__file__).resolve()): runner.sha(Path(__file__).resolve())})
    return entry, row, sources


def rebind(prepared, data):
    entry, target, _ = inputs()
    ref = data["references"][1]
    runner.require(np.array_equal(np.asarray(ref["states"][-1], np.float64),
                                  np.asarray(entry["root_state"], np.float64)), "physical root identity")
    task = prepared["source"]["tasks"][1]
    goal = list(target["latent"])
    ref["goal"] = np.asarray(entry["goal_state"], dtype=np.float64)
    ref["goal_rgb"] = entry["pixels"].copy()
    ref["root_sha256"] = runner.p3.moving.source.root_hash(ref["states"][-1], ref["goal"])
    task["goal"] = goal
    task["binding"]["goal_sha256"] = runner.digest(goal)
    task["binding"]["root_sha256"] = ref["root_sha256"]
    task["binding"]["task_id"] = runner.digest([NAME, task["binding"]["task_id"], target["pixel_sha256"]])
    task["binding"]["task_sha256"] = runner.digest([NAME, "task", task["binding"]["task_id"]])
    task["binding"]["task_contract_sha256"] = runner.digest([NAME, "contract", target["pixel_sha256"]])
    runner.p3.core.validate_task(task)
    return {"goal_pixel_sha256": target["pixel_sha256"], "goal_latent_sha256": runner.digest(goal),
            "root_sha256": ref["root_sha256"], "initial_success": False}


def execute(prepared, data, store, visual, runtime, factory):
    prepared, data = deepcopy(prepared), deepcopy(data)
    binding = rebind(prepared, data)
    store.put("cp10_goal_binding.json", binding)
    result = original_execute(prepared, data, store, visual, runtime, factory)
    result["cp10_goal_binding"] = binding
    result["E3_admission"] = False
    return result


def protocol():
    value = original_protocol()
    value.update(goal="root1_previously_observed_cp10", initial_native_success=False,
                 E3_admission=False, independent_confirmation=False)
    return value


def build():
    result = original_build()
    _, _, sources = inputs()
    result["sources"].update(sources)
    return result


def configure():
    extra.configure()
    runner.NAME = NAME
    runner.protocol, runner.build, runner.execute = protocol, build, execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    run = sub.add_parser("run")
    for key in ("prepared", "sha256", "output", "authorization"):
        run.add_argument("--" + key, required=True)
    args = parser.parse_args()
    configure()
    if args.command == "prepare":
        result = runner.prepare(args.output)
    else:
        with extra.gpu.locking.gpu_lock(extra.gpu.LOCK):
            result = runner.run(args.prepared, args.sha256, args.output, args.authorization)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.command == "run" and result["status"] != "recorded_known_root_not_admitted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
