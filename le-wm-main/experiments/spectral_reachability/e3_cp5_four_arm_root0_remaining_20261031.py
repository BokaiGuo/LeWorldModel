"""Four-arm real-RGB feedback on a fixed, previously witnessed cp5 goal."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from . import e3_cp5_target_cuda_20261031 as targets
from . import e3_initial_goal_qualification_20261031 as qualification
from . import p1_four_arm_extra32_root0_20261031 as extra

runner = extra.runner
ROOT = runner.ROOT
NAME = "e3_cp5_four_arm_root0_remaining_20261031"
PROTOCOL = ROOT.parent / "docs/D2P_E3_CP5_FOUR_ARM_ROOT0_REMAINING_PROTOCOL_20261031.md"
TARGET = ROOT / "results/e3_cp5_target_cuda_20261031_RUN01.json"
TARGET_SHA = "ff788213aff290ebd636fce22e55de64d3fddae128d8492bfdc4aa995c586518"
PREVIOUS = ROOT / "results/e3_cp5_four_arm_root0_20261031_RUN01/report.json"
PREVIOUS_SHA = "97c249e57ae85ce7a676dc63d0f9880b5521d1c4a524cfefe7cbf13dab92a6cb"
original_execute = runner.execute
original_build = extra.build
original_protocol = extra.protocol
original_roster = runner.rgb.roster


def inputs():
    runner.require(runner.sha(PREVIOUS) == PREVIOUS_SHA, "previous report identity")
    previous = json.loads(PREVIOUS.read_bytes())
    runner.require(previous["status"] == "partial_or_failed"
                   and [(r["arm"], r["status"]) for r in previous["rows"]] ==
                   [("structured", "native_stopped_control"), ("native", "failed"),
                    ("random", "not_started"), ("immediate_progress", "not_started")],
                   "only previously unstarted arms may continue")
    entries, sources = targets.collect()
    runner.require(runner.sha(TARGET) == TARGET_SHA and len(entries) == 2,
                   "fixed CUDA target receipt")
    receipt = json.loads(TARGET.read_bytes())
    row = receipt["rows"][0]
    entry = entries[0]
    runner.require(receipt["status"] == "complete" and row["root"] == 0
                   and row["exact_float32"] is True
                   and row["pixel_sha256"] == entry["pixel_sha256"]
                   and np.array_equal(np.asarray(row["latent"], np.float32), entry["existing"]),
                   "goal pixel/latent binding")
    data = json.loads(qualification.INPUT.read_bytes())
    candidate = qualification.analyze(data)["exploratory_cp5_candidates"][0]
    runner.require(candidate["initial_success"] is False, "target initially successful")
    sources.update({str(TARGET): TARGET_SHA, str(PREVIOUS): PREVIOUS_SHA,
                    str(PROTOCOL): runner.sha(PROTOCOL),
                    str(Path(__file__).resolve()): runner.sha(Path(__file__).resolve())})
    return entry, row, candidate, sources


def rebind(prepared, data):
    entry, target, candidate, _ = inputs()
    source = json.loads(qualification.INPUT.read_bytes())
    original = source["root_provenance"][0]["provenance"]["root_native_branches"][0]["c0"]["state"]
    ref = data["references"][0]
    runner.require(np.array_equal(np.asarray(ref["states"][-1], np.float64),
                                  np.asarray(original, np.float64)), "same physical starting root")
    base = prepared["source"]["tasks"][0]
    goal = list(target["latent"])
    ref["goal"] = np.asarray(candidate["goal_state"], dtype=np.float64)
    ref["goal_rgb"] = entry["pixels"].copy()
    ref["root_sha256"] = runner.p3.moving.source.root_hash(ref["states"][-1], ref["goal"])
    base["goal"] = goal
    base["binding"]["goal_sha256"] = runner.digest(goal)
    base["binding"]["root_sha256"] = ref["root_sha256"]
    base["binding"]["task_id"] = runner.digest([NAME, base["binding"]["task_id"], target["pixel_sha256"]])
    base["binding"]["task_sha256"] = runner.digest([NAME, "task", base["binding"]["task_id"]])
    base["binding"]["task_contract_sha256"] = runner.digest([NAME, "contract", target["pixel_sha256"]])
    runner.p3.core.validate_task(base)
    return {"goal_pixel_sha256": target["pixel_sha256"],
            "goal_latent_sha256": runner.digest(goal), "root_sha256": ref["root_sha256"],
            "initial_success": False}


def execute(prepared, data, store, visual, runtime, factory):
    # Never alter the sealed preparation or original source_data objects.
    prepared = deepcopy(prepared)
    data = deepcopy(data)
    binding = rebind(prepared, data)
    store.put("cp5_goal_binding.json", binding)
    result = original_execute(prepared, data, store, visual, runtime, factory)
    result["cp5_goal_binding"] = binding
    result["E3_admission"] = False
    return result


def protocol():
    result = original_protocol()
    result.update(goal="root0_previously_observed_cp5", initial_native_success=False,
                  arms=["random", "immediate_progress"],
                  E3_admission=False, independent_confirmation=False)
    return result


def build():
    result = original_build()
    _, _, _, sources = inputs()
    result["sources"].update(sources)
    return result


def configure():
    extra.configure()
    runner.NAME = NAME
    runner.protocol = protocol
    runner.build = build
    runner.execute = execute
    runner.rgb.roster = lambda: [row for row in original_roster()
                                 if row["arm"] in ("random", "immediate_progress")]


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
