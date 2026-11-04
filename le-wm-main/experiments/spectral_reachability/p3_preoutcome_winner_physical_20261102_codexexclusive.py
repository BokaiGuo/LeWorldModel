"""Four saved P3 winners on known moving roots; CPU-only, no neural inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from . import p3_preoutcome_cuda_host_20261102_codexexclusive as producer
from . import p3_preoutcome_saved_consumer_20261102_codexexclusive as model_consumer
from . import p3_venv_winner_physical_prepare_20261102_codexexclusive as prior
from . import p1_constructive_moving_history_physical_20261027_v1 as moving
from . import p1_moving_extended_baseline_physical_20261027_v1 as checks

ROOT = producer.ROOT
NAME = "p3_preoutcome_winner_physical_20261102_codexexclusive"
RUN = ROOT / "results/p3_preoutcome_cuda_host_20261102_codexexclusive_RUN_01"
CONSUMED = ROOT / "results/p3_preoutcome_cuda_host_20261102_codexexclusive_CONSUME_01.json"
MODEL_REPORT_SHA = "202ffc7a6c8fb7f26301a137028862a08cd4e014ca95c4387220b4adc49d8729"
MODEL_PREP_SHA = "f5f3c8caa423a413f972967f5a86c0cc9ce1e615c71139d51e74ee852ba6a11a"
MODEL_CONSUME_SHA = "85b16d3875cf360059fd166d6f961588dea94c4988d8e725ee8ce07380f2bed7"
AUTH = "P3_PREOUTCOME_KNOWN_ROOT_FOUR_WINNER_CPU_PHYSICS"


def require(value, message):
    if not value:
        raise ValueError(message)


def make_plans(checked, bank, slots, references, bridge, raw_commands):
    require(len(slots) == len(references) == len(bank) == 2, "two fixed roots")
    bindings, plans = {}, []
    for root in (0, 1):
        require(slots[root]["source_slot"] == root
                and slots[root]["root_sha256"] == references[root]["root_sha256"], "root identity")
        bridge(slots[root], references[root], bindings)
    for arm in ("signal", "reverse"):
        winners = checked["winners"][arm]
        require(len(winners) == 2 and [w["root"] for w in winners] == [0, 1], "winner roots")
        for root, winner in enumerate(winners):
            idx = winner["index"]
            require(type(idx) is int and 0 <= idx < checked["allocation"][arm][root]
                    and bank[root][idx]["index"] == idx, "winner fixed-prefix binding")
            raw = raw_commands(bank[root][idx]["commands"])
            require(raw.shape == (10, 2) and str(raw.dtype) == "float32", "action ABI")
            item = {"arm": arm, "root_slot": root, "root_sha256": slots[root]["root_sha256"],
                    "task_id": slots[root]["task_id"], "task_export_sha256": slots[root]["task_sha256"],
                    "candidate_index": idx, "model_cost": winner["score"],
                    "allocation": checked["allocation"][arm], "commands": raw.astype(float).tolist(),
                    "candidate_sha256": producer.control.digest(bank[root][idx]),
                    "raw_float32_sha256": hashlib.sha256(raw.tobytes()).hexdigest()}
            item["selection_digest"] = moving.source.digest(item)
            plans.append(item)
    return plans, bindings


def material():
    require(moving.sha(CONSUMED) == MODEL_CONSUME_SHA, "model consumer SHA")
    checked = model_consumer.consume(RUN, MODEL_REPORT_SHA, MODEL_PREP_SHA, producer.verify)
    require(moving.read(CONSUMED) == checked
            and checked["status"] == "saved_controller_consistency_known_roots", "model consumer reconstruction")
    _, _, banks = producer.verify(RUN / "preoutcome_prepare.json", MODEL_PREP_SHA)
    require(moving.sha(producer.host.PREVIOUS) == producer.old.PREDECESSOR_SHA, "source predecessor")
    slots = producer.host.prior.prior.read(producer.host.PREVIOUS)["slots"]
    data = moving.load_source(prior.source.SOURCEPACK, prior.source.SOURCEPACK_SHA,
                              prior.source.SOURCE_PREPARE, prior.source.SOURCE_PREPARE_SHA)
    plans, bound = make_plans(checked, banks, slots, data["references"], prior.physical.task_bridge,
                              producer.host.prior.prior.raw_commands)
    native, native_sha = checks.native_constructor_contract()
    bindings = dict(data["bindings"])
    bindings.update(bound)
    bindings.update({str(RUN / "report.json"): MODEL_REPORT_SHA,
                     str(RUN / "preoutcome_prepare.json"): MODEL_PREP_SHA,
                     str(RUN / "controller.jsonl"): moving.sha(RUN / "controller.jsonl"),
                     str(RUN / "events.jsonl"): moving.sha(RUN / "events.jsonl"),
                     str(CONSUMED): MODEL_CONSUME_SHA, native: native_sha})
    for path in (Path(__file__), Path(__file__).with_name(
            "p3_preoutcome_winner_physical_saved_20261102_codexexclusive.py"),
            Path(moving.__file__), Path(moving.low.__file__), Path(moving.moving.__file__)):
        bindings[str(path.resolve())] = moving.sha(path)
    for path, sha in bindings.items():
        require(moving.sha(path) == sha, "physical binding drift: " + path)
    report = moving.read(RUN / "report.json")
    model_result = model_consumer.read(RUN / "controller.jsonl")[-1]["result"]
    contract = moving.contract(600)
    contract.update(scope="four fixed model winners, two exposed roots, no retries or outcome-selected actions",
                    no_gpu=True, physical_replans=0,
                    native_stop_policy="stop at first terminated/truncated; retain partial prefix and censored endpoint",
                    failure_policy="preserve all attempted/completed counts, failed rows and later branches as recorded")
    return ({"schema": NAME + ".prepare.v1", "plans": plans, "contract": contract, "bindings": bindings,
             "sourcepack_sha256": prior.source.SOURCEPACK_SHA,
             "source_prepare_sha256": prior.source.SOURCE_PREPARE_SHA,
             "model_report_sha256": MODEL_REPORT_SHA, "model_prepare_sha256": MODEL_PREP_SHA,
             "model_consumer_sha256": MODEL_CONSUME_SHA,
             "model_accounting": {"resources_by_arm_and_phase": model_result["resources"],
                                  "model_host_elapsed_seconds": report["elapsed_seconds"],
                                  "peak_allocated_bytes": report.get("peak_allocated_bytes"),
                                  "peak_reserved_bytes": report.get("peak_reserved_bytes"),
                                  "unmetered": ["model setup and checkpoint loading", "internal model operations",
                                                "energy", "complete end-to-end FLOPs"]},
             "selection_fixed_before_physics": True, "model_outcomes_already_open": True,
             "physical_execution": False, "new_neural_calls": 0,
             "equal_total_compute": False, "independent_outcome": False,
             "P3_admission": False, "C3_admission": False}, data)


def output(path, prefix):
    p = Path(path)
    require(p.is_absolute() and p.parent == ROOT / "results" and p.name.startswith(NAME + prefix)
            and not os.path.lexists(p), "exclusive output")
    return p


def verify(path, sha):
    require(moving.sha(path) == sha, "external prepare SHA")
    saved = moving.read(path)
    current, data = material()
    require(saved == current, "prepare drift")
    return saved, data


def run(prepared, sha, destination, authority):
    require(authority == AUTH, "explicit physical authority")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA must be hidden")
    plan, data = verify(prepared, sha)
    out = output(destination, "_RUN_")
    store = moving.Store(out)
    store.put("execution_binding.json", {"prepared": str(Path(prepared).resolve()), "prepare_sha256": sha})
    def factory():
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                        with_target=False, render_action=False)
    return moving.execute(plan, data, store, factory, authorization=moving.AUTH)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "verify", "run"))
    p.add_argument("--output")
    p.add_argument("--prepared")
    p.add_argument("--sha256")
    p.add_argument("--authority")
    a = p.parse_args()
    if a.action == "prepare":
        plan, _ = material()
        path = output(a.output, "_PREP_")
        print(json.dumps({"path": str(path), "sha256": producer.host.write_new(path, plan)}))
    elif a.action == "verify":
        plan, _ = verify(a.prepared, a.sha256)
        print(json.dumps({"status": "prepared_only", "branches": len(plan["plans"]), "physical_run": False}))
    else:
        result = run(a.prepared, a.sha256, a.output, a.authority)
        print(json.dumps({"status": result["status"], "errors": result["errors"]}))


if __name__ == "__main__":
    main()
