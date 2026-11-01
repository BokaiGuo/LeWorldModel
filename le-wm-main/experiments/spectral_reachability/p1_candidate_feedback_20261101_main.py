"""All-index forced-prefix diagnostic followed by identical RGB replanning."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import p1_horizon_q_20261101_main as horizon
from . import p1_candidate_intervention_20261101_main as physical

NAME = "p1_candidate_feedback_20261101_main"
ROOT = horizon.parent.ROOT
DOC = ROOT.parent / "docs/D2P_P1_CANDIDATE_FEEDBACK_PROTOCOL_20261101_MAIN.md"


def setup(index):
    if type(index) is not int or not 0 <= index < 8:
        raise ValueError("fixed candidate index")
    source_report = json.loads((physical.DIRECTORY / "report.json").read_bytes())
    row = source_report["rows"][0]
    receipt = next(n["receipt"] for n in source_report["nodes"][row["episode_id"]] if n["boundary"] == 0)
    trace = Path(receipt["trace"])
    if physical.engine.sha(trace) != receipt["trace_sha256"]:
        raise ValueError("frozen source bank")
    events = [json.loads(x) for x in trace.read_bytes().splitlines()]
    bank = next(e["payload"]["raw"] for e in events if e["kind"] == "optimized_candidates")
    original_plan = horizon.parent.kernels.plan

    def plan(leaf, arm, task, config, policy, contract, *, device, emit):
        initial = config["planning_horizon_packets"] == 3

        def record(kind, payload):
            if initial and kind == "final_selection":
                emit("pre_intervention_model_selection", payload)
            else:
                emit(kind, payload)

        result = original_plan(leaf, arm, task, config, policy, contract, device=device, emit=record)
        if initial:
            physical.engine.require(config["intervention_candidate"] == index
                                    and np.array_equal(result["optimized_raw"].cpu().numpy(),
                                                       np.asarray(bank, np.float32)),
                                    "same sealed b0 bank before intervention")
            model_index = result["selected_index"]
            result.update(selected_raw=result["optimized_raw"][index].detach(), selected_index=index)
            emit("forced_prefix_intervention", dict(candidate=index, original_model_index=model_index,
                                                    source_trace_sha256=receipt["trace_sha256"],
                                                    not_model_selected=index != model_index,
                                                    P1_admission=False))
            emit("final_selection", dict(index=index, scores=result["scores"],
                                         id=policy["candidate_ids"][index],
                                         selection_rule="predeclared_all_eight_index_intervention"))
        return result

    horizon.parent.kernels.plan = plan
    experiment = horizon.setup("structured")
    horizon.base.CONFIG = dict(horizon.base.CONFIG, intervention_candidate=index)
    runner = experiment.runner
    original_build, original_protocol = runner.build, runner.protocol
    original_contract = horizon.control.contract_for
    source_sha = runner.sha(Path(__file__).resolve())

    def contract_for(task, model_sha, hardware_sha):
        policy, contract, config = original_contract(task, model_sha, hardware_sha)
        contract["identity"]["optimizer"] = source_sha
        for entry in contract["arms"].values():
            entry["optimizer_sha256"] = source_sha
        horizon.spec.preflight(contract, horizon.spec.digest(contract))
        return policy, contract, config

    def build():
        value = original_build()
        for path in (Path(__file__).resolve(), DOC, trace, physical.DIRECTORY / "report.json"):
            value["sources"][str(path)] = runner.sha(path)
        return value

    def protocol():
        value = original_protocol()
        value.update(intervention_candidate=index, fixed_candidate_roster=list(range(8)),
                     prefix_selection="forced_all_index_diagnostic_not_model_policy",
                     P1_admission=False, actual_cp5_rgb_refresh=True)
        return value

    horizon.control.contract_for = contract_for
    runner.build, runner.protocol = build, protocol
    runner.NAME = f"{NAME}_c{index}"
    return experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=int, choices=range(8), required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--authorize-host-cuda", action="store_true", required=True)
    args = parser.parse_args()
    if not args.tag.isalnum():
        raise ValueError("simple tag")
    experiment = setup(args.candidate)
    prefix = ROOT / "results" / f"{NAME}_c{args.candidate}"
    prepared = Path(str(prefix) + f"_PREP_{args.tag}.json")
    output = Path(str(prefix) + f"_RUN_{args.tag}")
    if prepared.exists() or output.exists():
        raise ValueError("exclusive output")
    receipt = experiment.runner.prepare(str(prepared))
    print(json.dumps(receipt), flush=True)
    with experiment.extra.gpu.locking.gpu_lock(experiment.extra.gpu.LOCK):
        result = experiment.runner.run(
            str(prepared), receipt["sha256"], str(output),
            "MAIN_AUTHORIZES_COMMON_SUFFIX_RGB_CUDA_AND_PHYSICS")
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "recorded_known_root_not_admitted" else 1)
