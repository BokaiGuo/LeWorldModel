"""Prospective all-stage Q objective on the eligible fixed return task."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import p1_return_goal_shard_20261101_main as parent
from . import p1_quadratic_kernels_20261101_main as kernels
from . import p1_first_feedback_carry_20261101_main as carry
from . import p1_response_diverse_carry_20261101_main as diverse
from . import e3_cp10_goal_dls_20261031 as dls

NAME = "p1_q_return_shard_20261101_main"
ROOT = Path(__file__).resolve().parents[2]
METRIC = ROOT / "results/p1_goal_cometric_cuda_20261101_main_MAIN01.json"
METRIC_SHA = "3ce857733d7ac413aa322e96fa4955f6db94bea53b55add7a5b0527148a21925"
TARGET = ROOT / "results/p1_return_target_cuda_20261101_main_MAIN06.json"
TARGET_SHA = "5bd5f5bbe82fe085cf26ceef12dc8ff6e0bb2fbc784e241ae4f5f8e4f693ed8a"
DOC = ROOT.parent / "docs/D2P_P1_Q_RETURN_PROTOCOL_20261101_MAIN.md"


def setup(arm):
    if parent.targets.source.base.sha(METRIC) != METRIC_SHA:
        raise ValueError("metric artifact")
    metric = json.loads(METRIC.read_bytes())
    if metric["status"] != "measured_not_control_validated" or metric["target_sha256"] != TARGET_SHA:
        raise ValueError("validated goal-only metric")
    kernels.MATRIX, kernels.GOAL, kernels.ARTIFACT_SHA = metric["sqrt_matrix"], metric["goal_latent"], METRIC_SHA
    control = carry.control
    control.plan, control.base.suffix.optimize, carry.selector.select = kernels.plan, kernels.optimize, kernels.select
    experiment = parent.setup(0, arm, str(TARGET), TARGET_SHA)
    runner = experiment.runner
    old_contract, old_build, old_protocol = control.contract_for, runner.build, runner.protocol
    matrix = np.asarray(kernels.MATRIX, np.float64)

    def goal_bank(jac, projection, goal, nominal):
        return diverse.response_bank(jac, matrix @ projection, goal, nominal)

    dls.goal_bank = goal_bank
    optimizer_sha = runner.sha(Path(kernels.__file__).resolve())
    initializer_sha = runner.sha(Path(__file__).resolve())

    def contract_for(task, model_sha, hardware_sha):
        policy, contract, config = old_contract(task, model_sha, hardware_sha)
        policy["metric"] = "goal_encoder_cometric_v1"
        contract["phase_policy_sha256"] = control.spec.digest(policy)
        contract["identity"]["optimizer"] = optimizer_sha
        for entry in contract["arms"].values():
            entry["optimizer_sha256"] = optimizer_sha
        contract["arms"]["structured"]["initializer_sha256"] = initializer_sha
        control.spec.preflight(contract, control.spec.digest(contract))
        return policy, contract, config

    def build():
        value = old_build()
        for path in (Path(__file__).resolve(), Path(kernels.__file__).resolve(), DOC, METRIC):
            value["sources"][str(path)] = runner.sha(path)
        return value

    def protocol():
        value = old_protocol()
        value.update(objective="goal_encoder_cometric_v1", metric_sha256=METRIC_SHA,
                     metric_construction_unamortized=dict(encoder_forwards=13, input_vjp_rows=192),
                     metric_applied_to=["spectral_generation", "suffix_optimization", "screening",
                                        "winner_refinement", "final_selection", "carry_selection"],
                     objective_changes_evaluator=False, fairness_proven=False)
        return value

    control.contract_for, runner.build, runner.protocol = contract_for, build, protocol
    runner.NAME = f"{NAME}_{arm}"
    return experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=parent.ARMS, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--authorize-host-cuda", action="store_true", required=True)
    args = parser.parse_args()
    if not args.tag.isalnum():
        raise ValueError("simple tag")
    exp = setup(args.arm)
    prefix = ROOT / "results" / f"{NAME}_{args.arm}"
    prepared, output = Path(str(prefix) + f"_PREP_{args.tag}.json"), Path(str(prefix) + f"_RUN_{args.tag}")
    if prepared.exists() or output.exists():
        raise ValueError("exclusive output")
    receipt = exp.runner.prepare(str(prepared))
    print(json.dumps(receipt), flush=True)
    with exp.extra.gpu.locking.gpu_lock(exp.extra.gpu.LOCK):
        result = exp.runner.run(str(prepared), receipt["sha256"], str(output),
                               "MAIN_AUTHORIZES_COMMON_SUFFIX_RGB_CUDA_AND_PHYSICS")
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "recorded_known_root_not_admitted" else 1)
