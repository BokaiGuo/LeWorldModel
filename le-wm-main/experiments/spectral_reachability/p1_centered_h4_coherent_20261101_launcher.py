"""Fresh-process coherent-initializer H4 launcher; no implicit CUDA execution."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from . import p1_centered_h4_parfit_20261101_launcher as centered
from . import p1_h4_coherent_20261101 as coherent
from . import p1_node_h4_coherent_20261101 as node

NAME = "p1_centered_h4_coherent_20261101_launcher"
PROTOCOL = centered.adapter.ROOT.parent / "docs/D2P_P1_COHERENT_CONTROL_PROTOCOL_20261101_MAIN.md"


def setup(arm):
    h4 = centered.h4
    # This mutates a module reference only in this fresh process.
    h4.node_impl = node
    experiment = centered.setup(arm)
    runner = experiment.runner
    control = h4.reverse.Q.carry.control
    node_sha = runner.sha(Path(node.__file__).resolve())
    response_sha = runner.sha(Path(coherent.__file__).resolve())
    initializer_sha = control.spec.digest({"response_source_sha256": response_sha,
                                            "feedback_mix_source_sha256": node_sha})
    old_contract, old_build, old_protocol = control.contract_for, runner.build, runner.protocol

    def contract_for(task, model_sha, hardware_sha):
        policy, contract, config = old_contract(task, model_sha, hardware_sha)
        contract["arms"]["structured"]["initializer_sha256"] = initializer_sha
        control.spec.preflight(contract, control.spec.digest(contract))
        return policy, contract, config

    def protocol():
        value = old_protocol()
        value.update(schema=NAME + ".protocol.v1", initializer_variant="packet_constant_coherent",
                     structured_initializer_sha256=initializer_sha,
                     structured_response_source_sha256=response_sha,
                     structured_initializer="Q_DLS_plus_minus_top3_packet_sum_Jacobian",
                     feedback_initializer="four_coherent_response_four_packet_constant_gaussian",
                     gaussian_rng="torch_CPU_count_4_1_2_repeat5",
                     coherent_initialization_only=True,
                     initializer_labels={"structured": "Q_DLS_plus_minus_top3_packet_coherent",
                                         "native": "zero_centered_packet_constant_gaussian",
                                         "random": "packet_constant_raw_gaussian_bank",
                                         "immediate_progress": "packet_constant_one_packet_top8_of16"},
                     P1_admission=False, independent_confirmation=False)
        return value

    def build():
        value = old_build()
        runner.require(value["schema"] == h4.NAME + ".prepare.v1", "shared H4 prepare ABI")
        value["coherent_task_schema"] = NAME + ".task.v1"
        value["protocol"] = protocol()
        for path in (Path(__file__), Path(node.__file__), Path(coherent.__file__), PROTOCOL):
            path = path.resolve()
            value["sources"][str(path)] = runner.sha(path)
        return value

    control.contract_for = contract_for
    runner.protocol, runner.build = protocol, build
    runner.NAME = f"{NAME}_{arm}"
    return experiment


def dry_prepare(experiment):
    runner, h4 = experiment.runner, centered.h4
    value = runner.build()
    runner.require(h4.node_impl is node
                   and value["sources"][str(Path(node.__file__).resolve())] == runner.sha(Path(node.__file__))
                   and value["sources"][str(Path(coherent.__file__).resolve())] == runner.sha(Path(coherent.__file__))
                   and value["sources"][str(PROTOCOL.resolve())] == runner.sha(PROTOCOL),
                   "coherent runtime and source binding")
    prepared = __import__("copy").deepcopy(value)
    data = runner.p3.source_data(value["source"])
    report, target, _, images, _ = centered.source_inputs()
    binding = centered.rebind(prepared, data, runner, report, target, images)
    control = h4.reverse.Q.carry.control
    contracts = {}
    for boundary in (0, 5, 10, 15):
        with patch.dict(control.base.CONFIG, h4.boundary_config(boundary)), \
                patch.object(control, "cost", lambda arm: h4.h4.node_cost(arm, boundary)):
            policy, contract, config = control.contract_for(
                prepared["source"]["tasks"][0], "a" * 64, "b" * 64)
        runner.require(contract["levels"][0]["caps"] ==
                       value["protocol"]["node_caps_by_boundary"][str(boundary)]
                       and policy["horizon"] == h4.h4.dimensions(boundary)[0]
                       and config.items() >= h4.boundary_config(boundary).items()
                       and control.spec.digest(config) == contract["identity"]["optimizer_config"]
                       and control.spec.digest(policy) == contract["phase_policy_sha256"]
                       and contract["arms"]["structured"]["initializer_sha256"] ==
                       value["protocol"]["structured_initializer_sha256"]
                       and contract["identity"]["optimizer"] == runner.sha(Path(node.__file__)),
                       "coherent H4 node contract/caps")
        contracts[str(boundary)] = control.spec.digest(contract)
    return dict(status="dry_prepared_not_written", schema=value["schema"],
                build_sha256=runner.digest(value), binding=binding,
                dry_contract_sha256=contracts, initializer_variant="packet_constant_coherent",
                initializer_sha256=value["protocol"]["structured_initializer_sha256"],
                protocol_sha256=runner.sha(PROTOCOL), cuda_used=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=centered.h4.h4.ARMS, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dry-prepare")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    run = sub.add_parser("run")
    for key in ("prepared", "sha256", "output", "authorization"):
        run.add_argument("--" + key, required=True)
    run.add_argument("--authorize-host-cuda", action="store_true", required=True)
    args = parser.parse_args()
    experiment = setup(args.arm)
    if args.command == "dry-prepare":
        result = dry_prepare(experiment)
    elif args.command == "prepare":
        result = experiment.runner.prepare(args.output)
    else:
        with experiment.extra.gpu.locking.gpu_lock(experiment.extra.gpu.LOCK):
            result = experiment.runner.run(args.prepared, args.sha256, args.output, args.authorization)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.command == "run" and result["status"] != "recorded_known_root_not_admitted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
