"""Bind the actual goal-DLS initializer source into each structured contract."""
import argparse
import json
from pathlib import Path

from . import e3_cp10_goal_dls_20261031 as dls
from . import p1_four_arm_extra32_budget_20261031 as control

NAME = "e3_cp10_dls_sourcefix_20261031"
PROTOCOL = Path(__file__).resolve().parents[3] / "docs/D2P_E3_GOAL_DLS_SOURCE_BINDING_20261031.md"


def bind_initializer(contract, source_sha):
    if not control.spec.is_hash(source_sha):
        raise ValueError("initializer source SHA")
    contract["arms"]["structured"]["initializer_sha256"] = source_sha
    control.spec.preflight(contract, control.spec.digest(contract))
    return contract


def setup(root):
    experiment = dls.setup(root)
    runner = experiment.runner
    old_build, old_protocol, old_contract = runner.build, runner.protocol, control.contract_for
    source_sha = runner.sha(Path(dls.__file__).resolve())

    def contract_for(task, model_sha, hardware_sha):
        policy, contract, config = old_contract(task, model_sha, hardware_sha)
        return policy, bind_initializer(contract, source_sha), config

    def build():
        value = old_build()
        for path in (Path(__file__).resolve(), PROTOCOL):
            value["sources"][str(path)] = runner.sha(path)
        return value

    def protocol():
        value = old_protocol()
        value.update(structured_initializer_sha256=source_sha,
                     source_binding_fix=True, independent_confirmation=False)
        return value

    control.contract_for = contract_for
    runner.NAME = f"{NAME}_root{root}"
    runner.build, runner.protocol = build, protocol
    return experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=int, choices=(0, 1), required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    run = sub.add_parser("run")
    for key in ("prepared", "sha256", "output", "authorization"):
        run.add_argument("--" + key, required=True)
    args = parser.parse_args()
    experiment = setup(args.root)
    runner = experiment.runner
    if args.command == "prepare":
        result = runner.prepare(args.output)
    else:
        print("Source-binding repair replay on host CUDA; no scientific admission.", flush=True)
        with experiment.extra.gpu.locking.gpu_lock(experiment.extra.gpu.LOCK):
            result = runner.run(args.prepared, args.sha256, args.output, args.authorization)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.command == "run" and result["status"] != "recorded_known_root_not_admitted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
