"""Known-root four-arm RGB feedback run using the spare-budget node."""
import argparse
import json
import time
from pathlib import Path

import torch

from . import codex_four_arm_common_rgb_launcher_20261030 as launcher
from . import p1_four_arm_hooked_cuda_20261030_main as hooks
from . import p1_four_arm_winner_budget_20261031 as control
from . import p1_four_arm_winner_budget_bridge_20261031 as bridge

NAME = "p1_four_arm_winner_budget_cuda_20261031"
ROOT = launcher.ROOT
PROTOCOL = ROOT.parent / "docs/D2P_FOUR_ARM_WINNER_BUDGET_PROTOCOL_20261031.md"
original_build = launcher.build
original_protocol = launcher.protocol


def plan_node(task, arm, boundary, store, runtime):
    device = runtime["device"]
    launcher.require(device.type == "cuda", "host CUDA required")
    torch.cuda.synchronize(device)
    started = time.monotonic()
    with hooks.ModuleCalls(runtime["model"]) as hook:
        result = bridge.plan_node(task, arm, boundary, store, runtime)
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    actual = hooks.validate(hook.counts, result["account"])
    audit = {"arm": arm, "boundary": boundary, "module_forward_pre_hooks": actual,
             "ledger_charged_upper_bounds": result["account"]["charged_upper_bounds"],
             "synchronized_wall_seconds_including_audit": elapsed,
             "backward_independently_measured": False, "full_cost_certified": False,
             "fairness_proven": False}
    store.put(f"s0_{arm}_b{boundary:02}_hook_audit.json", audit)
    result["module_hook_audit"] = audit
    return result


def protocol():
    value = original_protocol()
    costs = {arm: control.cost(arm) for arm in launcher.spec.ARMS}
    value.update(node_costs=costs,
                 node_caps={axis: max(costs[arm][axis] for arm in launcher.spec.ARMS)
                            for axis in launcher.spec.AXES},
                 extra_policy={arm: ("model_winner" if arm in control.EXTRA_ARMS else "none")
                               for arm in launcher.spec.ARMS},
                 module_hook_scope=["predictor", "action_encoder"], module_hooks_gate_each_node=True,
                 timing="synchronized wall time including planner and audit",
                 independent_backward_measurement=False, full_cost_certified=False)
    return value


def build():
    value = original_build()
    for path in (Path(__file__), Path(hooks.__file__), Path(control.__file__),
                 Path(bridge.__file__), PROTOCOL):
        path = path.resolve()
        value["sources"][str(path)] = launcher.sha(path)
    return value


def configure():
    launcher.NAME = NAME
    launcher.protocol = protocol
    launcher.build = build
    launcher.bridge.plan_node = plan_node


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    run = sub.add_parser("run")
    for key in ("prepared", "sha256", "output", "authorization"):
        run.add_argument("--" + key, required=True)
    args = parser.parse_args()
    configure()
    if args.command == "prepare":
        print(json.dumps(launcher.prepare(args.output)))
    else:
        print("Known-root CUDA RGB feedback with prespecified additional baseline optimization; "
              "not a fairness or P1 claim.", flush=True)
        result = launcher.run(args.prepared, args.sha256, args.output, args.authorization)
        print(json.dumps(result, sort_keys=True))
        if result["status"] != "recorded_known_root_not_admitted":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
