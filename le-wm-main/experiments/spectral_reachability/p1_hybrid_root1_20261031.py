"""Four-arm feedback with additional baseline optimization and grad-call metering."""
import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

import torch

from . import codex_four_arm_common_rgb_root1_20261030_main as runner
from . import p1_four_arm_hooked_cuda_20261030_main as hooks
from . import p1_hybrid_spectral_20261031 as control
from . import p1_hybrid_bridge_20261031 as bridge
from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as gpu

NAME = "p1_hybrid_root1_20261031"
ROOT = runner.ROOT
PROTOCOL = ROOT.parent / "docs/D2P_P1_HYBRID_ROOT1_PROTOCOL_20261031.md"
original_build, original_protocol = runner.build, runner.protocol
original_roster = runner.roster


class Root1Store:
    """Keep winner bridge's historical s0 filenames scoped correctly to root 1."""

    def __init__(self, wrapped):
        self.wrapped = wrapped

    @staticmethod
    def name(value):
        if not value.startswith("s0_"):
            raise ValueError("unexpected winner bridge namespace")
        return "s1_" + value[3:]

    def namespace(self, name):
        return self.wrapped.namespace(self.name(name))

    def put(self, name, value):
        return self.wrapped.put(self.name(name), value)

    def check(self):
        return self.wrapped.check()


def plan_node(task, arm, boundary, store, runtime):
    device = runtime["device"]
    runner.require(device.type == "cuda", "host CUDA required")
    torch.cuda.synchronize(device)
    started = time.monotonic()
    counts = {"grad_attempted": 0, "grad_completed": 0, "backward_attempted": 0}
    original_grad = torch.autograd.grad
    original_backward = torch.autograd.backward

    def measured_grad(*args, **kwargs):
        counts["grad_attempted"] += 1
        result = original_grad(*args, **kwargs)
        counts["grad_completed"] += 1
        return result

    def measured_backward(*args, **kwargs):
        counts["backward_attempted"] += 1
        return original_backward(*args, **kwargs)

    with hooks.ModuleCalls(runtime["model"]) as hook, \
            patch.object(torch.autograd, "grad", measured_grad), \
            patch.object(torch.autograd, "backward", measured_backward):
        result = bridge.plan_node(task, arm, boundary, Root1Store(store), runtime)
        torch.cuda.synchronize(device)
    actual = hooks.validate(hook.counts, result["account"])
    runner.require(counts["grad_attempted"] == counts["grad_completed"]
                   == result["account"]["charged_upper_bounds"]["backward_calls"]
                   and counts["backward_attempted"] == 0, "autograd API/ledger mismatch")
    audit = {"arm": arm, "root_slot": 1, "boundary": boundary,
             "module_forward_pre_hooks": actual,
             "ledger_charged_upper_bounds": result["account"]["charged_upper_bounds"],
             "synchronized_wall_seconds_including_audit": time.monotonic() - started,
             "autograd_api_counts": counts,
             "backward_independently_measured": "python_api_only", "full_cost_certified": False,
             "fairness_proven": False}
    store.put(f"s1_{arm}_b{boundary:02}_hook_audit.json", audit)
    result["module_hook_audit"] = audit
    return result


def protocol():
    value = original_protocol()
    costs = {arm: control.cost(arm) for arm in runner.spec.ARMS}
    value.update(arms=["structured"], initializer="hybrid_goal_spectral_deduplicated",
                 node_costs=costs,
                 node_caps={axis: max(costs[arm][axis] for arm in runner.spec.ARMS)
                            for axis in runner.spec.AXES},
                 extra_policy={arm: ("model_winner_32_updates" if arm in control.EXTRA_ARMS else "none")
                               for arm in runner.spec.ARMS},
                 module_hooks_gate_each_node=True,
                 timing="synchronized wall time including planner and audit",
                 independent_backward_measurement="python_autograd_api_only", full_cost_certified=False)
    return value


def build():
    value = original_build()
    for path in (Path(__file__), Path(hooks.__file__), Path(control.__file__),
                 Path(bridge.__file__), Path(gpu.__file__), Path(gpu.locking.__file__), PROTOCOL):
        path = path.resolve()
        value["sources"][str(path)] = runner.sha(path)
    return value


def configure():
    runner.NAME = NAME
    runner.protocol = protocol
    runner.build = build
    runner.bridge.plan_node = plan_node
    runner.roster = lambda: [row for row in original_roster() if row["arm"] == "structured"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    execute = sub.add_parser("run")
    for key in ("prepared", "sha256", "output", "authorization"):
        execute.add_argument("--" + key, required=True)
    args = parser.parse_args()
    configure()
    if args.command == "prepare":
        result = runner.prepare(args.output)
    else:
        print("Run four-arm real RGB feedback on known root 1 with CUDA; not independent confirmation.", flush=True)
        with gpu.locking.gpu_lock(gpu.LOCK):
            result = runner.run(args.prepared, args.sha256, args.output, args.authorization)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.command == "run" and result["status"] != "recorded_known_root_not_admitted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
