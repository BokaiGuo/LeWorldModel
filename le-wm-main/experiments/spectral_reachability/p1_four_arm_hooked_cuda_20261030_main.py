"""Four-arm RGB feedback with per-node module-call cross-checks."""
import argparse
import json
import time
from pathlib import Path

import torch

from . import codex_four_arm_common_rgb_launcher_20261030 as launcher

NAME = "p1_four_arm_hooked_cuda_20261030_main"
ROOT = launcher.ROOT
PROTOCOL = ROOT.parent / "docs/D2P_FOUR_ARM_MODULE_HOOK_PROTOCOL_20261030_MAIN.md"
original_build = launcher.build
original_protocol = launcher.protocol
original_plan = launcher.bridge.plan_node


class ModuleCalls:
    def __init__(self, model):
        self.counts = {"predict": 0, "action_encode": 0}
        self.handles = []
        self.model = model

    def __enter__(self):
        try:
            for label, attr in (("predict", "predictor"), ("action_encode", "action_encoder")):
                module = getattr(self.model, attr)
                def count(_module, _args, key=label):
                    self.counts[key] += 1
                self.handles.append(module.register_forward_pre_hook(count))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, _type, _value, _traceback):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()


def validate(counts, account):
    declared = account["charged_upper_bounds"]
    actual = {"predict_calls": counts["predict"], "action_encode_calls": counts["action_encode"]}
    if any(actual[k] != declared[k] for k in actual):
        raise ValueError(f"module hook and ledger mismatch: {actual}, {declared}")
    return actual


def plan_node(task, arm, boundary, store, runtime):
    device = runtime["device"]
    launcher.require(device.type == "cuda", "host CUDA required")
    torch.cuda.synchronize(device)
    start = time.monotonic()
    with ModuleCalls(runtime["model"]) as hook:
        result = original_plan(task, arm, boundary, store, runtime)
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - start
    actual = validate(hook.counts, result["account"])
    receipt = {"arm": arm, "boundary": boundary, "module_forward_pre_hooks": actual,
               "ledger_charged_upper_bounds": result["account"]["charged_upper_bounds"],
               "synchronized_wall_seconds_including_audit": elapsed,
               "backward_independently_measured": False, "full_cost_certified": False,
               "fairness_proven": False}
    store.put(f"s0_{arm}_b{boundary:02}_hook_audit.json", receipt)
    result["module_hook_audit"] = receipt
    return result


def protocol():
    value = original_protocol()
    value.update(module_hook_scope=["predictor", "action_encoder"],
                 module_hooks_gate_each_node=True,
                 timing="synchronized wall time including planner and audit",
                 independent_backward_measurement=False, full_cost_certified=False)
    return value


def build():
    result = original_build()
    for path in (Path(__file__).resolve(), PROTOCOL.resolve()):
        result["sources"][str(path)] = launcher.sha(path)
    return result


def configure():
    launcher.NAME = NAME
    launcher.protocol = protocol
    launcher.build = build
    launcher.bridge.plan_node = plan_node


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", required=True)
    r = sub.add_parser("run")
    for key in ("prepared", "sha256", "output", "authorization"):
        r.add_argument("--" + key, required=True)
    args = parser.parse_args()
    configure()
    if args.command == "prepare":
        print(json.dumps(launcher.prepare(args.output)))
    else:
        print("Run four known-root feedback arms with CUDA module hooks; no fairness claim.", flush=True)
        result = launcher.run(args.prepared, args.sha256, args.output, args.authorization)
        print(json.dumps(result, sort_keys=True))
        if result["status"] != "recorded_known_root_not_admitted":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
