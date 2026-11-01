"""Isolate fixed cross-goal arms so one physical failure cannot hide others."""
import argparse
import json
from pathlib import Path

from . import p1_cross_goal_carry_20261101_main as cross

NAME = "p1_cross_goal_shard_20261101_main"
DOC = Path(__file__).resolve().parents[3] / "docs/D2P_P1_CROSS_GOAL_SHARDS_20261101_MAIN.md"
ARMS = ("structured", "native", "random", "immediate_progress")


def setup(root, arm):
    if arm not in ARMS:
        raise ValueError("fixed arm roster")
    experiment = cross.setup(root)
    runner = experiment.runner
    owner = runner.rgb if root == 0 else runner
    old_roster, old_build, old_protocol = owner.roster, runner.build, runner.protocol
    owner.roster = lambda: [row for row in old_roster() if row["arm"] == arm]

    def build():
        value = old_build()
        for path in (Path(__file__).resolve(), DOC):
            value["sources"][str(path)] = runner.sha(path)
        return value

    def protocol():
        value = old_protocol()
        value.update(arms=[arm], panel_roster=list(ARMS), shard=True,
                     scientific_panel_denominator=8, ignore_other_arm_failure=False)
        return value

    runner.build, runner.protocol = build, protocol
    runner.NAME = f"{NAME}_root{root}_{arm}"
    return experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=int, choices=(0, 1), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--authorize-host-cuda", action="store_true", required=True)
    args = parser.parse_args()
    if not args.tag.isalnum():
        raise ValueError("simple tag")
    exp = setup(args.root, args.arm)
    prefix = exp.runner.ROOT / "results" / f"{NAME}_root{args.root}_{args.arm}"
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
