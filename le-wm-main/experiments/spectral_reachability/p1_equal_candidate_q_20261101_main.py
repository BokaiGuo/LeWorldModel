"""Fixed equal-per-candidate Q suffix allocation, no winner-only refinement."""
import argparse
import json
from pathlib import Path

from . import p1_q_return_shard_20261101_main as parent

NAME = "p1_equal_candidate_q_20261101_main"
ROOT = parent.ROOT
DOC = ROOT.parent / "docs/D2P_P1_EQUAL_CANDIDATE_Q_PROTOCOL_20261101_MAIN.md"


def allocation():
    control = parent.carry.control
    control.EXTRA_ARMS = ()
    control.base.CONFIG = dict(control.base.CONFIG, steps=6)


def setup(arm):
    experiment = parent.setup(arm)
    allocation()
    runner = experiment.runner
    old_build, old_protocol = runner.build, runner.protocol

    def build():
        value = old_build()
        for path in (Path(__file__).resolve(), DOC):
            value["sources"][str(path)] = runner.sha(path)
        return value

    def protocol():
        value = old_protocol()
        value.update(candidate_optimizer_steps=6, candidate_count=8,
                     winner_only_rounds=0, total_optimizer_steps=48,
                     immediate_progress_budget_increased=True,
                     equal_total_compute_claim=False,
                     allocation_source_sha256=runner.sha(Path(__file__).resolve()))
        return value

    runner.build, runner.protocol = build, protocol
    runner.NAME = f"{NAME}_{arm}"
    return experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=parent.parent.ARMS, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--authorize-host-cuda", action="store_true", required=True)
    args = parser.parse_args()
    if not args.tag.isalnum():
        raise ValueError("simple tag")
    experiment = setup(args.arm)
    prefix = ROOT / "results" / f"{NAME}_{args.arm}"
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
