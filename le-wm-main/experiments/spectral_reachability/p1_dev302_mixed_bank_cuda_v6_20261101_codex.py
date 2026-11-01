"""V6 mixed-bank CUDA selection with task-support feasibility."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import p1_dev302_mixed_bank_cuda_v5_20261101_codex as v5_runner
from . import p1_dev302_mixed_prefix_bank_v6_20261101_codex as bank_v6

runner = v5_runner.runner
locking = v5_runner.locking
NAME = "p1_dev302_mixed_bank_cuda_v6_20261101_codex"
ROOT = runner.ROOT


def protocol():
    value = v5_runner.BASE_PROTOCOL()
    _current_path, _goal_path, current, goal = runner._images()
    plans = runner._plans()
    value["schema"] = NAME + ".protocol.v1"
    value["prefix_bank"] = bank_v6.build(
        current,
        goal,
        plans["full10_conditional_D2P"][:5],
        plans["line_search_GN"][:5],
    )
    value["algorithm"]["contact_filter"] = value["prefix_bank"]["contact_guard"]
    value["algorithm"]["tail_randomization"] = {
        "kind": "common_random_numbers",
        "seed": runner.SEED,
        "same_sample_stream_for_every_prefix": True,
    }
    value["algorithm"]["prefix_candidates"] = len(value["prefix_bank"]["candidates"])
    n = len(value["prefix_bank"]["candidates"])
    value["budget"].update(
        action_encoder_calls=n * (1 + runner.ITERATIONS * runner.HORIZON),
        predict_calls=n * (1 + runner.ITERATIONS * runner.HORIZON),
        predict_samples=n * (
            1 + runner.ITERATIONS * runner.HORIZON * runner.SAMPLES * runner.VARIANTS
        ),
        candidate_evaluations=n,
    )
    for path in (Path(__file__).resolve(), Path(bank_v6.__file__).resolve()):
        value["bindings"][str(path)] = runner.sha(path)
    return value


def run(prepared, prepared_sha, output):
    old_name, old_protocol = runner.NAME, runner.protocol
    original_default_rng = runner.np.random.default_rng

    def common_rng(requested_seed=None):
        if requested_seed == bank_v6.geometry.RANDOM_SEED:
            return original_default_rng(requested_seed)
        return original_default_rng(runner.SEED)

    runner.NAME, runner.protocol = NAME, protocol
    runner.np.random.default_rng = common_rng
    try:
        return runner.run(prepared, prepared_sha, output)
    finally:
        runner.np.random.default_rng = original_default_rng
        runner.NAME, runner.protocol = old_name, old_protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    runner.require(args.tag.isalnum(), "tag")
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    runner.require(not os.path.lexists(prepared) and not os.path.lexists(output), "exclusive")
    digest = runner.write_new(prepared, protocol())
    print(json.dumps({"protocol": str(prepared), "sha256": digest}), flush=True)
    if args.run:
        with locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            result_sha = run(prepared, digest, output)
        print(json.dumps({"output": str(output), "report_sha256": result_sha}), flush=True)


if __name__ == "__main__":
    main()
