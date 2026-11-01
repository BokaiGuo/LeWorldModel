"""V12 CUDA selector using the bounded V9 task-support bank."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import p1_dev302_mixed_bank_cuda_v8_20261101_codex as v8_runner
from . import p1_dev302_mixed_prefix_bank_v9_20261101_codex as bank


runner, locking = v8_runner.runner, v8_runner.locking
NAME = "p1_dev302_mixed_bank_cuda_v12_20261101_codex"
ROOT = runner.ROOT


def protocol():
    value = v8_runner.v7r.v6_runner.v5_runner.BASE_PROTOCOL()
    _audit, _report, current, goal = runner._images()
    plans = runner._plans()
    value["schema"] = NAME + ".protocol.v1"
    value["prefix_bank"] = bank.build(
        current,
        goal,
        plans["full10_conditional_D2P"][:5],
        plans["line_search_GN"][:5],
    )
    value["algorithm"]["contact_filter"] = value["prefix_bank"]["contact_guard"]
    value["algorithm"]["tail_randomization"] = {
        "kind": "common_random_numbers",
        "seed": runner.SEED,
    }
    count = len(value["prefix_bank"]["candidates"])
    value["algorithm"]["prefix_candidates"] = count
    value["budget"].update(
        action_encoder_calls=count * (1 + runner.ITERATIONS * runner.HORIZON),
        predict_calls=count * (1 + runner.ITERATIONS * runner.HORIZON),
        predict_samples=count
        * (1 + runner.ITERATIONS * runner.HORIZON * runner.SAMPLES * runner.VARIANTS),
        candidate_evaluations=count,
    )
    for path in (Path(__file__).resolve(), Path(bank.__file__).resolve()):
        value["bindings"][str(path)] = runner.sha(path)
    return value


def run(prepared, digest, output):
    old_name, old_protocol = runner.NAME, runner.protocol
    original = runner.np.random.default_rng
    runner.NAME, runner.protocol = NAME, protocol
    runner.np.random.default_rng = lambda seed=None: (
        original(seed)
        if seed == bank.v8.v7.v6.geometry.RANDOM_SEED
        else original(runner.SEED)
    )
    try:
        return runner.run(prepared, digest, output)
    finally:
        runner.np.random.default_rng = original
        runner.NAME, runner.protocol = old_name, old_protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    runner.require(
        args.tag.isalnum()
        and not os.path.lexists(prepared)
        and not os.path.lexists(output),
        "exclusive",
    )
    digest = runner.write_new(prepared, protocol())
    print(json.dumps({"protocol": str(prepared), "sha256": digest}), flush=True)
    if args.run:
        with locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            report_sha = run(prepared, digest, output)
        print(
            json.dumps({"output": str(output), "report_sha256": report_sha}),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
