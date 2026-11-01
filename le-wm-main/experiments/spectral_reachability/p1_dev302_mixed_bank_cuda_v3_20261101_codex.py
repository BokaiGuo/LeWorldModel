"""V3 mixed-bank CUDA selection with RGB-calibrated contact-path filtering."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as locking
from . import p1_dev302_mixed_bank_cuda_20261101_codex as runner
from . import p1_dev302_mixed_bank_cuda_v2_20261101_codex as v2
from . import p1_dev302_mixed_prefix_bank_v3_20261101_codex as bank_v3

NAME = "p1_dev302_mixed_bank_cuda_v3_20261101_codex"
ROOT = runner.ROOT
BASE_PROTOCOL = v2.protocol


def protocol():
    value = BASE_PROTOCOL()
    _current_path, _goal_path, current, goal = runner._images()
    plans = runner._plans()
    value["schema"] = NAME + ".protocol.v1"
    value["prefix_bank"] = bank_v3.build(
        current, goal,
        plans["full10_conditional_D2P"][:5],
        plans["line_search_GN"][:5],
    )
    value["algorithm"]["contact_filter"] = value["prefix_bank"]["contact_guard"]
    for path in (Path(__file__).resolve(), Path(bank_v3.__file__).resolve(), Path(v2.__file__).resolve()):
        value["bindings"][str(path)] = runner.sha(path)
    return value


def run(prepared: Path, prepared_sha: str, output: Path) -> str:
    old_name, old_protocol = runner.NAME, runner.protocol
    runner.NAME, runner.protocol = NAME, protocol
    try:
        return runner.run(prepared, prepared_sha, output)
    finally:
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
            report_sha = run(prepared, digest, output)
        print(json.dumps({"output": str(output), "report_sha256": report_sha}), flush=True)


if __name__ == "__main__":
    main()
