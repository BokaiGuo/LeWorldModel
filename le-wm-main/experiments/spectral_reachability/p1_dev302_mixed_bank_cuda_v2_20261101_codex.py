"""V2 host-CUDA mixed-bank selection with componentwise contact feasibility."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as locking
from . import p1_dev302_mixed_bank_cuda_20261101_codex as v1
from . import p1_dev302_mixed_prefix_bank_v2_20261101_codex as bank_v2

NAME = "p1_dev302_mixed_bank_cuda_v2_20261101_codex"
ROOT = v1.ROOT
BASE_PROTOCOL = v1.protocol


def protocol():
    value = BASE_PROTOCOL()
    current_path, goal_path, current, goal = v1._images()
    plans = v1._plans()
    value["schema"] = NAME + ".protocol.v1"
    value["prefix_bank"] = bank_v2.build(
        current, goal,
        plans["full10_conditional_D2P"][:5],
        plans["line_search_GN"][:5],
    )
    value["algorithm"]["contact_filter"] = value["prefix_bank"]["contact_guard"]
    value["bindings"][str(Path(__file__).resolve())] = v1.sha(Path(__file__).resolve())
    value["bindings"][str(Path(bank_v2.__file__).resolve())] = v1.sha(
        Path(bank_v2.__file__).resolve()
    )
    # Retain the V1 implementation binding because its audited CUDA runner is reused.
    value["bindings"][str(Path(v1.__file__).resolve())] = v1.sha(Path(v1.__file__).resolve())
    assert current_path.is_file() and goal_path.is_file()
    return value


def run(prepared: Path, prepared_sha: str, output: Path) -> str:
    old_name, old_protocol = v1.NAME, v1.protocol
    v1.NAME, v1.protocol = NAME, protocol
    try:
        return v1.run(prepared, prepared_sha, output)
    finally:
        v1.NAME, v1.protocol = old_name, old_protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    v1.require(args.tag.isalnum(), "tag")
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    v1.require(not os.path.lexists(prepared) and not os.path.lexists(output), "exclusive")
    digest = v1.write_new(prepared, protocol())
    print(json.dumps({"protocol": str(prepared), "sha256": digest}), flush=True)
    if args.run:
        with locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            report_sha = run(prepared, digest, output)
        print(json.dumps({"output": str(output), "report_sha256": report_sha}), flush=True)


if __name__ == "__main__":
    main()
