"""Audited equal-budget cp5 tail replanning from the V11 physical prefixes."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path

from . import p1_dev302_mixed_bank_cuda_v12_20261101_codex as planner
from . import p1_dev302_v10_support_tail_replan_20261101_codex as core
from . import p1_dev302_v11_audited_prefix_physical_20261101_codex as physical


NAME = "p1_dev302_v13_audited_tail_replan_20261101_codex"
ROOT = core.ROOT
PLANNER_PREP = ROOT / "results/p1_dev302_mixed_bank_cuda_v12_20261101_codex_PREP_MAIN01.json"
PLANNER_PREP_SHA = "d2e5376480af4b75857bd6e9eb29ff24b514da4f9bb7827e90e4bba59a5a6f28"
PLANNER_RUN = ROOT / "results/p1_dev302_mixed_bank_cuda_v12_20261101_codex_RUN_MAIN01"
PLANNER_REPORT_SHA = "f7e66f1d045ff34f1468c02c480a296baf3c9b7f41dd4844f7d759f445f737ca"
PHYSICAL_PREP = ROOT / "results/p1_dev302_v11_audited_prefix_physical_20261101_codex_PREP_MAIN02.json"
PHYSICAL_PREP_SHA = "b32c9ed80fdca83ed9b1936457cd9e4ac203f8fef7e2090dea077503cd64989c"
PHYSICAL_RUN = ROOT / "results/p1_dev302_v11_audited_prefix_physical_20261101_codex_RUN_MAIN02"
PHYSICAL_REPORT_SHA = "3a682ba341db272bbcd2587db9c1a3170eca0083081c9c680314ca09609b2c43"
require, write_new = core.require, core.write_new


@contextmanager
def _patched():
    values = {
        "NAME": NAME,
        "prefix_planner": planner,
        "physical": physical,
        "PLANNER_PREP": PLANNER_PREP,
        "PLANNER_PREP_SHA": PLANNER_PREP_SHA,
        "PLANNER_RUN": PLANNER_RUN,
        "PLANNER_REPORT_SHA": PLANNER_REPORT_SHA,
        "PHYSICAL_PREP": PHYSICAL_PREP,
        "PHYSICAL_PREP_SHA": PHYSICAL_PREP_SHA,
        "PHYSICAL_RUN": PHYSICAL_RUN,
        "PHYSICAL_REPORT_SHA": PHYSICAL_REPORT_SHA,
    }
    previous = {key: getattr(core, key) for key in values}
    for key, value in values.items():
        setattr(core, key, value)
    try:
        yield
    finally:
        for key, value in previous.items():
            setattr(core, key, value)


def protocol():
    with _patched():
        value = core.protocol()
    value["schema"] = NAME + ".protocol.v1"
    value["bindings"][str(Path(__file__).resolve())] = core.sha256(
        Path(__file__).resolve()
    )
    value["P1_admission"] = False
    return value


def run(prepared, prepared_sha, output):
    with _patched():
        with core._patched():
            old_protocol = core.engine.protocol
            core.engine.protocol = protocol
            try:
                return core.engine.run(prepared, prepared_sha, output)
            finally:
                core.engine.protocol = old_protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.tag.isalnum(), "tag")
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    require(
        not os.path.lexists(prepared) and not os.path.lexists(output),
        "exclusive outputs",
    )
    digest = write_new(prepared, protocol())
    print(json.dumps({"prepared": str(prepared), "sha256": digest}), flush=True)
    if args.run:
        with core.engine.base.locking.gpu_lock(
            ROOT / "results/e3_p2_host_cuda_cooperative.lock"
        ):
            report_sha = run(prepared, digest, output)
        print(
            json.dumps({"output": str(output), "report_sha256": report_sha}),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
