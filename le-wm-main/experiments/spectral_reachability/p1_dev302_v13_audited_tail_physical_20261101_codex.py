"""Physical execution of the audited V13 equal-budget replanned tails."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path

from . import p1_dev302_v10_support_tail_physical_20261101_codex as core
from . import p1_dev302_v13_audited_tail_replan_20261101_codex as planner


NAME = "p1_dev302_v13_audited_tail_physical_20261101_codex"
ROOT = planner.ROOT
PLAN_PREP = ROOT / "results/p1_dev302_v13_audited_tail_replan_20261101_codex_PREP_MAIN02.json"
PLAN_PREP_SHA = "15130328b67d85f46d4650843200f6a02aa684ff71af3b691ec38f5603453efb"
PLAN_RUN = ROOT / "results/p1_dev302_v13_audited_tail_replan_20261101_codex_RUN_MAIN02"
PLAN_REPORT_SHA = "bc883118a85a765a277d899123d3a4f66bb3b52e6ee9139e882f92d83bffbd7e"
require, sha = core.require, core.sha
CORE_BUILD = core.build


@contextmanager
def _patched():
    values = {
        "NAME": NAME,
        "planner": planner,
        "PLAN_PREP": PLAN_PREP,
        "PLAN_PREP_SHA": PLAN_PREP_SHA,
        "PLAN_RUN": PLAN_RUN,
        "PLAN_REPORT_SHA": PLAN_REPORT_SHA,
        "ROLES": planner.core.ROLES,
    }
    previous = {key: getattr(core, key) for key in values}
    for key, value in values.items():
        setattr(core, key, value)
    try:
        yield
    finally:
        for key, value in previous.items():
            setattr(core, key, value)


def build():
    with _patched():
        value = CORE_BUILD()
    value["schema"] = NAME + ".contract.v1"
    value["bindings"][str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    value["P1_admission"] = False
    return value


def execute(prepared, prepared_sha, output):
    with _patched():
        old_build = core.build
        core.build = build
        try:
            return core.execute(prepared, prepared_sha, output)
        finally:
            core.build = old_build


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.tag.isalnum(), "tag")
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    require(
        not os.path.lexists(prepared) and (not args.run or not os.path.lexists(output)),
        "exclusive outputs",
    )
    digest = planner.write_new(prepared, build())
    print(json.dumps({"contract": str(prepared), "sha256": digest}), flush=True)
    if not args.run:
        return 0
    with planner.core.engine.base.locking.gpu_lock(
        ROOT / "results/e3_p2_host_cuda_cooperative.lock"
    ):
        report = execute(prepared, digest, output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "report_sha256": sha(output / "report.json"),
                "rows": [
                    {
                        "role": row["role"],
                        "object_position_error": row.get("final_metrics", {}).get(
                            "object_position_error"
                        ),
                        "contacts": [
                            item["step"]
                            for item in row["transitions"]
                            if item["n_contacts"] > 0
                        ],
                    }
                    for row in report["rows"]
                ],
                "errors": report["errors"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return int(report["status"] != "recorded_development_equal_budget_tails")


if __name__ == "__main__":
    raise SystemExit(main())
