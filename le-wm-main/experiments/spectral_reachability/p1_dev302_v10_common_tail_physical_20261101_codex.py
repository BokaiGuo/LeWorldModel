"""Apply the V10 selected tail to every V9 prefix for mechanism isolation."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import p1_dev302_v10_support_tail_physical_20261101_codex as source


NAME = "p1_dev302_v10_common_tail_physical_20261101_codex"
ROOT = source.ROOT
SOURCE_PREP = ROOT / "results/p1_dev302_v10_support_tail_physical_20261101_codex_PREP_MAIN01.json"
SOURCE_PREP_SHA = "630c3e5d8670db18a5226cdd18162a7522dd45e9d873ee542e22924c0e698313"
SOURCE_PLAN = ROOT / "results/p1_dev302_v10_support_tail_replan_20261101_codex_RUN_MAIN01/report.json"
SOURCE_PLAN_SHA = "18bbad2491b4072fd5e80fc74f98b777aabdcfe46918522f3a173d46258b4c2a"
engine = source.engine
require, sha = source.require, source.sha


def _action_sha(actions):
    return hashlib.sha256(np.asarray(actions, dtype="<f4").tobytes()).hexdigest()


def build():
    require(
        sha(SOURCE_PREP) == SOURCE_PREP_SHA and sha(SOURCE_PLAN) == SOURCE_PLAN_SHA,
        "sealed V10 sources",
    )
    source_contract = json.loads(SOURCE_PREP.read_bytes())
    plan_report = json.loads(SOURCE_PLAN.read_bytes())
    require(
        source_contract == source.build()
        and plan_report["status"] == "model_only_complete_not_admitted"
        and plan_report["post_cp5_physical_outcomes_used"] is False
        and plan_report["P1_admission"] is False,
        "V10 source contracts",
    )
    selected = next(row for row in plan_report["rows"] if row["role"] == "selected")
    common_tail = np.asarray(selected["selected"]["actions"], np.float32).reshape(10, 2)
    rows = []
    for row in source_contract["roles"]:
        prefix = np.asarray(row["actions"][:5], np.float32)
        actions = np.concatenate((prefix, common_tail), axis=0).astype(np.float32)
        rows.append(
            {
                **row,
                "actions": actions.tolist(),
                "actions_sha256": _action_sha(actions),
                "prefix_sha256": _action_sha(prefix),
                "selected_tail": {
                    **selected["selected"],
                    "mechanism_role": "same selected tail for every prefix",
                },
            }
        )
    bindings = {
        str(path.resolve()): sha(path.resolve())
        for path in (Path(__file__), Path(source.__file__), SOURCE_PREP, SOURCE_PLAN)
    }
    for path, digest in source_contract["bindings"].items():
        require(path not in bindings or bindings[path] == digest, "binding collision")
        bindings[path] = digest
    return {
        **source_contract,
        "schema": NAME + ".contract.v1",
        "roles": rows,
        "common_tail_role": "selected",
        "common_tail_sha256": _action_sha(common_tail),
        "same_replanning_budget": False,
        "mechanism_intervention": True,
        "performance_comparison": False,
        "P1_admission": False,
        "bindings": bindings,
    }


@contextmanager
def _patched():
    old_name, old_build = engine.NAME, engine.build
    engine.NAME, engine.build = NAME, build
    try:
        yield
    finally:
        engine.NAME, engine.build = old_name, old_build


def execute(prepared, prepared_sha, output):
    with _patched():
        return engine.execute(prepared, prepared_sha, output)


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
    digest = source.planner.write_new(prepared, build())
    print(json.dumps({"contract": str(prepared), "sha256": digest}), flush=True)
    if not args.run:
        return 0
    with source.planner.engine.base.locking.gpu_lock(
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
