"""Opt-in physical execution of the sealed dev302 V4 four-prefix selection."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path

from . import p1_dev302_mixed_bank_cuda_v4_20261101_codex as planner
from . import p1_dev302_mixed_prefix_physical_v3_20261101_codex as engine

NAME = "p1_dev302_mixed_prefix_physical_v4_20261101_codex"
ROOT = planner.ROOT
PLAN_PREP = ROOT / "results/p1_dev302_mixed_bank_cuda_v4_20261101_codex_PREP_MAIN01.json"
PLAN_PREP_SHA = "ac0fd2082a1ab12582df58da35d847a0396355e3a5a4e6d391b150c1e8c268d1"
PLAN_RUN = ROOT / "results/p1_dev302_mixed_bank_cuda_v4_20261101_codex_RUN_MAIN01"
PLAN_REPORT_SHA = "73a8cab5ac90aef32df33bbb1b02676fd373592cd1c8706a75d168a95c0a0db3"
V4_BANK_SCHEMA = "p1_dev302_mixed_prefix_bank_v4_20261101_codex.v1"
V4_GUARD_KIND = "rgb_calibrated_action_path_with_routed_terminal_overtravel"

source = engine.source
base = engine.base
require, sha = engine.require, engine.sha
ENGINE_BUILD = engine.build


def _planner_material():
    require(
        sha(PLAN_PREP) == PLAN_PREP_SHA
        and sha(PLAN_RUN / "report.json") == PLAN_REPORT_SHA,
        "sealed V4 planner input SHA",
    )
    protocol = base.read(PLAN_PREP)
    report = base.read(PLAN_RUN / "report.json")
    require(protocol == planner.protocol(), "V4 protocol reconstruction")
    require(
        protocol.get("schema") == planner.NAME + ".protocol.v1"
        and report.get("schema") == planner.NAME + ".report.v1"
        and report.get("status") == "model_only_complete"
        and report.get("protocol_sha256") == PLAN_PREP_SHA
        and protocol.get("physical_outcomes_used") is False
        and report.get("physical_outcomes_used") is False,
        "V4 planner contract",
    )
    return protocol, report


def _validate_rgb_only(protocol, report):
    bank = protocol.get("prefix_bank")
    guard = bank.get("contact_guard") if isinstance(bank, dict) else None
    require(
        isinstance(bank, dict)
        and bank.get("schema") == V4_BANK_SCHEMA
        and bank.get("candidate_order_fixed") is True
        and bank.get("simulator_state_used") is False
        and bank.get("physical_outcome_used") is False
        and isinstance(guard, dict)
        and guard.get("kind") == V4_GUARD_KIND
        and guard.get("simulator_state_used") is False
        and guard.get("route_terminal_overtravel_pixels") == 3.0
        and guard.get("contact_center_to_object_pixels") == 5.5,
        "RGB-only V4 prefix source",
    )
    require(
        protocol["algorithm"].get("contact_filter") == guard
        and report.get("physical_outcomes_used") is False,
        "RGB-only V4 selection binding",
    )
    current = source.source_directory() / "rgb_10.png"
    goal = source.source_directory() / "goal.png"
    bindings = protocol.get("bindings", {})
    require(
        bindings.get(str(current.resolve())) == sha(current)
        and bindings.get(str(goal.resolve())) == sha(goal),
        "RGB source bindings",
    )


@contextmanager
def _patched_engine():
    names = {
        "NAME": NAME,
        "planner": planner,
        "PLAN_PREP": PLAN_PREP,
        "PLAN_PREP_SHA": PLAN_PREP_SHA,
        "PLAN_RUN": PLAN_RUN,
        "PLAN_REPORT_SHA": PLAN_REPORT_SHA,
        "_planner_material": _planner_material,
        "_validate_rgb_only": _validate_rgb_only,
    }
    previous = {key: getattr(engine, key) for key in names}
    for key, value in names.items():
        setattr(engine, key, value)
    try:
        yield
    finally:
        for key, value in previous.items():
            setattr(engine, key, value)


def build():
    with _patched_engine():
        value = ENGINE_BUILD()
    value["schema"] = NAME + ".contract.v1"
    value["planner"]["name"] = planner.NAME
    value["planner"]["prep"] = str(PLAN_PREP.resolve())
    value["planner"]["prep_sha256"] = PLAN_PREP_SHA
    value["planner"]["report"] = str((PLAN_RUN / "report.json").resolve())
    value["planner"]["report_sha256"] = PLAN_REPORT_SHA
    value["prefix_source"]["contact_calibration"] = (
        "development RGB trajectories only; routed endpoint margin fixed "
        "after the sealed V3 false-positive diagnosis"
    )
    return value


def execute(prepared: Path, prepared_sha: str, output: Path):
    original_build = engine.build
    with _patched_engine():
        engine.build = build
        try:
            return engine.execute(prepared, prepared_sha, output)
        finally:
            engine.build = original_build


def main():
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
    digest = planner.runner.write_new(prepared, build())
    print(json.dumps({"contract": str(prepared), "sha256": digest}), flush=True)
    if not args.run:
        return 0
    with planner.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
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
                        "label": row["candidate_label"],
                        "status": row["status"],
                        "native_success": row.get("final_metrics", {}).get(
                            "native_success"
                        ),
                        "object_position_error": row.get("final_metrics", {}).get(
                            "object_position_error"
                        ),
                        "contact_steps": [
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
    return int(report["status"] != "recorded_development_v3_prefixes")


if __name__ == "__main__":
    raise SystemExit(main())
