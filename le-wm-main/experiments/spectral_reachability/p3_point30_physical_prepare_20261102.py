"""Bind all four fixed P3 point-30 winners to a new physical development plan.

This script only prepares and verifies a plan; it does not execute physics.
The model outcome has already been opened and these roots are not independent.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import p3_preoutcome_twopoint_curve_20261102_codexexclusive as curve
from . import p3_preoutcome_winner_physical_20261102_codexexclusive as prior

ROOT = curve.ROOT
NAME = "p3_point30_physical_20261102"
PREP = ROOT / "results/p3_preoutcome_twopoint_curve_20261102_codexexclusive_PREP_02.json"
RUN = ROOT / "results/p3_preoutcome_twopoint_curve_20261102_codexexclusive_RUN_01"
SAVED = ROOT / "results/p3_preoutcome_twopoint_curve_20261102_codexexclusive_CONSUME_01.json"
PREP_SHA = "d9afb2c9c695b6e4bf70bc866279116c602c6cb8df7e8143e89b736314a52557"
REPORT_SHA = "08a0ff7d065682918c688365f7f40fae163c908dcf42f4e6e26bdae42802025e"
SAVED_SHA = "1b65dcafa104b04b6d1f8a054c35a68746f781b757aaca53fa5e26059636e968"


def material():
    require, moving = prior.require, prior.moving
    require(moving.sha(SAVED) == SAVED_SHA, "curve receipt hash")
    checked = curve.consume(RUN, REPORT_SHA, PREP, PREP_SHA)
    require(checked == moving.read(SAVED) and checked["status"] == "both_saved_consistency_checked",
            "curve receipt reconstruction")
    _, _, bank = curve.verify(PREP, PREP_SHA)
    point = checked["points"][1]
    require(point["point"] == 30 and point["status"] == "saved_consistency_checked", "point30 status")
    source = prior.producer.host.PREVIOUS
    require(moving.sha(source) == prior.producer.old.PREDECESSOR_SHA, "source predecessor")
    slots = prior.producer.host.prior.prior.read(source)["slots"]
    data = moving.load_source(prior.prior.source.SOURCEPACK, prior.prior.source.SOURCEPACK_SHA,
                              prior.prior.source.SOURCE_PREPARE, prior.prior.source.SOURCE_PREPARE_SHA)
    plans, bound = prior.make_plans(point, bank, slots, data["references"],
                                    prior.prior.physical.task_bridge,
                                    prior.producer.host.prior.prior.raw_commands)
    native, native_sha = prior.checks.native_constructor_contract()
    bindings = dict(data["bindings"])
    bindings.update(bound)
    for path in (PREP, RUN / "report.json", RUN / "curve_prepare.json",
                 RUN / "controller.jsonl", RUN / "events.jsonl", SAVED,
                 Path(__file__), Path(__file__).with_name("p3_point30_physical_run_20261102.py"),
                 Path(prior.__file__), Path(moving.__file__),
                 Path(moving.low.__file__), Path(moving.moving.__file__), Path(native)):
        bindings[str(path.resolve())] = moving.sha(path)
    require(bindings[str(Path(native).resolve())] == native_sha, "native constructor drift")
    contract = moving.contract(600)
    contract.update(scope="four saved point30 winners, two exposed roots",
                    no_gpu=True, physical_replans=0,
                    native_stop_policy="stop on termination or truncation; preserve censored endpoint",
                    failure_policy="retain attempted counts and failures")
    return {"schema": NAME + ".prepare.v1", "plans": plans, "contract": contract,
            "sourcepack_sha256": prior.prior.source.SOURCEPACK_SHA,
            "bindings": bindings, "model_outcomes_already_open": True,
            "physical_execution": False, "independent_outcome": False,
            "equal_total_compute": False, "P3_admission": False, "C3_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    prior.require(path.is_absolute() and path.parent == ROOT / "results"
                  and path.name.startswith(NAME + "_PREP_") and not os.path.lexists(path),
                  "exclusive output")
    plan = material()
    sha = prior.producer.host.write_new(path, plan)
    print(json.dumps({"path": str(path), "sha256": sha, "arms": len(plan["plans"]),
                      "physical_execution": False}))


if __name__ == "__main__":
    main()
