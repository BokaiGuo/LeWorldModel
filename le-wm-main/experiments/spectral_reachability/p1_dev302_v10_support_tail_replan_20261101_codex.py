"""Equal-budget cp5 tail replanning after the V9 task-support prefixes."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path

import numpy as np

from . import p1_dev302_mixed_bank_cuda_v8_20261101_codex as prefix_planner
from . import p1_dev302_v5_contact_tail_replan_20261101_codex as engine
from . import p1_dev302_v9_task_greedy_prefix_physical_20261101_codex as physical


NAME = "p1_dev302_v10_support_tail_replan_20261101_codex"
ROOT = engine.ROOT
PLANNER_PREP = ROOT / "results/p1_dev302_mixed_bank_cuda_v8_20261101_codex_PREP_MAIN01.json"
PLANNER_PREP_SHA = "40d6f517ced03a790c21e2ec4d4a8ecfeb69269937df0e56de203e180c36a394"
PLANNER_RUN = ROOT / "results/p1_dev302_mixed_bank_cuda_v8_20261101_codex_RUN_MAIN01"
PLANNER_REPORT_SHA = "547d657e5155df3a6cca88969045b04d9bd1b365135f54b95958311119d7b18c"
PHYSICAL_PREP = (
    ROOT / "results/p1_dev302_v9_task_greedy_prefix_physical_20261101_codex_PREP_MAIN02.json"
)
PHYSICAL_PREP_SHA = "2a7bd99148abe8a72ccf320dae5ebde1ff4db63ba60267d86de77544c312a0ad"
PHYSICAL_RUN = (
    ROOT / "results/p1_dev302_v9_task_greedy_prefix_physical_20261101_codex_RUN_MAIN02"
)
PHYSICAL_REPORT_SHA = "aa118b50fa15b983c9c6f937af0f4d8f1bb68761b850ce579183eb0a98671e73"
ROLES = physical.ROLES
ROLE_INDICES = {
    "selected": 16,
    "zero": 0,
    "same_bank_greedy": 4,
    "random": 7,
}
base = engine.base
require, sha256, write_new = engine.require, engine.sha256, engine.write_new
ENGINE_PROTOCOL = engine.protocol


def _source_protocol() -> dict:
    require(
        sha256(PLANNER_PREP) == PLANNER_PREP_SHA
        and sha256(PLANNER_RUN / "report.json") == PLANNER_REPORT_SHA
        and sha256(PHYSICAL_PREP) == PHYSICAL_PREP_SHA
        and sha256(PHYSICAL_RUN / "report.json") == PHYSICAL_REPORT_SHA,
        "sealed V8/V9 source SHA",
    )
    planner_protocol = json.loads(PLANNER_PREP.read_bytes())
    planner_report = json.loads((PLANNER_RUN / "report.json").read_bytes())
    physical_contract = json.loads(PHYSICAL_PREP.read_bytes())
    physical_report = json.loads((PHYSICAL_RUN / "report.json").read_bytes())
    require(
        planner_protocol == prefix_planner.protocol()
        and planner_report["schema"] == prefix_planner.NAME + ".report.v1"
        and planner_report["status"] == "model_only_complete"
        and planner_report["protocol_sha256"] == PLANNER_PREP_SHA
        and physical_contract == physical.build()
        and physical_report["schema"] == physical.NAME + ".report.v1"
        and physical_report["status"] == "recorded_development_v3_prefixes"
        and physical_report["prepare_sha256"] == PHYSICAL_PREP_SHA
        and physical_report["errors"] == []
        and physical_report["post_prefix_outcomes_used_for_planning"] is False
        and physical_report["P1_admission"] is False,
        "V8/V9 source contracts",
    )
    base.read_physical_journal(PHYSICAL_RUN, physical_report)
    planner_rows = {row["index"]: row for row in planner_report["rows"]}
    physical_rows = {row["role"]: row for row in physical_report["rows"]}
    require(
        set(physical_rows) == set(ROLES)
        and set(ROLE_INDICES.values()).issubset(planner_rows),
        "four-role source roster",
    )
    canonical_rows = [planner_rows[ROLE_INDICES[role]] for role in ROLES]
    seed_tails = base._unique_tails(canonical_rows)
    bindings = {
        str(path.resolve()): sha256(path.resolve())
        for path in (
            Path(__file__),
            Path(prefix_planner.__file__),
            Path(physical.__file__),
            PLANNER_PREP,
            PLANNER_RUN / "report.json",
            PHYSICAL_PREP,
            PHYSICAL_RUN / "report.json",
            PHYSICAL_RUN / "journal.jsonl",
        )
    }
    roles = []
    for role in ROLES:
        expected = planner_rows[ROLE_INDICES[role]]
        observed = physical_rows[role]
        actual = np.asarray(
            [transition["action"] for transition in observed["transitions"]],
            np.float32,
        )
        planned = np.asarray(expected["actions"], np.float32)
        require(
            observed["candidate_index"] == expected["index"] == ROLE_INDICES[role]
            and observed["candidate_label"] == expected["label"]
            and observed["status"] == "prefix_completed"
            and observed["error"] is None
            and observed["plan_returned"] == 5
            and actual.shape == planned.shape == (5, 2)
            and np.array_equal(actual, planned)
            and observed["planned_actions_sha256"] == base.action_sha(planned),
            f"V9 physical identity: {role}",
        )
        cp5, cp5_path = base._read_cp5(PHYSICAL_RUN, observed)
        bindings[str(cp5_path)] = cp5["file_sha256"]
        old_tail = np.asarray(expected["tail_selected"]["actions"], np.float32)
        roles.append(
            {
                "role": role,
                "candidate_index": expected["index"],
                "candidate_label": expected["label"],
                "actual_prefix": actual.tolist(),
                "actual_prefix_sha256": base.action_sha(actual),
                "cp5_png": str(cp5_path),
                "cp5_file_sha256": cp5["file_sha256"],
                "cp5_pixel_sha256": cp5["pixel_sha256"],
                "old_v4_tail": old_tail.tolist(),
                "old_v4_tail_sha256": base.action_sha(old_tail),
            }
        )
    for path, expected in planner_protocol["bindings"].items():
        require(path not in bindings or bindings[path] == expected, "binding collision")
        bindings[path] = expected
    require(
        all(sha256(Path(path)) == expected for path, expected in bindings.items()),
        "source closure",
    )
    return {
        "schema": NAME + ".source.v1",
        "physical_directory": str(PHYSICAL_RUN.resolve()),
        "physical_report_sha256": PHYSICAL_REPORT_SHA,
        "roles": roles,
        "task": planner_protocol["task"],
        "policy": planner_protocol["policy"],
        "model_sha256": base.MODEL_SHA256,
        "seed_tails": [tail.tolist() for tail in seed_tails],
        "algorithm": {
            "source": "sealed V8 candidate bank and V9 physical cp5 observations",
        },
        "information_boundary": (
            "per-arm actual V9 cp5 RGB and executed five-action prefix; "
            "no post-cp5 physical outcome"
        ),
        "equal_cem_and_robust_budget": True,
        "post_cp5_physical_outcomes_used": False,
        "physical_execution": False,
        "training": False,
        "development_only": True,
        "independent_confirmation": False,
        "P1_admission": False,
        "bindings": bindings,
    }


@contextmanager
def _patched():
    old_name, old_source_protocol = engine.NAME, engine.base.protocol
    engine.NAME = NAME
    engine.base.protocol = lambda _directory, _report_sha: _source_protocol()
    try:
        yield
    finally:
        engine.base.protocol = old_source_protocol
        engine.NAME = old_name


def protocol() -> dict:
    with _patched():
        value = ENGINE_PROTOCOL()
    value["schema"] = NAME + ".protocol.v1"
    value["algorithm"]["task_support_constraint"] = (
        "actual cp5 nearest visible object boundary; V9 selected support face "
        "is aligned with the RGB goal-object displacement"
    )
    value["bindings"][str(Path(__file__).resolve())] = sha256(Path(__file__).resolve())
    value["P1_admission"] = False
    return value


def run(prepared: Path, prepared_sha: str, output: Path) -> str:
    old_protocol = engine.protocol
    with _patched():
        engine.protocol = protocol
        try:
            return engine.run(prepared, prepared_sha, output)
        finally:
            engine.protocol = old_protocol


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
        with engine.base.locking.gpu_lock(
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
