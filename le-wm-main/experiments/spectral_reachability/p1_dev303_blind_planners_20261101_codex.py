"""Development-only planner adapter for the explicitly seeded dev303 export."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

from . import p1_constructive_blind_planners_cuda_20261027_v1 as base

NAME = "p1_dev303_blind_planners_20261101_codex"
SEED = 301000003
TASK_SHA = "3787aba5dfe49c97b9c738e347e228c8089449af0df84a5f91788ceb769b2b12"
BLIND_SHA = "092af72253c9b0463e5c7a5814052599843dc4b6a979bd040d7f318c93dcddd9"
MANIFEST_SHA = "b7b5a1b2394acdad13981c1fb4258d97941e0f1863e816e9dc86d85b91b460dc"
ENCODE_PREP_SHA = "80bc35aec5ea95d43c245b5ffccadae0b2e65cd6c0fb85abdb093775c26f0ccb"
_validate = base.validate_export_envelope
_build = base.build


def validate_export(export):
    base.require(type(export) is dict and export.get("development_seed") == SEED
                 and type(export.get("development_seed")) is int, "exact development seed")
    remaining = dict(export)
    del remaining["development_seed"]
    _validate(remaining)
    contract = export["task_contract"]
    base.require(export["blind_source_sha256"] == BLIND_SHA
                 and contract["whitelist_sha"] == MANIFEST_SHA
                 and contract["prepare_sha256"] == ENCODE_PREP_SHA
                 and export["global_freshness"] == "unknown"
                 and export["source_admission"] == "development receipt chain verified; no independent admission",
                 "dev303 source identity and evidence boundary")


def build(task_exports, task_shas):
    base.require(len(task_exports) == 1 and task_shas == [TASK_SHA], "one approved dev303 task")
    result = _build(task_exports, task_shas)
    result["schema"] = NAME + ".prepare"
    result["bindings_sha256"][str(Path(__file__).resolve())] = base.sha(Path(__file__))
    result["classification"] = "development_only_not_independent"
    return result


@contextmanager
def patched():
    old_validate, old_build = base.validate_export_envelope, base.build
    base.validate_export_envelope, base.build = validate_export, build
    try:
        yield
    finally:
        base.validate_export_envelope, base.build = old_validate, old_build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "run", "consume"))
    parser.add_argument("--task-export")
    parser.add_argument("--prepared")
    parser.add_argument("--prepare-sha")
    parser.add_argument("--output")
    parser.add_argument("--report-sha")
    args = parser.parse_args()
    with patched():
        if args.command == "prepare":
            result = base.prepare(args.output, task_exports=[args.task_export], task_shas=[TASK_SHA])
        elif args.command == "verify":
            result = {"status": "verified_no_execution", "roots": len(base.verify(args.prepared, args.prepare_sha)["slots"])}
        elif args.command == "run":
            result = base.run(args.prepared, args.prepare_sha, args.output)
        else:
            result = base.consume(args.output, args.report_sha)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
