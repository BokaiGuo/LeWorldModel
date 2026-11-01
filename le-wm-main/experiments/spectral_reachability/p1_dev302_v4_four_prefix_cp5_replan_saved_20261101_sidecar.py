"""Saved-only verifier for the dev302 V4 four-prefix cp5 replanning sidecar."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path

import numpy as np

from . import p1_dev302_v4_four_prefix_cp5_replan_20261101_sidecar as runner


NAME = "p1_dev302_v4_four_prefix_cp5_replan_saved_20261101_sidecar"
ROOT = runner.ROOT
require = runner.require


def read_journal(path: Path, expected_tail: str, expected_events: int) -> list[dict]:
    raw = Path(path).read_bytes()
    require(raw and raw.endswith(b"\n"), "complete run journal")
    rows = [json.loads(line) for line in raw.splitlines()]
    require(len(rows) == expected_events, "run journal event count")
    previous = None
    for index, row in enumerate(rows):
        body = {key: value for key, value in row.items() if key != "sha256"}
        require(
            row.get("index") == index
            and row.get("previous") == previous
            and row.get("sha256") == runner.digest(body),
            "run journal hash chain",
        )
        previous = row["sha256"]
    require(previous == expected_tail, "run journal tail")
    return rows


def _resource_view(events: list[dict]) -> dict[str, dict[str, dict]]:
    attempts = {}
    returns = {}
    for event in events:
        if event["kind"] not in ("resource_attempt", "resource_return"):
            continue
        key = (event["role"], event["axis"], event["call_index"])
        require(event["role"] in runner.ROLES and event["axis"] in runner.AXIS_CAPS, "resource identity")
        require(
            type(event["samples"]) is int
            and event["samples"] > 0
            and type(event["started_seconds"]) in (int, float)
            and math.isfinite(event["started_seconds"]),
            "resource values",
        )
        if event["kind"] == "resource_attempt":
            require(key not in attempts, "duplicate resource attempt")
            attempts[key] = event
        else:
            require(
                key in attempts
                and key not in returns
                and type(event["ended_seconds"]) in (int, float)
                and math.isfinite(event["ended_seconds"])
                and event["ended_seconds"] >= event["started_seconds"]
                and event["samples"] == attempts[key]["samples"]
                and event["started_seconds"] == attempts[key]["started_seconds"],
                "matched resource return",
            )
            returns[key] = event
    require(set(attempts) == set(returns), "all resources returned")
    view = {
        role: {
            axis: {
                "attempted_calls": 0,
                "completed_calls": 0,
                "attempted_samples": 0,
                "completed_samples": 0,
            }
            for axis in runner.AXIS_CAPS
        }
        for role in runner.ROLES
    }
    indices = defaultdict(list)
    for (role, axis, call_index), attempt in attempts.items():
        indices[role, axis].append(call_index)
        row = view[role][axis]
        row["attempted_calls"] += 1
        row["completed_calls"] += 1
        row["attempted_samples"] += attempt["samples"]
        row["completed_samples"] += attempt["samples"]
    for role in runner.ROLES:
        for axis, cap in runner.AXIS_CAPS.items():
            require(
                sorted(indices[role, axis]) == list(range(cap["calls"])),
                "resource call indices",
            )
            row = view[role][axis]
            require(
                row["attempted_calls"] == row["completed_calls"] == cap["calls"]
                and row["attempted_samples"] == row["completed_samples"] == cap["samples"],
                "exact saved axis budget",
            )
    return view


def _candidate_rows(
    protocol: dict,
    report_row: dict,
    rows: list[dict],
) -> dict:
    require(len(rows) == runner.ITERATIONS * runner.SAMPLES, "candidate denominator")
    by_iteration = defaultdict(list)
    for row in rows:
        require(
            set(row)
            == {
                "iteration",
                "candidate",
                "nominal_q",
                "variant_q",
                "robust_score",
                "actions",
            },
            "candidate row shape",
        )
        require(
            type(row["iteration"]) is int
            and 0 <= row["iteration"] < runner.ITERATIONS
            and type(row["candidate"]) is int
            and 0 <= row["candidate"] < runner.SAMPLES,
            "candidate identity",
        )
        by_iteration[row["iteration"]].append(row)
    require(
        all(
            [row["candidate"] for row in by_iteration[iteration]]
            == list(range(runner.SAMPLES))
            for iteration in range(runner.ITERATIONS)
        ),
        "candidate ordering",
    )
    seeds = np.asarray(protocol["seed_tails"], np.float32)
    mean = np.asarray(report_row["old_v4_tail"], np.float32)
    std = np.full_like(mean, runner.INITIAL_STD)
    rng = np.random.default_rng(runner.SEED)
    for iteration in range(runner.ITERATIONS):
        block = by_iteration[iteration]
        bank = runner.common.make_bank(rng, mean, std, seeds, iteration)
        actual_bank = np.asarray([row["actions"] for row in block], np.float32)
        require(np.array_equal(actual_bank, bank), "saved CEM bank")
        variant_scores = np.asarray([row["variant_q"] for row in block], np.float64)
        require(
            variant_scores.shape == (runner.SAMPLES, runner.VARIANTS)
            and np.isfinite(variant_scores).all(),
            "saved variant scores",
        )
        robust = runner.common.robust_scores(variant_scores)
        require(
            np.array_equal(
                robust,
                np.asarray([row["robust_score"] for row in block], np.float64),
            )
            and np.array_equal(
                variant_scores[:, 0],
                np.asarray([row["nominal_q"] for row in block], np.float64),
            ),
            "saved robust scores",
        )
        next_mean, next_std, elite = runner.common.update(bank, robust)
        summary = report_row["iterations"][iteration]
        require(
            summary["iteration"] == iteration
            and summary["elite_indices"] == elite
            and np.array_equal(np.asarray(summary["mean"], np.float32), next_mean)
            and np.array_equal(np.asarray(summary["std"], np.float32), next_std),
            "saved CEM update",
        )
        mean, std = next_mean, next_std
    selected = min(
        rows,
        key=lambda item: (
            item["robust_score"],
            item["iteration"],
            item["candidate"],
        ),
    )
    require(report_row["selected"] == selected, "saved selected candidate")
    return {
        "selected_iteration": selected["iteration"],
        "selected_candidate": selected["candidate"],
        "selected_robust_score": selected["robust_score"],
    }


def verify_payload(protocol: dict, report: dict, events: list[dict], candidate_loader) -> dict:
    require(
        protocol["schema"] == runner.NAME + ".protocol.v1"
        and protocol["equal_cem_and_robust_budget"] is True
        and protocol["post_cp5_physical_outcomes_used"] is False
        and protocol["physical_execution"] is False
        and protocol["P1_admission"] is False,
        "protocol scientific boundary",
    )
    require(
        report.get("schema") == runner.NAME + ".report.v1"
        and report.get("status") == "model_only_complete_not_admitted"
        and report.get("error") is None
        and report.get("physical_report_sha256") == protocol["physical_report_sha256"]
        and report.get("equal_cem_and_robust_budget") is True
        and report.get("post_cp5_physical_outcomes_used") is False
        and report.get("physical_execution") is False
        and report.get("training") is False
        and report.get("independent_confirmation") is False
        and report.get("P1_admission") is False
        and report.get("model_before") == report.get("model_after")
        and report["model_before"]["original_state_dict"] == runner.MODEL_SHA256
        and report.get("module_hooks")
        == {
            "action_encoder": len(runner.ROLES) * runner.AXIS_CAPS["action_encoder"]["calls"],
            "predict": len(runner.ROLES) * runner.AXIS_CAPS["predict"]["calls"],
        },
        "report terminal boundary",
    )
    require(
        events
        and events[0]["kind"] == "run_started"
        and events[-1]["kind"] == "run_complete"
        and events[-1]["arms"] == len(runner.ROLES)
        and events[-1]["P1_admission"] is False
        and not any(event["kind"] == "run_failed" for event in events),
        "journal terminal boundary",
    )
    resource_view = _resource_view(events)
    starts = [event for event in events if event["kind"] == "arm_started"]
    completes = [event for event in events if event["kind"] == "arm_complete"]
    require(
        [event["role"] for event in starts] == list(runner.ROLES)
        and [event["role"] for event in completes] == list(runner.ROLES),
        "arm event roster",
    )
    require(
        len(report.get("rows", [])) == len(runner.ROLES)
        and [row["role"] for row in report["rows"]] == list(runner.ROLES),
        "report row roster",
    )
    protocol_rows = {row["role"]: row for row in protocol["roles"]}
    summaries = []
    for row, complete in zip(report["rows"], completes, strict=True):
        role = row["role"]
        source = protocol_rows[role]
        source_prefix_sha = runner.action_sha(source["actual_prefix"])
        row_prefix_sha = runner.action_sha(row["actual_prefix"])
        source_tail_sha = runner.action_sha(source["old_v4_tail"])
        row_tail_sha = runner.action_sha(row["old_v4_tail"])
        require(
            row["status"] == "replanned"
            and row["candidate_index"] == source["candidate_index"]
            and row["candidate_label"] == source["candidate_label"]
            and row["cp5_pixel_sha256"] == source["cp5_pixel_sha256"]
            and row["actual_prefix"] == source["actual_prefix"]
            and source_prefix_sha
            == row_prefix_sha
            == row["actual_prefix_sha256"]
            == source["actual_prefix_sha256"]
            and row["old_v4_tail"] == source["old_v4_tail"]
            and source_tail_sha == row_tail_sha == source["old_v4_tail_sha256"]
            and row["axis_ledger"] == resource_view[role]
            and complete["axis_ledger"] == resource_view[role],
            "row source and ledger",
        )
        candidates = candidate_loader(row)
        summary = _candidate_rows(protocol, row, candidates)
        require(
            complete["selected_iteration"] == summary["selected_iteration"]
            and complete["selected_candidate"] == summary["selected_candidate"]
            and complete["selected_robust_score"] == summary["selected_robust_score"]
            and complete["candidate_file_sha256"] == row["candidate_file_sha256"],
            "arm completion summary",
        )
        summaries.append({"role": role, **summary, "axis_ledger": resource_view[role]})
    require(
        all(resource_view[role] == resource_view[runner.ROLES[0]] for role in runner.ROLES),
        "identical four-arm measured budget",
    )
    return {
        "schema": NAME + ".result.v1",
        "status": "saved_model_only_four_prefix_replan_verified_not_admitted",
        "rows": summaries,
        "equal_measured_axis_budget": True,
        "actual_cp5_rgb_and_action_history_bound": True,
        "physical_execution": False,
        "independent_confirmation": False,
        "P1_admission": False,
    }


def verify(prepared: Path, prepared_sha256: str, directory: Path, report_sha256: str) -> dict:
    prepared = Path(prepared).resolve(strict=True)
    directory = Path(directory).resolve(strict=True)
    require(
        runner.sha256(prepared) == prepared_sha256
        and directory.parent == ROOT / "results"
        and not directory.is_symlink(),
        "external paths and SHA",
    )
    protocol = runner.read_json(prepared)
    require(
        protocol
        == runner.protocol(
            Path(protocol["physical_directory"]),
            protocol["physical_report_sha256"],
        ),
        "protocol reconstruction",
    )
    report_path = directory / "report.json"
    require(runner.sha256(report_path) == report_sha256, "report SHA")
    report = runner.read_json(report_path)
    require(report["protocol_sha256"] == prepared_sha256, "report/protocol binding")
    events = read_journal(
        directory / "events.jsonl",
        report["journal_tail"],
        report["journal_events"],
    )

    def load_candidates(row):
        path = (directory / row["candidate_file"]).resolve(strict=True)
        require(
            path.is_relative_to(directory)
            and not path.is_symlink()
            and runner.sha256(path) == row["candidate_file_sha256"],
            "candidate file identity",
        )
        return runner.candidate_file_rows(path)

    result = verify_payload(protocol, report, events, load_candidates)
    result.update(
        {
            "prepared": str(prepared),
            "prepare_sha256": prepared_sha256,
            "directory": str(directory),
            "report_sha256": report_sha256,
            "journal_sha256": runner.sha256(directory / "events.jsonl"),
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepare-sha256", required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).absolute()
    require(
        output.parent == ROOT / "results"
        and output.name.startswith(NAME + "_")
        and not os.path.lexists(output),
        "exclusive output",
    )
    value = verify(
        Path(args.prepared),
        args.prepare_sha256,
        Path(args.directory),
        args.report_sha256,
    )
    digest = runner.write_new(output, value)
    print(json.dumps({"output": str(output), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
