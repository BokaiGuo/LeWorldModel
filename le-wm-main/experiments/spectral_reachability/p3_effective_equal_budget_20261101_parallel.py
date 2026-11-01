"""Saved-first P3 equal-axis comparison with an optional host-CUDA refresh.

The scientific comparison is deliberately narrow: ``signal`` and ``reverse``
pay the same recorded resource vector at each fixed budget point, but allocate
the candidate quota to the two known roots in opposite directions.  Equality
of recorded calls is not claimed to be equality of FLOPs, energy, or total
compute.

This module never trains or mutates the frozen model.  The default ``verify``
action only consumes saved artifacts.  ``run`` delegates the neural execution
to the already-audited host-CUDA child and writes a new, exclusive run folder.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from experiments.spectral_reachability import codex_p3_online_budget_host_20261030_v1 as host
from experiments.spectral_reachability import codex_p3_quota_journal_20261030_v1 as journal


ROOT = host.ROOT
NAME = "p3_effective_equal_budget_20261101_parallel"
PRIMARY_RUN = ROOT / "results/codex_p3_online_budget_host_20261030_v1_RUN_01"
PRIMARY_CONSUME = ROOT / "results/codex_p3_online_budget_host_20261030_v1_CONSUME_01.json"
RESOLVED_CURVE = ROOT / "results/codex_p3_resolved_budget_curve_20261030_v1_CODEX02.json"

PRIMARY_REPORT_SHA256 = "7d2ff7ef0c554077cf0a2461ab733d80f0f14a5658bdf1c554fd58ac6f4dacde"
PRIMARY_PREPARE_SHA256 = "bfab85393221afe02f8ac804130040c4f493d5d3deffdeb5486e14cc6e5cb305"
PRIMARY_EVENTS_SHA256 = "e6e2e51cb7eb8a809ae26ca052ee7397c0382b23ada318b48c758eaade9387ad"
PRIMARY_CONSUME_SHA256 = "03a3eaa5f7cce26298ab578440c02dac87b7d984ab0e2cb3eccfdc07c28adc67"
RESOLVED_CURVE_SHA256 = "3d08a2014aa2ad3b0cbef6491a5e2936c14e41203a9b5f90fd9280ca1cf02c2c"

POINTS = (20, 30)
ARMS = ("signal", "reverse")
FIXED_ALLOCATIONS = {
    20: {"signal": [15, 5], "reverse": [5, 15]},
    30: {"signal": [22, 8], "reverse": [8, 22]},
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path: Path, expected_sha256: str) -> dict:
    require(sha256(path) == expected_sha256, f"sha256 mismatch: {path.name}")
    value = json.loads(path.read_bytes())
    require(type(value) is dict, f"JSON object required: {path.name}")
    return value


def _rows_by_key(saved: dict) -> dict[tuple[int, str], dict]:
    require(
        saved.get("status") == "semantic_saved_checked_not_scientifically_admitted"
        and saved.get("scientific_pass") is False
        and saved.get("physical_certified") is False
        and saved.get("_all_operations_returned_verified") is True
        and len(saved.get("rows", [])) == 6,
        "primary saved boundary",
    )
    rows = {}
    for row in saved["rows"]:
        key = (row.get("point"), row.get("arm"))
        require(key not in rows, "duplicate primary row")
        rows[key] = row
    require(
        set(rows) == {(point, arm) for point in POINTS for arm in ("uniform",) + ARMS},
        "primary fixed roster",
    )
    return rows


def _certify_returned(saved: dict) -> dict:
    """Annotate the information proved, but omitted, by the old consumer."""
    result = copy.deepcopy(saved)
    for row in result.get("rows", []):
        require("returned" not in row, "unexpected legacy returned field")
        row["returned"] = copy.deepcopy(row["attempted"])
    result["_all_operations_returned_verified"] = True
    return result


def _verify_prepare_saved(path: Path, expected_sha256: str) -> tuple[dict, list[dict], list[list[dict]]]:
    """Verify a prepare without the old process-import-dependent closure check."""
    plan = load_json(path, expected_sha256)
    host.identity.verify(plan["sources"])
    predecessor, tasks, _ = host.prior.verify(host.PREVIOUS, host.PREVIOUS_SHA)
    banks = host.bridge.banks(host.prior.dependency().old.BALL)
    for bank in banks:
        for candidate in bank:
            host.prior.dependency().old.legal(candidate["commands"], candidate["logits"])
    require(
        plan["predecessor_sha256"] == host.PREVIOUS_SHA
        and plan["task_sha256"] == [host.prior.digest(task) for task in tasks]
        and plan["bank_sha256"] == [host.prior.digest(bank) for bank in banks]
        and plan["model_path"] == str(Path(predecessor["policy"]).resolve())
        and plan["expected_model_hash"] == predecessor["expected_model_hash"],
        "prepare reconstructed inputs",
    )
    return plan, tasks, banks


def _finite_costs(row: dict) -> list[float]:
    costs = [winner.get("cost") for winner in row.get("winners", [])]
    require(
        len(costs) == 2
        and all(type(value) in (int, float) and math.isfinite(value) for value in costs),
        "two finite winner costs",
    )
    return costs


def _resource_timing(events: list[dict], point: int, arm: str) -> dict:
    attempts = {}
    returns = {}
    for event in events:
        if event.get("kind") not in ("resource_attempt", "resource_return"):
            continue
        entry = event.get("entry", {})
        if entry.get("scope", [None, None])[:2] != [point, arm]:
            continue
        index = entry.get("index")
        if event["kind"] == "resource_attempt":
            require(index not in attempts, "duplicate timing attempt")
            attempts[index] = entry
        else:
            require(index not in returns, "duplicate timing return")
            returns[index] = entry
    require(attempts and set(attempts) == set(returns), "complete timing pairs")
    durations = []
    for index, attempt in attempts.items():
        returned = returns[index]
        require(
            returned.get("resource") == attempt.get("resource")
            and returned.get("scope") == attempt.get("scope")
            and returned.get("ended_seconds", -math.inf) >= attempt.get("started_seconds", math.inf),
            "timing pair identity",
        )
        durations.append(returned["ended_seconds"] - attempt["started_seconds"])
    start = min(entry["started_seconds"] for entry in attempts.values())
    end = max(entry["ended_seconds"] for entry in returns.values())
    return {
        "resource_calls": len(attempts),
        "summed_resource_seconds": sum(durations),
        "resource_span_seconds": end - start,
        "scope": "instrumented resource calls; excludes unmetered host work",
    }


def verify_primary_run(
    run: Path,
    report_sha256: str,
    expected_consume: dict | None = None,
) -> tuple[dict, list[dict], dict]:
    """Rebuild saved semantics and independently read the journal hash chain."""
    run = Path(run)
    require(not run.is_symlink(), "run symlink forbidden")
    report = load_json(run / "report.json", report_sha256)
    require(
        report.get("status") == "panel_completed_unverified"
        and report.get("error") is None
        and report.get("scientific_pass") is False
        and report.get("physical_release") is False
        and report.get("model_before") == report.get("model_after"),
        "host terminal boundary",
    )
    prepare_sha = report.get("prepare_sha256")
    require(type(prepare_sha) is str and sha256(run / "prepare.json") == prepare_sha, "prepare binding")
    plan, tasks, banks = _verify_prepare_saved(run / "prepare.json", prepare_sha)
    require(
        plan.get("known_roots") is True
        and plan.get("physical_release") is False
        and plan.get("scientific_pass") is False
        and plan.get("budget_points") == list(POINTS),
        "prepare scientific boundary",
    )
    events = journal.read(run / "events.jsonl", report["journal_tail"], host.CAP)
    require(len(events) == report.get("journal_events"), "journal event count")
    rebuilt = host.consumer.verify_events(events, tasks, banks)
    if expected_consume is not None:
        require(rebuilt == expected_consume, "saved consume reconstruction")
    return report, events, _certify_returned(rebuilt)


def _resolved_wall_rows(resolved: dict) -> list[dict]:
    require(
        resolved.get("schema") == "codex_p3_resolved_budget_curve_20261030_v1"
        and resolved.get("P3_admission") is False
        and resolved.get("independent_confirmation") is False
        and resolved.get("full_total_compute_equal") is False
        and resolved.get("flops") == "unknown"
        and resolved.get("energy") == "unknown",
        "resolved curve boundary",
    )
    result = []
    points = resolved.get("points", [])
    require([row.get("budget_point") for row in points] == list(POINTS), "resolved points")
    for point_row in points:
        point = point_row["budget_point"]
        arms = {row.get("arm"): row for row in point_row.get("arms", [])}
        require(set(arms) == {"uniform", "signal", "reverse"}, "resolved roster")
        for arm in ARMS:
            row = arms[arm]
            require(
                row.get("status") == "consumed"
                and type(row.get("wall_seconds")) in (int, float)
                and math.isfinite(row["wall_seconds"])
                and row["wall_seconds"] >= 0
                and row.get("axis_attempted") == row.get("axis_returned"),
                "resolved consumed timing row",
            )
            result.append(
                {
                    "budget_point": point,
                    "arm": arm,
                    "allocation": row["allocation"],
                    "process_wall_seconds": row["wall_seconds"],
                    "scheduler_meter_seconds": row["scheduler_meter_seconds"],
                    "axis_attempted": row["axis_attempted"],
                    "root_model_costs": row["root_model_costs"],
                }
            )
    return result


def assemble(
    primary_saved: dict,
    primary_events: list[dict],
    primary_report: dict,
    resolved: dict,
    *,
    source_hashes: dict,
) -> dict:
    """Build the fail-closed comparison from already verified inputs."""
    rows = _rows_by_key(primary_saved)
    curve = []
    for point in POINTS:
        signal = rows[point, "signal"]
        reverse = rows[point, "reverse"]
        expected = FIXED_ALLOCATIONS[point]
        require(signal.get("allocation") == expected["signal"], "fixed signal allocation")
        require(reverse.get("allocation") == expected["reverse"], "fixed reverse allocation")
        require(signal["allocation"] == list(reversed(reverse["allocation"])), "opposite allocation")
        require(sum(signal["allocation"]) == sum(reverse["allocation"]) == point, "equal candidate total")
        require(
            signal.get("weights") == reverse.get("weights")
            and len(signal["weights"]) == 2
            and signal["weights"][0] != signal["weights"][1],
            "shared unequal online weights",
        )
        require(
            signal.get("attempted") == signal.get("returned")
            and reverse.get("attempted") == reverse.get("returned")
            and signal["attempted"] == reverse["attempted"],
            "equal completed measured axes",
        )
        signal_costs = _finite_costs(signal)
        reverse_costs = _finite_costs(reverse)
        curve.append(
            {
                "budget_point": point,
                "candidate_total_per_arm": point,
                "signal": {
                    "allocation": copy.deepcopy(signal["allocation"]),
                    "root_model_costs": signal_costs,
                    "mean_model_cost": sum(signal_costs) / 2,
                    "timing": _resource_timing(primary_events, point, "signal"),
                },
                "reverse": {
                    "allocation": copy.deepcopy(reverse["allocation"]),
                    "root_model_costs": reverse_costs,
                    "mean_model_cost": sum(reverse_costs) / 2,
                    "timing": _resource_timing(primary_events, point, "reverse"),
                },
                "signal_minus_reverse_root_model_cost": [
                    a - b for a, b in zip(signal_costs, reverse_costs, strict=True)
                ],
                "signal_minus_reverse_mean_model_cost": (
                    sum(signal_costs) - sum(reverse_costs)
                )
                / 2,
                "attempted_and_returned_each": copy.deepcopy(signal["attempted"]),
                "equal_measured_axes": True,
                "allocation_strategies_differ": True,
            }
        )
    resolved_rows = _resolved_wall_rows(resolved)
    return {
        "schema": NAME,
        "status": "saved_and_host_cuda_evidence_checked_not_scientifically_admitted",
        "fixed_before_analysis": {
            "budget_points": list(POINTS),
            "arms": list(ARMS),
            "allocations": {str(key): value for key, value in FIXED_ALLOCATIONS.items()},
            "selection_rule": "compare signal with reverse at every fixed point; no budget selection",
        },
        "primary_online_cuda_contract": {
            "source_hashes": source_hashes,
            "host_elapsed_seconds": primary_report["elapsed_seconds"],
            "gpu_snapshot": primary_report["gpu_snapshot"],
            "device_binding": primary_report["device_binding"],
            "model_before": primary_report["model_before"],
            "model_after": primary_report["model_after"],
            "peak_allocated_bytes": primary_report["peak_allocated_bytes"],
            "peak_reserved_bytes": primary_report["peak_reserved_bytes"],
            "curve": curve,
        },
        "auxiliary_resolved_wall_contract": {
            "rows": resolved_rows,
            "scope": resolved["wall_scope"],
            "warning": (
                "different execution contract and allocations; timing is descriptive only "
                "and is not joined to the primary online-CUDA cost curve"
            ),
        },
        "budget_accounting": {
            "equal_measured_axes_signal_vs_reverse": True,
            "equal_candidate_total_signal_vs_reverse": True,
            "actual_calls_reported": True,
            "actual_elapsed_reported": True,
            "budget_curve_reported": True,
            "full_total_compute_equal": False,
            "flops": "unknown",
            "energy": "unknown",
            "forced_padding": False,
        },
        "scientific_boundary": {
            "known_roots": True,
            "physical_outcome": False,
            "independent_confirmation": False,
            "model_cost_only": True,
            "P3_admission": False,
            "C3_admission": False,
            "gate_relaxed": False,
        },
    }


def build_saved() -> dict:
    primary_consume = load_json(PRIMARY_CONSUME, PRIMARY_CONSUME_SHA256)
    primary_report, primary_events, rebuilt = verify_primary_run(
        PRIMARY_RUN, PRIMARY_REPORT_SHA256, primary_consume
    )
    require(sha256(PRIMARY_RUN / "prepare.json") == PRIMARY_PREPARE_SHA256, "pinned prepare")
    require(sha256(PRIMARY_RUN / "events.jsonl") == PRIMARY_EVENTS_SHA256, "pinned events")
    resolved = load_json(RESOLVED_CURVE, RESOLVED_CURVE_SHA256)
    return assemble(
        rebuilt,
        primary_events,
        primary_report,
        resolved,
        source_hashes={
            "report_sha256": PRIMARY_REPORT_SHA256,
            "prepare_sha256": PRIMARY_PREPARE_SHA256,
            "events_sha256": PRIMARY_EVENTS_SHA256,
            "consume_sha256": PRIMARY_CONSUME_SHA256,
            "resolved_curve_sha256": RESOLVED_CURVE_SHA256,
        },
    )


def _exclusive_result(path: str) -> Path:
    output = Path(path)
    require(
        output.is_absolute()
        and output.parent == ROOT / "results"
        and output.name.startswith("p3_")
        and output.name.endswith("_20261101_parallel.json")
        and not os.path.lexists(output),
        "exclusive result path",
    )
    return output


def _exclusive_run(path: str) -> Path:
    output = Path(path)
    require(
        output.is_absolute()
        and output.parent == ROOT / "results"
        and output.name.startswith("p3_")
        and output.name.endswith("_20261101_parallel")
        and not os.path.lexists(output),
        "exclusive run path",
    )
    return output


def run_host_cuda(path: str) -> tuple[Path, dict, list[dict], dict]:
    """Run the fixed old kernel under a new exclusive result identity."""
    output = _exclusive_run(path)
    plan, tasks, banks = host.material()
    require(
        plan["budget_points"] == list(POINTS)
        and plan["roster"] == ["uniform", "signal", "reverse"]
        and plan["known_roots"] is True
        and plan["physical_release"] is False,
        "fixed host plan",
    )
    require(os.stat(host.LOCK, follow_symlinks=False), "GPU lock missing")
    with host.lock.gpu_lock(host.LOCK):
        snapshot = host.gpu_snapshot()
        output.mkdir(mode=0o700)
        host.write_new(output / "prepare.json", plan)
        pid = os.fork()
        if pid == 0:
            try:
                fds = [int(value) for value in os.listdir("/proc/self/fd") if value.isdigit()]
                lock_fd = next(fd for fd in fds if fd > 2 and host._same_lock(fd))
                report = host.child(plan, tasks, banks, output, lock_fd, snapshot)
                os._exit(0 if report["status"] == "panel_completed_unverified" else 1)
            except BaseException:
                os._exit(1)
        _, status = os.waitpid(pid, 0)
    require(os.waitstatus_to_exitcode(status) == 0, "host CUDA child failed; output preserved")
    report_sha = sha256(output / "report.json")
    report, events, saved = verify_primary_run(output, report_sha)
    # This assertion makes runtime quota equality impossible to pass silently.
    _ = assemble(
        saved,
        events,
        report,
        load_json(RESOLVED_CURVE, RESOLVED_CURVE_SHA256),
        source_hashes={
            "report_sha256": report_sha,
            "prepare_sha256": sha256(output / "prepare.json"),
            "events_sha256": sha256(output / "events.jsonl"),
            "consume_sha256": "rebuilt_in_memory",
            "resolved_curve_sha256": RESOLVED_CURVE_SHA256,
        },
    )
    return output, report, events, saved


def build_from_run(path: str) -> dict:
    run = Path(path)
    report_sha = sha256(run / "report.json")
    report, events, saved = verify_primary_run(run, report_sha)
    return assemble(
        saved,
        events,
        report,
        load_json(RESOLVED_CURVE, RESOLVED_CURVE_SHA256),
        source_hashes={
            "report_sha256": report_sha,
            "prepare_sha256": sha256(run / "prepare.json"),
            "events_sha256": sha256(run / "events.jsonl"),
            "consume_sha256": "rebuilt_in_memory",
            "resolved_curve_sha256": RESOLVED_CURVE_SHA256,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("verify", "run"))
    parser.add_argument("--run")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "verify":
        result = build_from_run(args.run) if args.run else build_saved()
    else:
        require(args.run is not None, "--run is required for host CUDA")
        run_host_cuda(args.run)
        result = build_from_run(args.run)
    output = _exclusive_result(args.output)
    digest = host.write_new(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": digest,
                "points": len(result["primary_online_cuda_contract"]["curve"]),
                "P3_admission": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
