"""Verify and analyze the fixed P1 full-tail cross intervention."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import p1_full_tail_cross_intervention_20261101_exclusive as experiment
from . import p1_moving_live_feedback_analysis_20261027_v2 as journal

engine = experiment.engine
ROOT = experiment.ROOT
NAME = "p1_full_tail_cross_verify_20261101_exclusive"
METRICS = ("native_distance", "object_position_error", "wrapped_object_angle_error")


def mean(values):
    array = np.asarray(values, dtype=np.float64)
    engine.require(array.ndim == 1 and len(array) > 0 and np.isfinite(array).all(), "finite mean")
    return float(array.mean())


def analyze(protocol, rows):
    lookup = {(row["source_slot"], row["prefix_role"], row["suffix_role"]): row for row in rows}
    roots = {}
    for root in protocol["roots"]:
        slot = root["root_slot"]
        tail_labels = [tail["label"] for tail in root["tails"]]
        engine.require(tail_labels[0] == "zero", "zero-tail baseline first")
        prefix_metrics = {}
        for role in experiment.ROLES:
            row = lookup[(slot, role, "zero")]
            cp5 = row["prefix_checkpoint"]
            prefix_metrics[role] = engine.old.kernel.metric(root["root"]["goal"], cp5["state"])
        comparisons = []
        for tail in tail_labels:
            selected = lookup[(slot, "selected", tail)]
            zero = lookup[(slot, "zero", tail)]
            greedy = lookup[(slot, "greedy", tail)]
            item = {
                "tail": tail,
                "tail_sha256": next(x["sha256"] for x in root["tails"] if x["label"] == tail),
                "statuses": {role: lookup[(slot, role, tail)]["status"] for role in experiment.ROLES},
                "selected_minus_zero": {
                    metric: float(selected["final_metrics"][metric] - zero["final_metrics"][metric])
                    for metric in METRICS
                },
                "selected_minus_greedy": {
                    metric: float(selected["final_metrics"][metric] - greedy["final_metrics"][metric])
                    for metric in METRICS
                },
                "success": {
                    role: bool(lookup[(slot, role, tail)]["final_metrics"]["native_success"])
                    for role in experiment.ROLES
                },
            }
            comparisons.append(item)
        nonzero = comparisons[1:]
        aggregate = {
            "tail_count": len(nonzero),
            "selected_minus_zero_mean": {
                metric: mean([x["selected_minus_zero"][metric] for x in nonzero])
                for metric in METRICS
            },
            "selected_minus_greedy_mean": {
                metric: mean([x["selected_minus_greedy"][metric] for x in nonzero])
                for metric in METRICS
            },
            "success_counts": {
                role: sum(x["success"][role] for x in nonzero)
                for role in experiment.ROLES
            },
            "selected_success_without_zero": sum(
                x["success"]["selected"] and not x["success"]["zero"] for x in nonzero
            ),
            "zero_success_without_selected": sum(
                x["success"]["zero"] and not x["success"]["selected"] for x in nonzero
            ),
            "selected_success_without_greedy": sum(
                x["success"]["selected"] and not x["success"]["greedy"] for x in nonzero
            ),
            "greedy_success_without_selected": sum(
                x["success"]["greedy"] and not x["success"]["selected"] for x in nonzero
            ),
        }
        roots[str(slot)] = {
            "prefix_metrics": prefix_metrics,
            "tail_comparisons": comparisons,
            "whole_bank_aggregate": aggregate,
            "selected_greedy_prefix_alias": bool(np.array_equal(
                np.asarray(root["prefixes"]["selected"], np.float32),
                np.asarray(root["prefixes"]["greedy"], np.float32),
            )),
        }
    return roots


def verify(protocol_path, protocol_sha, directory):
    engine.require(engine.sha(protocol_path) == protocol_sha, "protocol SHA")
    protocol = json.loads(protocol_path.read_bytes())
    engine.require(protocol == experiment.build() and protocol["planned_branches"] == 30,
                   "protocol/source closure")
    report_path = directory / "report.json"
    report = json.loads(report_path.read_bytes())
    engine.require(report["status"] == "recorded_development" and not report["errors"]
                   and report["protocol_sha256"] == protocol_sha
                   and len(report["rows"]) == protocol["planned_branches"], "complete report")
    engine.require(engine.sha(directory / "journal.jsonl") == report["journal_sha256"], "journal SHA")
    journal.journal((directory / "journal.jsonl").read_bytes(), report["journal_chain_tail"])
    rows = sorted(report["rows"], key=lambda row: row["global_index"])
    engine.require([row["global_index"] for row in rows] == list(range(30)), "global branch roster")
    for branch, row in zip(protocol["branches"], rows, strict=True):
        engine.require(
            row["source_slot"] == branch["root_slot"]
            and row["branch"] == branch["index"]
            and row["candidate"] == branch["candidate"]
            and row["prefix_role"] == branch["prefix_role"]
            and row["suffix_role"] == branch["suffix_role"]
            and row["suffix_aliases"] == branch["suffix_aliases"],
            "branch identity",
        )
        expected = np.asarray(branch["commands"], np.float32)
        actual = np.asarray(row["executed_raw"], np.float32)
        engine.require(row["status"] in ("completed15", "native_stopped", "censored_out_of_domain")
                       and 5 <= len(actual) <= 15
                       and np.array_equal(actual, expected[:len(actual)]), "actual action prefix")
        engine.require(row["prefix_checkpoint"] == next(
            root["prefix_checks"][branch["prefix_role"]]
            for root in protocol["roots"] if root["root_slot"] == branch["root_slot"]
        ), "cp5 identity")
        engine.require(row.get("final_metrics") is not None, "final metrics")
        saved = directory / f"r{branch['root_slot']}_b{branch['index']:02d}_result.json"
        engine.require(json.loads(saved.read_bytes()) == row, "row shard")
    engine.require(all(engine.sha(path) == digest for path, digest in protocol["bindings"].items()),
                   "post-run source closure")
    return {
        "schema": NAME + ".result.v1",
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "report": str(report_path),
        "report_sha256": engine.sha(report_path),
        "planned_branches": 30,
        "verified_branches": len(rows),
        "analysis": analyze(protocol, rows),
        "interpretation": {
            "whole_bank_only": True,
            "best_tail_selection_forbidden": True,
            "known_development_roots": 2,
            "root1_selected_greedy_identity_alias": True,
            "full_end_to_end_budget_fairness": False,
            "independent_confirmation": False,
            "P1_admission": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    engine.require(output.is_absolute() and output.parent == ROOT / "results"
                   and not os.path.lexists(output), "exclusive result output")
    result = verify(Path(args.protocol), args.protocol_sha256, Path(args.directory))
    raw = experiment.canonical_raw(result)
    with output.open("xb") as stream:
        engine.require(stream.write(raw) == len(raw), "complete result write")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"output": str(output), "sha256": engine.sha(output),
                      "verified_branches": result["verified_branches"]}))


if __name__ == "__main__":
    main()
