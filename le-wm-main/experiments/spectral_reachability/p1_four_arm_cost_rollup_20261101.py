"""CPU-only rollup of existing four-arm node journals and episode meters.

Usage: python -m experiments.spectral_reachability.p1_four_arm_cost_rollup_20261101
       REPORT.json EXPECTED_SHA256

This consumes saved local executions, not author-reported results. Output is
printed to stdout; no file is created. It does not certify meter coverage.
"""
import hashlib
import json
import sys
from pathlib import Path

from . import codex_four_arm_consumer_20261030 as consumer
from . import codex_four_arm_full_cost_20261030 as spec


def rollup(path, expected_sha):
    path = Path(path)
    spec.require(path.is_file() and not path.is_symlink() and spec.is_hash(expected_sha), "report identity")
    raw = path.read_bytes()
    spec.require(hashlib.sha256(raw).hexdigest() == expected_sha, "report digest")
    report = json.loads(raw, object_pairs_hook=consumer.unique,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    protocol = report["protocol"]
    boundaries = protocol["boundaries"]
    spec.require(type(boundaries) is list and boundaries and
                 all(type(x) is int and x >= 0 for x in boundaries) and
                 boundaries == sorted(set(boundaries)), "boundary schedule")
    spec.vector(protocol["node_caps"])
    rows = report["rows"]
    spec.require(type(rows) is list and len(rows) == len(spec.ARMS) and
                 {r["arm"] for r in rows} == set(spec.ARMS) and
                 len({r["episode_id"] for r in rows}) == len(rows), "arm roster")
    result = {}
    for row in rows:
        arm, episode = row["arm"], row["episode_id"]
        nodes = report["nodes"][episode]
        spec.require(len(nodes) == len(boundaries) and
                     [n["boundary"] for n in nodes] == boundaries and
                     all(n["status"] == "sealed" for n in nodes), "incomplete nodes")
        total = dict.fromkeys(spec.AXES, 0)
        for node in nodes:
            receipt = node["receipt"]
            ledger = receipt["ledger"]
            journal = Path(ledger["journal"])
            # Bind each referenced journal to this report directory.
            spec.require(journal.is_absolute() and journal.resolve().is_relative_to(path.parent.resolve())
                         and journal.name == "events.jsonl", "journal path")
            account = consumer.consume(str(journal), ledger["sha256"], receipt["contract_sha256"],
                                       arm, "node", arm)
            spec.require(account == receipt["account"] and account["status"] == "closed_unverified",
                         "node account mismatch")
            cost = spec.vector(account["charged_upper_bounds"])
            spec.require(all(cost[k] <= protocol["node_caps"][k] for k in spec.AXES), "node cap mismatch")
            audit = receipt.get("module_hook_audit")
            if audit is not None:
                spec.require(audit["arm"] == arm and audit["boundary"] == node["boundary"] and
                             audit["ledger_charged_upper_bounds"] == cost and
                             audit["module_forward_pre_hooks"] ==
                             {k: cost[k] for k in ("predict_calls", "action_encode_calls")},
                             "hook mismatch")
                if "autograd_api_counts" in audit:
                    a = audit["autograd_api_counts"]
                    spec.require(a["grad_attempted"] == a["grad_completed"] == cost["backward_calls"]
                                 and a["backward_attempted"] == 0, "autograd mismatch")
            for key in spec.AXES:
                total[key] += cost[key]
        meters = row["physical_visual_counts"]
        spec.require(type(meters) is dict and meters, "missing episode meters")
        for key, counts in meters.items():
            spec.require(type(key) is str and type(counts) is dict and set(counts) == {"attempted", "completed"}
                         and all(type(v) is int and v >= 0 for v in counts.values())
                         and counts["completed"] <= counts["attempted"], "invalid episode meter")
        result[arm] = {"node_count": len(nodes), "planning_charged_upper_bounds": total,
                       "physical_visual_meter": meters, "episode_status": row["status"]}
    spec.require(report["status"] == "recorded_known_root_not_admitted" and not report["errors"],
                 "partial or failed report")
    return {"schema": "p1_four_arm_cost_rollup_20261101.v1", "report_sha256": expected_sha,
            "feedback": protocol["feedback"], "boundaries": boundaries, "arms": result,
            "node_caps_are_per_axis_not_equal_compute": True, "hook_coverage_verified": False,
            "episode_meter_independently_replayed": False, "full_cost_certified": False,
            "fairness_proven": False, "P1_admission": False}


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(json.dumps(rollup(sys.argv[1], sys.argv[2]), sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
