"""Post-hoc eligible-only accounting from an existing paid fixed-panel run.

This does not dispatch neural calls, does not turn unused work into a refund,
and does not claim a prospective independently executed comparison.
"""
import argparse
from pathlib import Path
import subprocess
import sys

from experiments.spectral_reachability import codex_p3_quota_host_20261030_v2 as host
from experiments.spectral_reachability import codex_p3_quota_journal_20261030_v1 as journal
from experiments.spectral_reachability import codex_p3_quota_saved_consumer_20261030_v1 as saved

NAME = "codex_p3_quota_no_shadow_saved_20261030_v1"
RUN = host.ROOT / "results/codex_p3_quota_host_20261030_v2_RUN_01"
REPORT_SHA = "f3f818f2026bc86df90983ad31f52f81a6ddc1c49175b24fbf7e40c08f699565"


def summarize(events, tasks, banks, weights):
    checked = saved.verify_events(events, tasks, banks, weights)
    rows = []
    for e in events:
        if e.get("kind") != "candidate_complete":
            continue
        r = e["row"]
        rows.append({"policy": r["policy"], "root": r["group"], "index": r["index"],
                     "eligible": r["eligible"], "candidate_sha256": r["candidate_sha256"],
                     "model_cost": r["cost"]})
    policies = ("uniform", "signal", "reverse")
    arms = []
    for policy, winners in zip(policies, checked["winners"], strict=True):
        groups = [[r for r in rows if r["policy"] == policy and r["root"] == root]
                  for root in (0, 1)]
        eligible = [[r for r in group if r["eligible"]] for group in groups]
        if [len(g) for g in groups] != [15, 15] or sum(map(len, eligible)) != 20:
            raise ValueError("candidate census")
        for group, winner in zip(eligible, winners, strict=True):
            chosen = min(group, key=lambda r: (r["model_cost"], r["index"]))
            if (chosen["index"], chosen["model_cost"], chosen["candidate_sha256"]) != (
                winner["index"], winner["cost"], winner["candidate_sha256"]
            ):
                raise ValueError("winner identity")
        arms.append({"policy": policy, "actual_candidates_by_root": [len(x) for x in groups],
                     "eligible_candidates_by_root": [len(x) for x in eligible],
                     "actual_neural_axis_calls_by_root": [30, 30],
                     "eligible_subset_axis_calls_by_root": [2 * len(x) for x in eligible],
                     "actual_shadow_candidate_count": sum(len(g) for g in groups) - sum(len(g) for g in eligible),
                     "winners": [{"root": w["group"], "index": w["index"], "model_cost": w["cost"],
                                  "candidate_sha256": w["candidate_sha256"]} for w in winners]})
    return {"schema": NAME, "status": "posthoc_saved_subset_description",
            "report_sha256": REPORT_SHA, "arms": arms,
            "actual_measured_axes": list(host.bridge.AXES),
            "actual_measured_axis_total_per_arm": 60,
            "subset_counterfactual_axis_total_per_arm": 40,
            "subset_counterfactual_is_not_actual_separate_run": True,
            "actual_shadow_not_refunded": True, "neural_runs_this_analysis": 0,
            "signal_acquisition": "historical shared; standalone attribution unknown",
            "model_loading_cuda_context_cost": "unknown",
            "FLOPs": None, "per_arm_wall_seconds": None,
            "source_authenticated": False, "module_arrays_independently_reconstructed": False,
            "physical_certified": False, "scientific_pass": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = host.prior.prior.read(RUN / "report.json")
    host.require(host.prior.sha(RUN / "report.json") == REPORT_SHA and
                 report["status"] == "panel_completed_unverified", "pinned completed report")
    subprocess.run([sys.executable, "-B", "-m", "experiments.spectral_reachability." + host.NAME,
                    "consume", "--run", str(RUN),
                    "--sha256", REPORT_SHA], check=True, stdout=subprocess.DEVNULL)
    plan = host.prior.prior.read(RUN / "prepare.json")
    host.require(host.prior.sha(RUN / "prepare.json") == report["prepare_sha256"], "prepare sha")
    host.identity.verify(plan["sources"])
    original, tasks, banks = host.prior.verify(host.PREVIOUS, host.PREVIOUS_SHA)
    host.require(plan["weights"] == original["weights"] and
                 plan["task_sha256"] == [host.prior.digest(t) for t in tasks] and
                 plan["bank_sha256"] == [host.prior.digest(b[:15]) for b in banks],
                 "fixed task, bank and weights")
    events = journal.read(RUN / "events.jsonl", report["journal_tail"], host.CAP)
    result = summarize(events, tasks, banks, plan["weights"])
    result["journal_tail"] = report["journal_tail"]
    result["analysis_source_sha256"] = host.prior.sha(__file__)
    output = Path(args.output)
    host.require(output.is_absolute() and output.parent == host.ROOT / "results" and
                 output.name.startswith(NAME + "_") and not output.exists() and not output.is_symlink(),
                 "exclusive analysis output")
    print(host.write_new(output, result))


if __name__ == "__main__":
    main()
