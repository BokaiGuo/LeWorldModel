"""Validate Q-CEM journal using the common all-prefix consumer."""
import argparse
import json

from experiments.spectral_reachability import audit_reacher_response_svd_cp5_cem_20261102 as audit
from experiments.spectral_reachability import reacher_pc_p1_fd_cp5_saved_20261102_codexexclusive as saved
from experiments.spectral_reachability import reacher_pc_p1_fd_cp5_host_20261102_codexexclusive as host
from experiments.spectral_reachability import reacher_response_svd_qcp5_cem_20261102 as qcem


def consume(plan, plan_sha, journal, journal_sha):
    events = saved.verify_journal(journal)
    returns = [e for e in events if e["kind"] == "arm_returned"]
    host.require(len(returns) == 26 and all(
        e["result"]["suffix_receipt"]["objective"] == "authenticated_goal_q_weighted_planning_coordinate"
        for e in returns), "Q objective receipt missing")
    previous = audit.experiment.make_plan
    try:
        audit.experiment.make_plan = qcem.make_plan
        result = audit.consume(plan, plan_sha, journal, journal_sha)
    finally:
        audit.experiment.make_plan = previous
    result["tail_objective"] = "goal_only_task_weighted_Q"
    return result


def main():
    parser = argparse.ArgumentParser()
    for key in ("plan", "plan-sha", "journal", "journal-sha", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    result = consume(args.plan, args.plan_sha, args.journal, args.journal_sha)
    host.write_new(args.output, result)
    print(json.dumps({"status": result["status"], "arms": len(result["results"]),
                      "sha256": host.bridge.sha_file(args.output)}))


if __name__ == "__main__":
    main()
