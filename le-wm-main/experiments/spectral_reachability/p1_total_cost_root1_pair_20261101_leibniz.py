"""Independent saved root1 structured/CEM reconciliation; no model or physics execution."""
import argparse
import hashlib
import json
from pathlib import Path

from . import p1_total_cost_reconcile_20261101_leibniz as costs
from . import p1_common_cp10_pair_audit_20261101_leibniz as pair
from . import codex_p1_structured_root1_cp10_saved_20261101_exclusive as saved
from . import codex_p1_variable_cem_root0_audit_20261101_exclusive as cem_audit
from . import codex_p1_variable_cem_host_v2_20261101_exclusive as anchor

ROSTER = {
    "structured": ("codex_p1_structured_root1_cp10_20261101_exclusive_RUN_CODEXROOT1DEV01",
                   "762eb75e9338abf38d4e546a13ad04ae6a480255a0d5430cdab67407810ee6d6"),
    "cem": costs.ROSTER["root1_cem"],
}


def verify_compatibility(reports, prepared):
    s, c = reports["structured"], reports["cem"]
    costs.require(s["cp10_goal_binding"] == c["cp10_goal_binding"]
                  == prepared["structured"]["cp10_identity"]
                  == prepared["cem"]["cem_source_identity"]["rebound_goal_binding"], "goal/root mismatch")
    costs.require(s["model_before"] == s["model_after"] == c["model_before"] == c["model_after"],
                  "model mismatch")
    sp, cp = s["protocol"], c["protocol"]
    costs.require(sp["common_node_caps"] == cp["common_node_caps"]
                  and sp["boundaries"] == cp["boundaries"] == [0, 5, 10, 15]
                  and sp["packet_steps"] == cp["packet_steps"] == 5
                  and sp["terminal"] == cp["terminal"] == 20, "schedule/caps mismatch")
    st, ct = prepared["structured"]["source"]["tasks"][1], prepared["cem"]["source"]["tasks"][1]
    costs.require(all(st[k] == ct[k] for k in ("action_mean", "action_std", "projection"))
                  and pair.digest(st["goal"]) == s["cp10_goal_binding"]["goal_latent_sha256"],
                  "task normalization/projection/goal mismatch")


def verify_execution(arm, report, folder):
    row = report["rows"][0]
    nodes = report["nodes"][row["episode_id"]]
    events = pair.chain(folder / "journal.jsonl", "physical")
    observations = {e["receipt"]["step"]: e["receipt"] for e in events
                    if e["kind"] == "fresh_observation" and e["receipt"]["phase"] == "plan"}
    cem_events = None
    if arm == "cem":
        anchor.verify_anchor(folder / "full_cem_events.jsonl", report["cem_journal"])
        cem_events = [e["payload"]["event"] for e in pair.chain(folder / "full_cem_events.jsonl", "cem")]
    selections, previous = {}, None
    for node in nodes:
        if node["status"] == "not_started":
            continue
        b, receipt = node["boundary"], node["receipt"]
        if arm == "structured":
            previous = pair.structured_selection(node, previous, observations[b]["pixel_sha256"])
            value = receipt
        else:
            cem_audit.verify_cem_node(cem_events, node, b)
            value = receipt["commands"]
        selections[b] = receipt["commands"]
        seals = [e for e in events if e["kind"] == "boundary_sealed" and e["boundary"] == b]
        costs.require(len(seals) == 1 and seals[0]["plan_sha256"] == pair.digest(value), "boundary seal")
    costs.require(list(selections) == list(range(0, row["control_returned"], 5)), "execution roster")
    if arm == "structured":
        saved.audit(folder, ROSTER[arm][1])
    return pair.physical_check(report, arm, folder, selections)


def run():
    arms = {arm: costs.reconcile("root1_" + arm, meta) for arm, meta in ROSTER.items()}
    reports, prepared = {}, {}
    for arm, (directory, _) in ROSTER.items():
        folder = costs.RESULTS / directory
        reports[arm] = json.loads((folder / "report.json").read_bytes())
        prepared[arm] = json.loads(Path(reports[arm]["prepared"]).read_bytes())
        arms[arm]["execution_audit"] = verify_execution(arm, reports[arm], folder)
    verify_compatibility(reports, prepared)
    s, c = (arms[k]["execution_audit"] for k in ("structured", "cem"))
    costs.require(all(s[k] == c[k] for k in ("initial_state", "initial_pixel_sha256", "goal_state")),
                  "actual initial state/RGB/goal mismatch")
    return {"schema": "p1_total_cost_root1_pair_20261101_leibniz.v1", "arms": arms,
            "same_root_goal_model_initial_state_rgb_and_node_caps": True,
            "structured_prepare_after_cem_visible": True,
            "upstream_total_cost": None, "full_cost_certified": False,
            "fairness_proven": False, "independent_confirmation": False,
            "interpretation": "known-root exploratory, not prospective equal-compute comparison"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).absolute()
    costs.require(output.parent == costs.RESULTS and output.name.startswith("p1_total_cost_root1_pair_20261101_leibniz_")
                  and output.suffix == ".json", "exclusive output path")
    data = (json.dumps(run(), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with output.open("xb") as f:
        f.write(data)
    print(json.dumps({"path": str(output), "sha256": hashlib.sha256(data).hexdigest(),
                      "full_cost_certified": False}))


if __name__ == "__main__":
    main()
