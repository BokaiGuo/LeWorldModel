"""Paired known-root CEM/structured cost receipts; no fairness inference."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
HASHES = {
    "p1_first_feedback_carry_20261101_main_root0_RUN_MAIN02/report.json":
        "7aaa901a28974c09bd0cb12fb4b7a7efe939a533dbe2649327d9c33802fb5d9d",
    "p1_first_feedback_carry_20261101_main_root1_RUN_MAIN02/report.json":
        "4d4f4f074a1b3cdc0b9999c09c71414d78cd56825c067c139e1446f73e19aab2",
    "codex_p1_full_cem_host_20261101_exclusive_root0_RUN_MAIN01/report.json":
        "a973757107da8b98aed242cfb01d5a9079115150d0871c26fe255cc7b960e356",
    "codex_p1_full_cem_host_20261101_exclusive_root1_RUN_MAIN01/report.json":
        "0149183cf66149344f28eb3689f80d08149538bc8ad4143be78346c30b8efeb2",
}
AXES = ("predict_calls", "predict_samples", "action_encode_calls", "backward_calls",
        "vjp_rows", "jacobian_calls", "svd_calls", "replay_calls", "selection_evaluations")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def one(report, slot, arm):
    episode = f"s{slot}_{'full_cem' if arm == 'cem' else 'structured'}"
    rows = [r for r in report["rows"] if r["episode_id"] == episode]
    require(len(rows) == 1, "episode roster")
    row = rows[0]
    nodes = report["nodes"][episode]
    totals = dict.fromkeys(AXES, 0)
    observed = []
    unstarted = False
    for node in nodes:
        if node["status"] == "not_started":
            unstarted = True
            continue
        require(node["status"] == "sealed", "incomplete node")
        require(not unstarted, "sealed node after unstarted node")
        receipt = node["receipt"]
        account = receipt["account"]
        require(account["status"] == "closed_unverified", "account state")
        charge = account["charged_upper_bounds"]
        require(all(type(charge[k]) is int and charge[k] >= 0 for k in AXES), "charge")
        if arm == "structured":
            hooks = receipt["module_hook_audit"]["module_forward_pre_hooks"]
            require(all(hooks[k] == charge[k] for k in ("predict_calls", "action_encode_calls")),
                    "structured hook mismatch")
        for key in AXES:
            totals[key] += charge[key]
        observed.append(node["boundary"])
    require(observed == list(range(0, row["control_returned"], 5)), "feedback chronology")
    return dict(arm=arm, success=row["native_success"], control_steps=row["control_returned"],
                node_boundaries=observed, charges=totals,
                ledger_scope="dispatch upper bounds; not independently metered GPU compute",
                selection_accounting=("CEM candidate replay and selection are not charged in this ledger"
                                      if arm == "cem" else "structured replay and selection charged"),
                visual_encoder_calls=len(row["encodings"]))


def compare(structured, cem, slot):
    require(structured["cp10_goal_binding"] == cem["cp10_goal_binding"], "root/goal mismatch")
    require(structured["model_before"] == structured["model_after"] ==
            cem["model_before"] == cem["model_after"], "model identity")
    return dict(root=slot, goal_binding=cem["cp10_goal_binding"],
                methods=[one(structured, slot, "structured"), one(cem, slot, "cem")],
                independent_confirmation=False, equal_total_compute=False,
                caveat="observed development task; inherited CEM generic protocol fields are incorrect")


def main():
    reports = {}
    for relative, expected in HASHES.items():
        data = (RESULTS / relative).read_bytes()
        require(hashlib.sha256(data).hexdigest() == expected, "source digest")
        reports[relative] = json.loads(data)
    rows = []
    for slot in (0, 1):
        old = f"p1_first_feedback_carry_20261101_main_root{slot}_RUN_MAIN02/report.json"
        cem = f"codex_p1_full_cem_host_20261101_exclusive_root{slot}_RUN_MAIN01/report.json"
        rows.append(compare(reports[old], reports[cem], slot))
    print(json.dumps({"rows": rows, "sources": HASHES, "neural_calls": 0,
                      "physical_execution": False}, sort_keys=True))


if __name__ == "__main__":
    main()
