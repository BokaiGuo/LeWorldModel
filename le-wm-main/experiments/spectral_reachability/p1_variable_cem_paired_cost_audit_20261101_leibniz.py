"""Read-only paired audit of known-root structured and variable CEM receipts."""
import hashlib
import json

from . import codex_p1_variable_cem_root0_audit_20261101_exclusive as saved
from . import p1_full_cem_cost_compare_20261101_main as prior

BOUNDARIES = (0, 5, 10, 15)
CAP = 195


def paired(structured, cem, root):
    prior.require(root in (0, 1), "root")
    # Reuse the earlier audit's binding, chronology and structured hook checks.
    compared = prior.compare(structured, cem, root)
    arms = []
    for arm, report, episode in (
        ("structured", structured, f"s{root}_structured"),
        ("variable_cem", cem, f"s{root}_full_cem"),
    ):
        nodes = report["nodes"][episode]
        prior.require([node["boundary"] for node in nodes] == list(BOUNDARIES), "node roster")
        entries = []
        for node in nodes:
            status = node["status"]
            prior.require(status in ("sealed", "not_started"), "node status")
            if status == "not_started":
                entries.append({"boundary": node["boundary"], "status": status, "charge": None})
                continue
            charge = node["receipt"]["account"]["charged_upper_bounds"]
            prior.require(charge["predict_calls"] <= CAP and charge["action_encode_calls"] <= CAP
                          and charge["predict_samples"] <= CAP, "observed forward cap")
            # CEM does perform selection and replay; its ledger does not charge them.
            vector = {key: charge[key] for key in prior.AXES}
            if arm == "variable_cem":
                vector["replay_calls"] = vector["selection_evaluations"] = None
            entries.append({"boundary": node["boundary"], "status": status, "charge": vector})
        summary = compared["methods"][0 if arm == "structured" else 1]
        arms.append({"arm": arm, "success": summary["success"], "control_steps": summary["control_steps"],
                     "visual_encoder_calls": summary["visual_encoder_calls"], "nodes": entries})
    return {"root": root, "goal_binding": compared["goal_binding"], "arms": arms,
            "cap_scope": "observed node dispatch upper bounds only; not prospective common enforcement",
            "equal_total_compute": False, "independent_confirmation": False}


def audit(root):
    result = saved.audit(root)
    cem, _, _, _ = saved.load_sources(root)
    name = f"p1_first_feedback_carry_20261101_main_root{root}_RUN_MAIN02/report.json"
    data = (prior.RESULTS / name).read_bytes()
    prior.require(hashlib.sha256(data).hexdigest() == prior.HASHES[name], "structured report SHA")
    structured = json.loads(data)
    pair = paired(structured, cem, root)
    prior.require(pair["arms"][1]["control_steps"] == result["steps"]
                  and pair["arms"][1]["success"] == result["native_success"], "saved audit mismatch")
    pair["sources"] = {"structured_report_sha256": prior.HASHES[name],
                       "variable_cem_report_sha256": result["report_sha256"]}
    pair["variable_cem_saved_audit"] = result
    return pair


def main():
    print(json.dumps({"pairs": [audit(0), audit(1)], "neural_calls": 0,
                      "physical_execution": False,
                      "limitations": ["known development roots", "dispatch upper bounds, not measured compute",
                                      "CEM event chain not anchored by historical report",
                                      "structured report and hooks checked, not full physical replay",
                                      "shared and model loading costs excluded"]},
                     sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
