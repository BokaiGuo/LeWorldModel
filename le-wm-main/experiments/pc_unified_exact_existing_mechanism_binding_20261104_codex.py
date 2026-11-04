"""Bind a unified exact decision to its previously sealed mechanism diagnostics."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
UNIFIED = RESULTS / "pc_unified_multidirection_interface_verify_20261103_codex.json"
PLANS = RESULTS / "pc_multidirection_successor_exact_dev_20261103_codex_plans.json"
DIAG = RESULTS / "pc_multidirection_successor_exact_dev_20261103_codex_same_chain_diagnostic.json"
FEEDBACK = RESULTS / "pc_multidirection_equal_feedback_mechanism_diag_20261103_codex_plans.json"
FEEDBACK_AUDIT = RESULTS / "pc_multidirection_equal_feedback_mechanism_diag_20261103_codex_audit.json"
OUTPUT = RESULTS / "pc_unified_exact_existing_mechanism_binding_20261104_codex.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique(rows, **selection):
    matches = [row for row in rows if all(row.get(key) == value for key, value in selection.items())]
    if len(matches) != 1:
        raise ValueError(f"non-unique row: {selection}")
    return matches[0]


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    unified, plans, diagnostic, feedback, audit = [json.loads(path.read_text())
                                                    for path in (UNIFIED, PLANS, DIAG, FEEDBACK, FEEDBACK_AUDIT)]
    for data in (unified, plans, diagnostic, feedback):
        for path, digest in data["dependencies"].items():
            if sha(path) != digest:
                raise ValueError(f"dependency mismatch: {path}")
    if audit["plans_sha256"] != sha(FEEDBACK):
        raise ValueError("feedback audit binding mismatch")
    step = unique(unified["exact_feedback"], step=0)
    plan, diag = unique(plans["rows"], root=0), unique(diagnostic["rows"], root=0)
    chain = step["selected"]["chain"]
    if (unified["exact_task"] != plan["task"] or plan["task"] != diag["task"]
            or chain != plan["result"]["selected"]["chain"] or chain != diag["chain"]):
        raise ValueError("exact task or selected chain mismatch")
    controls = diag["controls"]
    rows = {name: unique(feedback["rows"], root=0, name=name)
            for name in ("selected", "zero", "greedy", "random_bridge", "lookahead")}
    for name, row in rows.items():
        if row["task"] != diag["task"] or row["prefix"] != controls[name]["prefix"]:
            raise ValueError(f"feedback control mismatch: {name}")
    result = {
        "scope": "retrospective binding only; no new planning or interventions",
        "source_sha256": {str(path): sha(path) for path in
                          (UNIFIED, PLANS, DIAG, FEEDBACK, FEEDBACK_AUDIT, Path(__file__))},
        "root": 0, "task": diag["task"], "selected_chain_exact_match": True,
        "selected_prefix_equals_greedy": controls["selected"]["prefix"] == controls["greedy"]["prefix"],
        "nonprogress": diag["nonprogress"], "initial_distance": diag["initial_distance"],
        "prefix_distances": {name: value["prefix_distance"] for name, value in controls.items()},
        "response_difference_frobenius": diag["response_difference_frobenius"],
        "selected_common_tail_distances": {name: value["common_scores"][-1] for name, value in controls.items()},
        "selected_common_tail_changes": {name: value["active_changes"][-1] for name, value in controls.items()},
        "equal_open_loop_replan": {name: value["own_score"] for name, value in controls.items()},
        "equal_feedback_replan": {name: value["final_distance"] for name, value in rows.items()},
        "base_intersection": diag["base_intersection"],
        "all_controls_intersection": diag["all_controls_intersection"],
        "limitations": ["Mechanism diagnostic was specified after original planning and uses exposed tasks.",
                        "The selected tail in the common-suffix family was selected during original planning.",
                        "No independent tasks, physical RGB mechanism, or strong-baseline confirmation."],
    }
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
