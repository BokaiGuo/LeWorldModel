"""Read one pinned E1 saved panel; never rank unequal or unknown budgets."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "results/agent_e1_forced_candidate_paired_summary_20261029_v1_MAIN01.json"
PANEL_SHA = "9d15df249e17ecc35a99ad57db38ef6fc581ca0f1f7e5e7a40bdf406a6431efd"
CONTROLLER = ROOT / "results/agent_e1_same_target_host_cuda_20261029_v1_consume_MAIN01.json"
CONTROLLER_SHA = "f385e1de76ca391a7ef4d03c198211cd303eb2e4d2bc3bdd69537146e8330eba"
ARMS = ("bank_winner", "zero_extra", "witness_extra")
STEPS = (0, 5, 10)
AXES = ("proposal", "controller", "replay", "encoder", "native", "other")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def load_pinned(path, expected):
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == expected, "pinned source bytes changed")
    def pairs(items):
        obj = {}
        for key, value in items:
            need(key not in obj, "duplicate key")
            obj[key] = value
        return obj
    def reject(_):
        raise ValueError("nonfinite JSON")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)


def same_budget(a, b):
    """Structural check only; an external verifier must authenticate the ledgers."""
    for ledger in (a, b):
        need(type(ledger) is dict and set(ledger) == set(AXES), "complete budget axes required")
        for value in ledger.values():
            need(type(value) is dict and set(value) == {"attempted", "completed"},
                 "budget counts required")
            attempted, completed = value["attempted"], value["completed"]
            need(type(attempted) is int and type(completed) is int and
                 0 <= completed <= attempted, "invalid budget counts")
    need(a == b, "different budget")


def check_comparable(a, b):
    """Validate declarations; this does not authenticate evidence or authorize ranking."""
    keys = {"root", "step", "task_sha256", "target_sha256", "metric", "budget"}
    for item in (a, b):
        need(type(item) is dict and set(item) == keys, "comparison metadata fields")
        for key in ("task_sha256", "target_sha256"):
            value = item[key]
            need(type(value) is str and len(value) == 64 and
                 all(c in "0123456789abcdef" for c in value), "invalid identity")
        need(type(item["root"]) is int and type(item["step"]) is int and
             type(item["metric"]) is str and bool(item["metric"]), "invalid comparison identity")
    for key in ("root", "step", "task_sha256", "target_sha256", "metric"):
        need(a[key] == b[key], "different comparison " + key)
    same_budget(a["budget"], b["budget"])


def analyze(panel, controller):
    need(panel["schema"] == "agent_e1_forced_candidate_paired_summary_20261029_v1.analysis",
         "panel schema")
    need(panel["receipt_sha256"] == "519431b8db8541626ec5bd8e933280b053752f01ef955ed4c021d35575464da3" and
         panel["report_sha256"] == "31736c26f5242dc4be38104ad5f51ef61ee7e3cfb88a96b85e7c359614ea69de",
         "physical receipt binding")
    result = panel["result"]
    need(result["same_target"] is True and result["scientific_admission"] is False and
         result["equal_total_compute"] is False and result["flops"] is None, "scope mismatch")
    need(result["known_roots"] == 2 and result["fresh_roots"] == 0 and
         result["slot_denominator"] == 6 and result["checkpoint_denominator"] == 18 and
         result["pair_denominator"] == 18, "denominator mismatch")
    need(controller["schema"] == "agent_e1_same_target_host_cuda_20261029_v1.consume" and
         controller["status"] == "four_saved_controller_replay_verified" and
         controller["scientific_admission"] is False and controller["per_axis"] == 600 and
         controller["verification"]["verified"] is True and
         controller["verification"]["slots"] == 4 and
         controller["report_sha256"] == "92a8c6d5c9a6fa60bb1738cfd5ed9abd70d5cc889ada04d62886b14c1ca5f3b7",
         "controller receipt mismatch")
    cells = result["cells"]
    need(type(cells) is list and len(cells) == 18, "checkpoint grid length")
    expected = {(r, a, s) for r in (0, 1) for a in ARMS for s in STEPS}
    seen, goals, rows = set(), {}, []
    for cell in cells:
        key = (cell["root"], cell["arm"], cell["step"])
        need(type(key[0]) is int and type(key[2]) is int and key in expected and key not in seen,
             "invalid checkpoint grid")
        seen.add(key)
        need(cell["status"] in ("observed", "missing"), "invalid status")
        if cell["status"] == "observed":
            need(cell["missing_reason"] is None and type(cell["value"]) is dict, "invalid observed cell")
            goal = cell["value"]["goal"]
            need(type(goal) is list and len(goal) == 7 and
                 all(type(v) in (int, float) and math.isfinite(v) for v in goal), "invalid goal")
            goal_bytes = json.dumps(goal, separators=(",", ":"), allow_nan=False)
            need(key[0] not in goals or goals[key[0]] == goal_bytes, "different native goals")
            goals[key[0]] = goal_bytes
        else:
            need(cell["value"] is None and bool(cell["missing_reason"]), "missing cell requires reason")
        rows.append({"root": key[0], "arm": key[1], "step": key[2], "status": cell["status"],
                     "missing_reason": cell["missing_reason"], "saved_value": cell["value"]})
    need(seen == expected, "incomplete checkpoint grid")
    need(type(result["pairs"]) is list and len(result["pairs"]) == 18, "pair denominator")
    return {
        "schema": "codex.e1.strict_saved_consumer.v1",
        "status": "blocked_equal_budget_comparison",
        "panel_sha256": PANEL_SHA, "controller_receipt_sha256": CONTROLLER_SHA,
        "rows": rows,
        "controller_counts": {"per_slot_per_axis": {"controller": 144, "extra": 2, "replay": 4},
                              "four_slots_per_axis": 600,
                              "note": "nested axes are not additive; inherited receipt"},
        "physical_panel_counts": result["physical_resources"],
        "unresolved": ["per_arm_proposal_and_inherited_costs", "per_arm_end_to_end_ledger",
                       "flops", "information_parity"],
        "rankings": None, "equal_total_budget": "unknown",
        "scientific_admission": False, "fresh_roots": 0,
        "trust": "pinned saved summaries and receipt; no upstream fullstrict or physical rerun",
    }


def main():
    try:
        result = analyze(load_pinned(PANEL, PANEL_SHA), load_pinned(CONTROLLER, CONTROLLER_SHA))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "blocked_invalid_input", "reason": str(exc)}))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
