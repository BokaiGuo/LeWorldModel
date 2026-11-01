"""Post-hoc saved-only matched-cp5 audit; never dispatches actions or inference."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SAME = RESULTS / "p1_diag32_samebank_tail_20261101_codex_run_CODEX01"
SAME_SHA = "f87f9c6c207076108c158c6c2335f05b2afa688eff9debacab793f92096da0d3"
OUT = RESULTS / "codex_p1_diag32_cp5_saved_paired_audit_20261101_exclusive_CODEX01.json"
IDENTITY = ("state", "snapshot", "pixel_sha256", "goal", "goal_rgb_sha256")


def f32(value):
    return np.asarray(value, dtype="<f4").reshape(-1, 2).tobytes()


def same_cp5(a, b):
    return all(a[k] == b[k] for k in IDENTITY)


def matches(hold_row, tail, candidate_row, schedule):
    if len(candidate_row["observations"]) <= 5:
        return False
    commands = schedule["commands"]
    return (len(commands) == 15
            and f32(hold_row["executed_raw"][:5]) == f32(commands[:5])
            and f32(tail) == f32(commands[5:])
            and same_cp5(hold_row["observations"][5], candidate_row["observations"][5]))


def outcome(row):
    obs = row["observations"][-1]
    state, goal = np.asarray(obs["state"]), np.asarray(obs["goal"])
    metrics = row["final_metrics"]
    n = row["control_returned"]
    return {"returned_steps": n, "unobserved_planned_steps": 15 - n,
            "native_success": metrics["native_success"],
            "strict_wrapped_success": metrics["wrapped_allxy_success"],
            "joint_xy_error": float(np.linalg.norm(state[:4] - goal[:4])),
            "native_angle_term": metrics["native_angle_term"],
            "strict_wrapped_angle_error": metrics["wrapped_object_angle_error"],
            "terminated": row["terminated"], "truncated": row["truncated"]}


def repeated(a, b):
    fields = ("state", "pixel_sha256", "goal", "goal_rgb_sha256", "snapshot")
    return (f32(a["executed_raw"]) == f32(b["executed_raw"])
            and len(a["observations"]) == len(b["observations"])
            and all(all(x[k] == y[k] for k in fields)
                    for x, y in zip(a["observations"], b["observations"], strict=True))
            and all(a[k] == b[k] for k in ("terminated", "truncated", "control_returned")))


def load(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    from . import codex_p1_diag32_cp5_hold_verify_20261101_exclusive as hold_v
    from . import codex_p1_diag32_cp5_live_verify_20261101_exclusive as live_v
    from . import codex_p1_diag32_fixed_physical_verify_20261101_exclusive as fixed_v
    from . import p1_diag32_samebank_saved_20261101_codex as same_v

    verified = {"hold": hold_v.verify(), "live": live_v.verify(),
                "fixed": fixed_v.verify(), "samebank": same_v.verify(SAME, SAME_SHA)}
    hold, live, fixed = (load(x.DIRECTORY / "report.json") for x in (hold_v, live_v, fixed_v))
    same, prepared, contract = load(SAME / "report.json"), load(SAME / "prepared.json"), load(hold_v.CONTRACT)
    live_rows = {r["role"]: r for r in live["rows"]}
    pairs = []
    for row in hold["rows"]:
        role = row["role"]
        tail = contract["hold_tails"][role]["raw"]
        found = [i for i, (r, s) in enumerate(zip(same["rows"], prepared["schedule"], strict=True))
                 if matches(row, tail, r, s)]
        assert len(found) <= 1, "ambiguous historical match"
        assert repeated(row, live_rows[role]), f"live replay mismatch: {role}"
        comp = row["comparison"]
        pair = {"role": role, "cp5_pixel_sha256": row["observations"][5]["pixel_sha256"],
                "hold_tail_sha256": comp["hold_sha256"],
                "optimized_tail_sha256": row["suffix_sha256"],
                "model_q2_hold": comp["hold_q2"], "model_q2_optimized": comp["optimized_q2"],
                "chosen": comp["choice"], "optimized_physical": outcome(row),
                "optimized_live_historical_repeat_checked": True,
                "hold_samebank_branch": found[0] if found else None,
                "hold_physical": outcome(same["rows"][found[0]]) if found else None,
                "hold_physical_missing": None if found else "not found among six audited branches"}
        pairs.append(pair)
    assert [p["hold_samebank_branch"] for p in pairs] == [0, 3, None], "unexpected pairing"
    assert repeated(fixed["row"], same["rows"][0]), "selected hold repeat mismatch"
    sources = {"hold": hold_v.DIRECTORY / "report.json", "live": live_v.DIRECTORY / "report.json",
               "fixed": fixed_v.DIRECTORY / "report.json", "samebank": SAME / "report.json",
               "samebank_prepared": SAME / "prepared.json", "hold_contract": hold_v.CONTRACT}
    return {"schema": "saved_cp5_paired_audit_v1", "source_files": {
                k: {"path": str(v), "sha256": sha(v)} for k, v in sources.items()},
            "audit_source_sha256": sha(Path(__file__)), "verification": verified, "pairs": pairs,
            "selected_hold_fixed_and_samebank_repeat_checked": True,
            "repeat_checked_fields": ["float32 executed actions", "all observation state/snapshot/pixel/goal/goal_rgb",
                                      "returned count", "terminated", "truncated"],
            "resources": {"hold_run_physical": hold["counts"], "hold_run_neural_by_role":
                          {r["role"]: r["neural"] for r in hold["rows"]},
                          "live_run_physical": live["counts"],
                          "live_run_neural_by_role": {r["role"]: r["neural"] for r in live["rows"]},
                          "samebank_physical": same["counts"], "samebank_neural": same["neural_calls"],
                          "fixed_physical": fixed["counts"], "fixed_neural": fixed["neural_calls"],
                          "new_neural_calls": 0, "new_physical_calls": 0,
                          "prior_generation_and_source_costs": "not fully accounted or equalized"},
            "limitations": ["post-hoc known outcomes, shared development root, not independent confirmation",
                            "historical repeats incur real costs but are not independent samples",
                            "model q2 is latent MSE, not a calibrated physical success probability",
                            "saved score and ledger verification only; no inference replay",
                            "native stop censors only unreturned planned steps",
                            "no selector modification, no fair total cost claim, no P1 admission"]}


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    write_new(OUT, build())
    print(json.dumps({"output": str(OUT), "sha256": sha(OUT)}))
