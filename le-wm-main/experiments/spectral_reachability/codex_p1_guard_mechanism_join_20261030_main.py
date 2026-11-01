"""Join paired mechanism interventions with the physical RGB-guard feedback panel."""
import argparse
import json
from pathlib import Path

from . import codex_p1_complete_paired_view_20261030_main as existing

ROOT = existing.base.ROOT
NAME = "codex_p1_guard_mechanism_join_20261030_main"
VIEW = ROOT / "results/codex_p1_multistart_paired_view_20261030_main_RUN01.json"
PREP = ROOT / "results/codex_p1_rgb_shape_guard_20261030_main_prepare_RUN01.json"
SAVED = ROOT / "results/codex_p1_rgb_shape_guard_20261030_main_saved_RUN01.json"
PAIRED = ROOT / "results/codex_p1_rgb_pusher_ablation_compare_20261030_main_GUARD01.json"
SHAS = ("d378b83c4cbc7c2434b44bad901c7eb186da147acdc0c8f5cd1a9dfc1045b551",
        "fae5162bad2b3dcb5bbde184bf88932b957a2f0e17aa4083f1723364e6750d1f",
        "52761b20aa2fde1370f6307ba389c719fb4a848b68b4d8759047ea5c82c77900",
        "d671d51892cf96540192b6e10b7433c01b67e316efc69e1febde4c98a2c8561f")
FIELDS = ("object_position_error", "object_angle_error_wrapped",
          "pusher_position_error", "native_joint_xy_error")
IDS = ("index", "logical_pair", "root_slot", "condition", "arm", "role")


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def delta(a, b):
    if a is None or b is None:
        return None
    return {k: a[k] - b[k] for k in FIELDS}


def build(view, prep, saved, paired):
    need(all(len(x) == 24 for x in (view["rows"], prep["roster"], saved["rows"], paired["rows"])),
         "full fixed denominator")
    rows = []
    for i, (v, p, s, q) in enumerate(zip(view["rows"], prep["roster"], saved["rows"],
                                          paired["rows"], strict=True)):
        identity = v["identity"]
        need(identity["index"] == i and all(identity[k] == p[k] == s[k] == q["identity"][k]
                                             for k in IDS), "prefix identity")
        need(identity["task_id"] == p["task_id"]
             and identity["physical_root_sha256"] == p["physical_root_sha256"]
             and identity["prefix5_float32_hex"] == p["prefix5_float32_hex"]
             and identity["cp5_file_sha256"] == p["checkpoint"]["file_sha256"]
             and identity["cp5_pixel_sha256"] == p["checkpoint"]["pixel_sha256"]
             and identity["cp5_state"] == p["checkpoint"]["state"], "prefix source binding")
        need(q["common_first5_and_rgb"] and q["new_status"] == s["status"]
             and q["new_decision"] == s["rgb_decision"]
             and q["new_steps"] == s["accepted_suffix_steps"]
             and q["new_endpoint"] == s["endpoint_metrics"], "feedback match")
        steps, endpoint = s["accepted_suffix_steps"], s["endpoint_metrics"]
        if s["status"] == "rgb_rejected":
            need(steps == 5 and endpoint is None and s["rgb_decision"] == "rgb_rejected", "rejected row")
        elif s["status"] == "native_stopped":
            need(steps <= 10 and ((endpoint is not None) == (steps == 10)), "native stop/censor")
        else:
            need(s["status"] == "probe_completed" and steps == 10 and endpoint is not None, "complete row")
        item = {"status": s["status"], "decision": s["rgb_decision"], "accepted_steps": steps,
                "horizon_metrics": endpoint, "last_observed_metrics": s["last_observed_metrics"],
                "physical_resources": s["physical_resources"], "stage_resources": s["stage_resources"],
                "delta_vs_original_feedback": q["new_minus_old"],
                "same_first5_and_midpoint_rgb": True}
        rows.append({**v, "rgb_guard_feedback": item})
    groups = []
    for g in range(8):
        triplet = rows[g*3:g*3+3]
        need([r["identity"]["role"] for r in triplet] == ["selected", "zero", "greedy"]
             and all(r["identity"]["logical_pair"] == g for r in triplet), "triplet roles")
        s, z, greedy = [r["rgb_guard_feedback"] for r in triplet]
        groups.append({"logical_pair": g, "root_slot": triplet[0]["identity"]["root_slot"],
                       "condition": triplet[0]["identity"]["condition"],
                       "selected_status": s["status"], "zero_status": z["status"],
                       "greedy_status": greedy["status"],
                       "selected_minus_zero": delta(s["horizon_metrics"], z["horizon_metrics"]),
                       "selected_minus_greedy": delta(s["horizon_metrics"], greedy["horizon_metrics"]),
                       "success": {role: item["horizon_metrics"]["native_success"]
                                   if item["horizon_metrics"] is not None else None
                                   for role, item in zip(("selected", "zero", "greedy"), (s, z, greedy), strict=True)}})
    return {"schema": NAME, "status": "descriptive_join", "rows": rows, "groups": groups,
            "sources": {str(p): h for p, h in zip((VIEW, PREP, SAVED, PAIRED), SHAS, strict=True)},
            "limitations": ["two known roots, repeated trajectories, post-outcome policy",
                            "common suffix effect is not independently optimized capability",
                            "refreshed method is RGB heuristic, not proven D2P mechanism",
                            "no independent confirmation or total budget fairness"],
            "P1_admission": False, "independent_confirmation": False, "equal_total_budget_certified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_")
         and not out.exists() and not out.is_symlink(), "exclusive output")
    inputs = [existing.gap.read(p, sha)[0] for p, sha in zip((VIEW, PREP, SAVED, PAIRED), SHAS, strict=True)]
    result = build(*inputs)
    result["code_sha256"] = existing.base.sha(Path(__file__).resolve())
    print(json.dumps({"sha256": existing.base.write(out, result), "rows": len(result["rows"]),
                      "groups": result["groups"]}), flush=True)


if __name__ == "__main__":
    main()
