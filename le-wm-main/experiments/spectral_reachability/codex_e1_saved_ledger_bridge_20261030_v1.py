"""Read pinned E1 receipts into incomplete ledger inputs, without producing runs."""
import base64
import hashlib
import json
from pathlib import Path
import sys

from . import codex_e1_end_to_end_ledger_interface_20261030_v1 as ledger

ROOT = Path(__file__).resolve().parents[2] / "results"
SOURCES = {
    "targets": ("agent_e1_six_slot_common_prepare_20261029_v1_prepare_MAIN01.json",
                "cd96f6e28b25d3b7e3b6649c760b2f111747aec5c0ce9c8a41741318209df8a0"),
    "prepare": ("agent_e1_six_slot_common_cuda_20261029_v2_prepare_MAIN01.json",
                "df548a6a9d8356f654e6aac823e7dd90383570da56631c38155b539bad8ae0dc"),
    "report": ("agent_e1_six_slot_common_cuda_20261029_v2_cuda_MAIN01/report.json",
               "cf3cb2a506ec1739ca53f58faa00bb07972c6e546cd75e02e0c609dccd7b7578"),
    "controller": ("agent_e1_six_slot_common_cuda_20261029_v2_consume_MAIN01.json",
                   "1ca8e949cc4b219b9b03adb3d34ee859e924e0078f9f81c519be2c03ed77be3d"),
    "hi": ("agent_hilewm_rgb_high_low_20261028_v1_consume_MAIN01.json",
           "40f28414682c9540e9e195b361a36dfc514f816358adbeacb379fedc460e4ef0"),
}
ARMS = ("arbitrary_latent_waypoint", "existing_learned_prior", "dynamics_compiled_witness")
MAP = {"predict_API": "predict_api_calls", "action_encoder": "action_encoder_calls",
       "pred_proj": "pred_proj_calls"}


def need(value, msg):
    if not value:
        raise ValueError(msg)


def load_sources():
    out = {}
    for key, (path, expected) in SOURCES.items():
        raw = (ROOT / path).read_bytes()
        need(hashlib.sha256(raw).hexdigest() == expected, "source digest mismatch: " + key)
        out[key] = ledger.parse(raw)
    return out


def count(value):
    need(type(value) is dict and set(value) == {"attempted", "completed"}, "count fields")
    a, c = value["attempted"], value["completed"]
    need(type(a) is int and type(c) is int and 0 <= c <= a, "count range")
    return {"attempted": a, "completed": c}


def build(sources):
    t, p, r, c, h = (sources[k] for k in ("targets", "prepare", "report", "controller", "hi"))
    need(t["schema"] == "agent_e1_six_slot_common_prepare_20261029_v1.prepare" and
         p["schema"] == "agent_e1_six_slot_common_cuda_20261029_v2.prepare" and
         r["schema"] == "agent_e1_six_slot_common_cuda_20261029_v2.report" and
         c["schema"] == "agent_e1_six_slot_common_cuda_20261029_v2.consume" and
         h["schema"] == "agent_hilewm_rgb_high_low_20261028_v1.consume", "schema mismatch")
    need(p["six_sha256"] == SOURCES["targets"][1] and p["targets_sha256"] == t["target_seal_sha256"]
         and r["prepare_sha256"] == SOURCES["prepare"][1]
         and c["report_sha256"] == SOURCES["report"][1], "broken source chain")
    need(r["status"] == "all_six_sealed" and c["status"] == "all_six_saved_controller_and_replays_verified"
         and c["scientific_admission"] is False, "incomplete source")
    need(len(t["rows"]) == len(r["rows"]) == len(c["rows"]) == 6, "six slots required")
    slots = []
    for i, (tr, rr, cr) in enumerate(zip(t["rows"], r["rows"], c["rows"], strict=True)):
        root, arm = divmod(i, 3)[0], ARMS[i % 3]
        need(type(tr["root_slot"]) is int and type(rr["root"]) is int and type(cr["root"]) is int
             and tr["root_slot"] == rr["root"] == cr["root"] == root and
             tr["arm"] == rr["arm"] == cr["arm"] == arm, "slot identity mismatch")
        need(cr["status"] == "sealed" and cr["selection_sha256"] == rr["selection_sha256"],
             "selection identity mismatch")
        target = base64.b64decode(tr["target_f32_le_base64"], validate=True)
        need(len(target) == 768 and hashlib.sha256(target).hexdigest() == tr["target_data_sha256"],
             "target bytes mismatch")
        need(tr["origin"]["data_sha256"] == tr["target_data_sha256"], "target origin mismatch")
        phases = {phase: {"scope": "unknown", "receipt_sha256": None,
                          "measurements": {axis: None for axis in ledger.AXES}}
                  for phase in ledger.PHASES}
        projections = {}
        for phase in ("controller", "replay"):
            phases[phase]["scope"] = "per_arm"
            phases[phase]["receipt_sha256"] = SOURCES["controller"][1]
            resource = cr[phase + "_resources"]
            need(set(resource) == {"predict_API", "predictor", "action_encoder", "pred_proj"},
                 "controller axes")
            projections[phase] = {dest: count(resource[src]) for src, dest in MAP.items()}
            projections[phase]["predictor_samples"] = count(cr[phase + "_samples"])
            # predictor module calls are not predictor samples.
            projections[phase]["predictor_module_calls"] = count(resource["predictor"])
        if arm == "existing_learned_prior":
            hr = h["rows"][0 if root == 0 else 2]
            need(hr["root_slot"] == root and hr["condition"] == "off" and
                 hr["RGB_source"]["blind_sha256"] == tr["blind_sha256"] and
                 hr["high_target_sha256"] == hr["low_goal_sha256"] == tr["target_data_sha256"]
                 and hr["target_delivered_exact"] is True, "Hi target identity mismatch")
            phases["inherited"]["scope"] = "shared_unallocated"
            phases["inherited"]["receipt_sha256"] = SOURCES["hi"][1]
        data = {"schema": ledger.SCHEMA, "scope_sha256": SOURCES["prepare"][1], "arm": arm,
                "task_sha256": tr["task_sha256"], "target_sha256": tr["target_data_sha256"],
                "phases": phases}
        check = ledger.inspect(data)
        need(check["structural_status"] == "incomplete", "incomplete ledger required")
        slots.append({"root": root, "arm": arm, "ledger_input": data, "ledger_preflight": check,
                      "receipt_counts_not_event_sets": projections})
    modules = h["resources"]["modules"]
    hi_summary = {}
    for key in ("encoder", "high_cost", "high_predictor", "high_selected_rollout", "low_predictor"):
        value = modules[key]
        attempted, returned, rows = value["attempted"], value["returned"], value["input_rows"]
        need(all(type(v) is int and v >= 0 for v in (attempted, returned, rows)) and returned <= attempted,
             "Hi resource count")
        hi_summary[key] = {"attempted": attempted, "returned": returned, "input_rows": rows}
    return {"schema": "codex.e1.saved_ledger_bridge.v1", "status": "blocked_incomplete",
            "source_sha256": {k: v[1] for k, v in SOURCES.items()},
            "slots": slots, "hi_four_slot_shared_unallocated": hi_summary,
            "event_sets_verified": False, "coverage_verified": False,
            "equal_budget_certified": False, "ranking_authorized": False,
            "scientific_admission": False,
            "boundary": "pinned saved receipt declarations; no upstream strict reconstruction or production"}


def main():
    try:
        result = build(load_sources())
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({"status": "blocked_invalid_source", "reason": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
