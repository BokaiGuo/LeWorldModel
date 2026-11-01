"""Audit saved beam/nominal-MPPI task and phase-level resource alignment."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = "codex_beam_mppi_protocol_budget_20261030"
BEAM = "matched_domain_H2_beam20x6"
MPPI = "kernel_faithful_mppi_nominal"
SOURCES = {
    "beam": ("results/agent_kinodynamic_matched_domain_20261028_v1_cuda_MAIN01/report.json", "d4976aeca357b5324c74be8b2b7e3bfc254e82b834b3acfe1b0ab90a1ef4614c"),
    "sampling": ("results/moving_prediction_matched_sampling_cuda_20261027_v1_cuda_MAIN01/report.json", "60872ad114700731d82d293a616a4dede54a53b7be6514ad0cdf231d838d3ed8"),
    "comparison": ("results/agent_baseline_matched_rgb_comparison_20261028_v1_MAIN01.json", "eb661bfd89b75b4865b37ba517de4412f015472928a9ed3c2b76ec15367ed128"),
}


def read_sources():
    result = {}
    for key, (path, expected) in SOURCES.items():
        raw = (ROOT / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("source SHA mismatch: " + key)
        result[key] = json.loads(raw)
    return result


def one(rows, root, arm, root_field):
    matches = [r for r in rows if r[root_field] == root and r["arm"] == arm]
    if len(matches) != 1:
        raise ValueError("missing or duplicate fixed slot")
    return matches[0]


def audit(data):
    beam, sampling, comparison = data["beam"], data["sampling"], data["comparison"]
    b, s = beam["contract"], sampling["contract"]
    if len(beam["rows"]) != 2 or len(sampling["rows"]) != 4 or len(comparison["rows"]) != 10:
        raise ValueError("fixed roster mismatch")
    if any(x["errors"] for x in (beam, sampling)) or not comparison["all_expected_same_identity"]:
        raise ValueError("incomplete upstream report")
    if beam["model_hash_before"] != beam["model_hash_after"] or sampling["model_hash_before"] != sampling["model_hash_after"] or beam["model_hash_before"] != sampling["model_hash_before"]:
        raise ValueError("model identity mismatch")
    if beam["inherited_encoding_once"] != sampling["inherited_encoding_once"]:
        raise ValueError("inherited encoding cohort mismatch")
    schedule = b["schedule"]
    beam_search = schedule["first20"] + schedule["second120"]
    beam_replay = schedule["fresh_whole_predict"] + schedule["fresh_segmented_predict"]
    mppi_search = s["iterations"] * s["samples"] * s["horizon"]
    mppi_replay = s["extra_replay"]
    if (beam_search, beam_replay, mppi_search, mppi_replay) != (140, 4, 144, 0):
        raise ValueError("phase schedule mismatch")
    if b["horizon"] != s["horizon"] or b["raw_steps"] != s["raw_steps"] or b["packet_logit_cap"] != s["cap"] or b["with_target"] != s["with_target"]:
        raise ValueError("task contract mismatch")
    if comparison["equal_total_compute"] is not False or comparison["equal_FLOPs"] is not False:
        raise ValueError("fairness boundary mismatch")
    out = []
    for root in (0, 1):
        rb = one(beam["rows"], root, BEAM, "source_slot")
        rm = one(sampling["rows"], root, MPPI, "source_slot")
        cb = one(comparison["rows"], root, BEAM, "root_slot")
        cm = one(comparison["rows"], root, MPPI, "root_slot")
        if rb["status"] != "model_plan_sealed" or rm["status"] != "model_plan_sealed":
            raise ValueError("unsealed row")
        if cb["identity"] != cm["identity"] or cb["identity_sha256"] != cm["identity_sha256"]:
            raise ValueError("task identity mismatch")
        identity = cb["identity"]
        if identity["horizon"] != b["horizon"] or identity["packet_steps"] != b["packet_steps"] or identity["with_target"] != b["with_target"]:
            raise ValueError("identity task protocol mismatch")
        for rr in (rb, rm):
            if rr["task_id"] != identity["task_id"] or rr["blind_sha256"] != identity["blind_sha256"]:
                raise ValueError("source task mismatch")
        if identity["model_tensor_sha256"] != beam["model_hash_before"] or identity["source_slot"] != root:
            raise ValueError("comparison model/root mismatch")
        if cb["domain"]["admissible"] != cm["domain"]["admissible"]:
            raise ValueError("admissible domain mismatch")
        if cb["domain"]["admissible"]["logit_cap"] != b["packet_logit_cap"]:
            raise ValueError("domain cap mismatch")
        for rr, cc in ((rb, cb), (rm, cm)):
            if rr["resources"] != cc["resources"]["original_names"]:
                raise ValueError("resource projection mismatch")
            for key in ("predict_API", "predictor"):
                if rr["resources"][key] != {"attempted": 144, "completed": 144}:
                    raise ValueError("forward counter mismatch")
            if rr["predictor_samples"] != {"attempted": 144, "completed": 144}:
                raise ValueError("predictor samples mismatch")
        out.append({
            "root_slot": root, "task_id": identity["task_id"], "task_sha256": identity["task_sha256"],
            "identity_sha256": cb["identity_sha256"], "common_horizon_packets": b["horizon"],
            "common_raw_steps": b["raw_steps"], "common_admissible": cb["domain"]["admissible"],
            "beam": {"recorded_resources": rb["resources"], "recorded_predictor_samples": rb["predictor_samples"],
                     "schedule_search_predict": beam_search, "schedule_replay_predict": beam_replay,
                     "support": cb["domain"]["search_support"], "selected_domain": cb["domain"]},
            "mppi": {"recorded_resources": rm["resources"], "recorded_predictor_samples": rm["predictor_samples"],
                     "schedule_search_predict": mppi_search, "schedule_replay_predict": mppi_replay,
                     "support": cm["domain"]["search_support"], "selected_domain": cm["domain"]},
            "total_predict_api_equal": True, "total_predict_samples_equal": True,
            "planning_predict_equal": False, "search_support_equal": False,
            "total_flops": None, "per_arm_wall_seconds": None,
        })
    return {
        "schema": NAME, "sources": {k: {"path": v[0], "sha256": v[1]} for k, v in SOURCES.items()},
        "rows": out, "scope": "known two roots; local bounded algorithms; not official MPPI or general kinodynamic search",
        "phase_accounting": "contract schedule decomposes observed 144 completed calls; this audit does not independently replay event ledgers",
        "inherited_encoding": beam["inherited_encoding_once"],
        "encoding_scope": "shared cohort; not allocated per arm; evaluation RGB/physical cohorts extra and not apportioned",
        "search_support_note": "same declared admissible ball does not imply same finite proposals or equal coverage",
        "fair_total_compute": False, "fair_ranking": False,
        "trust": "SHA-pinned reports and prior identity summary; upstream strict and raw event traces not rebuilt",
        "new_gpu_calls": 0, "new_neural_calls": 0, "new_physics_calls": 0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("build", "verify"))
    p.add_argument("--output", required=True)
    p.add_argument("--sha256")
    args = p.parse_args()
    path = Path(args.output)
    if not path.is_absolute() or path.parent != ROOT / "results" or not path.name.startswith(NAME + "_"):
        raise ValueError("invalid exclusive output")
    raw = (json.dumps(audit(read_sources()), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    if args.command == "build":
        with path.open("xb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
    elif args.sha256 != digest or path.read_bytes() != raw:
        raise ValueError("digest or reconstruction mismatch")
    print(json.dumps({"path": str(path), "sha256": digest, "command": args.command}))


if __name__ == "__main__":
    main()
