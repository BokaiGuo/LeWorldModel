"""Prospective permission contract on an already-observed panel; no inference."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STEM = "p5_observation_permission_contract_20261030_codex_v1"
PANEL = ROOT / "results/p5_crossmodel_panel_binding_20261030_codex_v1_MAIN01.json"
PANEL_SHA = "80ed10d3b8f623d361cc25e2764d390bc9cc758178c89e7467e99da3d785ef98"
BRIDGE = ROOT / "results/agent_p5_raw164_jepa_input_bridge_20261029_v1_prepare_MAIN01.json"
BRIDGE_SHA = "1e042999e6970261882324c8adfc45ef6303decd047e2ac7f01f88942488586f"
CHANNELS = ("rgb_history", "agent_position", "agent_velocity", "object_state", "goal", "successor")
ARMS = {
    "lewm_existing": {
        "observation_channels": ["rgb_history"],
        "goal_access": "goal_conditioned_lewm_task_binding",
        "model_input_history": "not_independently_verified_by_panel_binding",
        "source_evidence": "inherited_existing_model_task_binding",
    },
    "jepa_native_h1": {
        "observation_channels": ["rgb_history", "agent_position", "agent_velocity"],
        "goal_access": "none_in_predictor",
        "model_input_history": "global_5_10_15",
        "source_evidence": "saved_bridge_prepare_not_runtime_revalidation",
    },
}
CLAIMS = {
    "same_physical_panel_descriptive": "allowed_with_permission_disclosure_and_upstream_boundary",
    "rgb_only_jepa": "prohibited_native_jepa_has_position_velocity",
    "matched_observation_cross_model": "blocked_no_two_runtime_verified_same_permission_arms",
    "causal_effect_of_proprio": "blocked_no_same_model_matched_permission_intervention",
    "cross_model_raw_latent_mse_ranking": "prohibited_incommensurate_latent_spaces",
    "physical_control_benefit": "blocked_no_matched_control_evaluation",
    "p5_scientific_admission": "blocked",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path, expected, cap):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= cap, "bounded input")
    data = path.read_bytes()
    require(sha(data) == expected, f"SHA mismatch: {path}")
    def pairs(items):
        obj = {}
        for k, v in items:
            require(k not in obj, "duplicate key")
            obj[k] = v
        return obj
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_arm_declarations(arms):
    """Exact observation allowlist: unknown/absent channels and false RGB-only fail closed."""
    require(isinstance(arms, dict) and set(arms) == set(ARMS), "fixed arm set")
    for name, expected in ARMS.items():
        require(arms[name] == expected, f"arm permission declaration mismatch: {name}")
    return True


def validate_comparison(arms, claim):
    """Fail closed for unsupported comparisons, even if a caller relabels an arm."""
    validate_arm_declarations(arms)
    require(claim == "same_physical_panel_descriptive", "comparison not authorized by this contract")
    return {"claim": claim, "permission_disclosure_required": True,
            "same_observation_permissions": False, "scientific_admission": False}


def build(panel, bridge):
    require(panel["schema"] == "p5_crossmodel_panel_binding_20261030_codex_v1"
            and panel["status"] == "saved_input_physical_identity_bound_not_scientific_admission"
            and panel["branch_denominator"] == 164 and panel["pair_denominator"] == 80
            and panel["P1_admission"] is False and panel["P5_admission"] is False,
            "panel ABI")
    require(bridge["schema"] == "agent_p5_raw164_jepa_input_bridge_20261029_v1.prepare"
            and bridge["status"] == "saved_inputs_prepared_not_cuda_executed"
            and bridge["protocol"]["context_global_frames"] == [5, 10, 15]
            and bridge["protocol"]["successor_global_frame"] == 20, "bridge ABI")
    require(panel["input_sha256"]["bridge"] == BRIDGE_SHA, "bridge binding")
    require(panel["permissions"] == {
        "jepa": "history RGB plus agent position and velocity",
        "lewm": "RGB-only; model task binding inherited from physical receipt"}, "permission provenance")
    rows, requests = panel["rows"], bridge["requests"]
    require(len(rows) == len(requests) == 164, "164 denominator")
    annotated = []
    for i, (r, q) in enumerate(zip(rows, requests, strict=True)):
        slot = i // 41
        require((r["index"], r["root_slot"], r["condition"]) ==
                (i, slot // 2, ("off", "on")[slot % 2]), "fixed row schedule")
        require(all(r[k] == q[k] for k in ("index", "root_slot", "condition", "task_id", "selection_digest")),
                "request identity")
        require(q["context_global_frames"] == [5, 10, 15] and q["successor_global_frame"] == 20
                and q["images"] == r["history_rgb"] and q["label"] == r["successor_rgb_identity_only"],
                "history and label identity")
        require(len(q["images"]) == 3
                and all(valid_hash(x["file_sha256"]) and valid_hash(x["pixel_sha256"]) for x in q["images"])
                and valid_hash(r["raw_proprio_float64_sha256"])
                and valid_hash(r["normalized_proprio_float32_sha256"])
                and valid_hash(r["raw15_float32_sha256"])
                and valid_hash(r["normalized_packets_float32_sha256"]), "input identifiers")
        annotated.append({
            "index": i, "root_slot": r["root_slot"], "condition": r["condition"],
            "task_id": r["task_id"], "selection_digest": r["selection_digest"],
            "jepa_rgb_history_file_sha256": [x["file_sha256"] for x in q["images"]],
            "jepa_rgb_history_pixel_sha256": [x["pixel_sha256"] for x in q["images"]],
            "jepa_raw_agent_position_velocity_float64_sha256": r["raw_proprio_float64_sha256"],
            "jepa_normalized_agent_position_velocity_float32_sha256": r["normalized_proprio_float32_sha256"],
            "jepa_raw15_float32_sha256": r["raw15_float32_sha256"],
            "jepa_normalized_packets_float32_sha256": r["normalized_packets_float32_sha256"],
            "successor_file_sha256_label_only": q["label"]["file_sha256"],
            "lewm_task_binding": r["lewm_model_task_binding"],
            "lewm_exact_history_and_runtime_input_receipt": None,
        })
    require(panel["baseline_indices"] == [0, 41, 82, 123] and len(panel["pairs"]) == 80,
            "fixed baseline/pair denominator")
    for i, p in enumerate(panel["pairs"]):
        slot, local = divmod(i, 20)
        expected = (slot, slot // 2, ("off", "on")[slot % 2], local % 10,
                    (0.01, 0.001)[local // 10], 41 * slot + 1 + 2 * local, 41 * slot + 2 + 2 * local)
        require(tuple(p[k] for k in ("slot", "root_slot", "condition", "coord", "h", "minus_index", "plus_index"))
                == expected, "pair schedule")
    validate_arm_declarations(ARMS)
    validate_comparison(ARMS, "same_physical_panel_descriptive")
    return {
        "schema": STEM, "status": "prospective_contract_on_seen_panel_not_experiment_or_preregistration",
        "inputs": {"panel_sha256": PANEL_SHA, "jepa_bridge_sha256": BRIDGE_SHA},
        "denominator": {"branches": 164, "pairs": 80, "baselines": [0, 41, 82, 123],
                        "known_roots": 2, "certified_fresh_roots": 0},
        "arms": ARMS, "channel_vocabulary": list(CHANNELS), "rows": annotated,
        "pairs": panel["pairs"], "claim_policy": CLAIMS,
        "comparison_contract": {
            "current": "same saved physical branch only; different observation permissions and potentially different histories",
            "matched_rgb_only": "blocked; requires validated RGB-only JEPA-compatible implementation without position or velocity, plus both runtime input receipts and matched temporal/action/goal contracts",
            "within_model_permission_ablation": "not available; requires precommitted same-checkpoint intervention and runtime receipts, and disclosure if zeroing is out-of-distribution; zeroed proprio is not RGB-only",
            "goal": "LeWM goal-conditioned task and JEPA goal-free predictor are separate information channels; reconcile before interpreting matched task comparison",
            "future_target": "successor is evaluation label only; no predictor input or selector access",
            "metrics": "no cross-space raw latent MSE ranking; task comparison needs same independently defined native outcome and matched task/decision protocol",
            "resource": "record encoding/predictor calls and batch/frame counts per arm; no equal-compute claim without verified matching",
            "failure": "retain fixed 164/80 and all missing/partial statuses; no substitution, padding, outcome-based selection or retry",
        },
        "evidence_boundary": "This contract is new and downstream of observed outcomes; not a pre-registration or claim of new scientific evidence. SHA binds local bytes, not independent time attestation. No fullstrict rerun or loaded-runtime verification.",
        "new_neural_calls": 0, "new_physics_calls": 0, "training_calls": 0,
        "P1_admission": False, "P5_admission": False,
    }


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def main():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command", required=True)
    w = subs.add_parser("write")
    w.add_argument("--output", type=Path, required=True)
    v = subs.add_parser("verify")
    v.add_argument("--artifact", type=Path, required=True)
    v.add_argument("--sha256", required=True)
    check = subs.add_parser("check")
    check.add_argument("--artifact", type=Path, required=True)
    check.add_argument("--sha256", required=True)
    check.add_argument("--claim", required=True)
    args = parser.parse_args()
    panel = read(PANEL, PANEL_SHA, 8 * 1024 * 1024)
    bridge = read(BRIDGE, BRIDGE_SHA, 8 * 1024 * 1024)
    data = canonical(build(panel, bridge))
    require(sha(PANEL.read_bytes()) == PANEL_SHA and sha(BRIDGE.read_bytes()) == BRIDGE_SHA, "source drift")
    if args.command == "write":
        require(args.output.name.startswith(STEM + "_"), "exclusive output stem")
        with args.output.open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        print(json.dumps({"status": "written", "sha256": sha(data), "branches": 164, "pairs": 80}))
    else:
        saved = read(args.artifact, args.sha256, 8 * 1024 * 1024)
        validate_arm_declarations(saved["arms"])
        require(canonical(saved) == data, "contract mismatch")
        if args.command == "check":
            print(json.dumps(validate_comparison(saved["arms"], args.claim)))
        else:
            print(json.dumps({"status": "same_implementation_reconstruction_verified", "sha256": sha(data)}))


if __name__ == "__main__":
    main()
