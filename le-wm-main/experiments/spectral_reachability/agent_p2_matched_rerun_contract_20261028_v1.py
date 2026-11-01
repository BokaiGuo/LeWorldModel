"""Prospective two-moving-root rerun wiring; no experiment launcher or quota invention."""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import shutil

from experiments.spectral_reachability import agent_p2_control_evidence_resolver_20261028_v1 as saved

NAME = "agent_p2_matched_rerun_contract_20261028_v1"
ROOT, v3, require = saved.contract.ROOT, saved.contract, saved.require
PREP_CAP, FLOOR = 256 * 1024, 200 * 1024**2
PREPARES = {
    "kinodynamic": (ROOT / "results/agent_kinodynamic_matched_domain_20261028_v1_prepare_MAIN01.json", "de3db628c5bd15f62f700fa56c917582d5e9fa27cf3cb979abf25a181d5e8dc9"),
    "no_composition": (ROOT / "results/p1_moving_no_composition_baseline_cuda_20261027_v1_prepared01.json", "bdaec3ee1b0bb424de368cf9dbe21ed84a075946bc33a637c04dd5cbed6ae7b1"),
}
MODULES = {
    "kinodynamic": "agent_kinodynamic_matched_domain_20261028_v1",
    "no_composition": "p1_moving_no_composition_baseline_cuda_20261027_v1",
}
SOURCES = {
    "experiments/spectral_reachability/agent_kinodynamic_matched_domain_20261028_v1.py": "8e6d03cdbae69382cd87bcb7f443fb882023b7930ef138bd02aa37f0950d46b8",
    "experiments/spectral_reachability/p1_moving_no_composition_baseline_cuda_20261027_v1.py": "8c709bed26d88e093b0293482be07cfc34702fb095b2fe23a22edadbbb985963",
    "experiments/checkpoint_compat.py": "50b01e20864dc929943df7cb3566c271a771e69c06ea0fdbc9b598c59e6e9dff",
    "experiments/spectral_reachability/p1_conditional_direct_screen_20261027_v1.py": "a05a2bea10ab365c8f2ee2cfcbec6ce4ccc53dc525e8b92b6e96a76529f1c104",
    "experiments/spectral_reachability/p1_constructive_blind_planners_cuda_20261027_v1.py": "e56bee610f4eb412cc03fbe49b3df91054d0defe23d24b6fd9c77e6a7f1e9f27",
}
ARMS = {"kinodynamic": "matched_domain_H2_beam20x6", "no_composition": "joint_H2_spectral_packet_mapped"}


def source_entry(path, function):
    """AST-only lookup; never import the loader/model/old execution modules."""
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function]
    require(len(nodes) == 1, "exact existing callable " + function)
    node = nodes[0]
    return {"path": str(path), "sha256": v3.sha(path), "function": function,
            "line": node.lineno, "end_line": node.end_lineno,
            "positional_parameters": [a.arg for a in node.args.args]}


def native_vectors(prepares):
    beam, spectral = prepares["kinodynamic"]["contract"], prepares["no_composition"]["contract"]
    require(beam["schedule"] == dict(first20=20, beam_width=6, second120=120, planning_predict=140,
        fresh_whole_predict=2, fresh_segmented_predict=2, total_predict=144, high_level_dispatches=143), "sealed beam schedule")
    require(spectral["per_root_budget"] == dict(predict=88, action_encoder=88, jacobian=1,
        vjp_directions=192, svd=2, bounded_solve=1), "sealed spectral budget")
    require(spectral["calls"] == dict(nominal=2, AD_forward=2, candidates=80, winner_whole=2, winner_segmented=2), "88 includes AD forwards")
    totals = {"kinodynamic": dict(predict_API=144, predictor_samples=144, action_encoder=144,
        predictor=144, pred_proj=144, encoder=0, projector=0, AD_forward=0, VJP=0, J=0, SVD=0, solve=0, physics_steps=0),
        "no_composition": dict(predict_API=88, predictor_samples=88, action_encoder=88,
        predictor=88, pred_proj=88, encoder=0, projector=0, AD_forward=2, VJP=192, J=1, SVD=2, solve=1, physics_steps=0)}
    result = {}
    for method, axes in totals.items():
        result[method] = {}
        for axis in v3.RESOURCE_AXES:
            count = axes.get(axis)
            result[method][axis] = {"limit": count, "unit": "seconds" if axis == "wall_seconds" else
                ("samples" if axis == "predictor_samples" else ("steps" if axis == "physics_steps" else "calls")),
                "scope": "per_root_per_arm_new_planning_including_search_and_replays",
                "inherited": 4 if axis in ("encoder", "projector") else None,
                "inherited_scope": "shared_input_attribution_not_repeated_execution" if axis in ("encoder", "projector") else "not_censused_not_assumed_zero",
                "status": "missing_original_per_root_wall_limit" if count is None else
                    ("new_hook_required_from_fixed_call_graph" if method == "no_composition" and axis in ("predict_API", "predictor_samples", "predictor", "pred_proj") else "sealed_protocol_limit"),
                "included_in": "predict_API/predictor/action_encoder/pred_proj" if axis == "AD_forward" else None}
    return result


def decision_status(decision):
    if decision is None:
        return {"selected": None, "blocked": True, "reason": "MAIN must choose native unequal quotas or commission a new equal-budget protocol before execution"}
    require(set(decision) == {"schema", "budget_policy", "acknowledge_unequal_API_AD", "wall_policy"}, "exact decision fields")
    require(decision["schema"] == NAME + ".decision", "decision schema")
    require(decision["budget_policy"] == "retain_native_144_vs_88", "unsupported quota change; equal144/equal88 need a separately frozen algorithm")
    require(decision["acknowledge_unequal_API_AD"] is True, "explicit unequal axes acknowledgement")
    require(decision["wall_policy"] == "retain_original_scopes_no_new_wall_claim", "no invented wall quota")
    return {"selected": decision, "blocked": False, "reason": "preserves old declared algorithms; not equal prediction or AD budget"}


def command_plan(method):
    prefix = [str(ROOT / ".venv/bin/python"), "-B", "-m", "experiments.spectral_reachability." + MODULES[method]]
    prepared = "NEW_" + method + "_PREPARE.json"
    output = "NEW_" + method + "_CUDA_OUTPUT"
    return {"callables": {f: source_entry(ROOT / "experiments/spectral_reachability" / (MODULES[method] + ".py"), f)
                           for f in ("prepare", "verify", "run", "consume")},
            "argv_templates": [prefix + ["prepare", "--output", prepared],
                prefix + ["verify", "--prepared", prepared, "--prepare-sha", "ACTUAL_NEW_PREPARE_SHA"],
                prefix + ["run", "--prepared", prepared, "--prepare-sha", "ACTUAL_NEW_PREPARE_SHA", "--output", output],
                prefix + ["consume", "--output", output, "--report-sha", "ACTUAL_NEW_REPORT_SHA"]],
            "launchable_under_this_contract": False,
            "reason": "old CLI callable, but missing same-consumed-byte loader receipts/new independent module hooks; MAIN-owned new adapter required, no monkeypatch or sealed edit"}


def loader_requirements(model_bindings, expected_tensor):
    return {"schema": NAME + ".new_loader_receipt", "loader_source": source_entry(ROOT / "experiments/checkpoint_compat.py", "load_compatible_pretrained"),
            "expected_files": model_bindings, "expected_tensor_hash": expected_tensor,
            "device": "cuda:0", "frozen_eval": True, "MATH": True, "TF32": False,
            "required_order": ["durable_load_intent", "open_config_and_weights", "hash_exact_consumed_buffers_before_deserialization",
                "json_parse_same_config_buffer", "torch_load_same_weights_buffer_weights_only_true",
                "original_key_remap_and_no_missing_or_unexpected", "CUDA_frozen_tensor_hash_before_first_dispatch",
                "actual_module_intents_returns_failures", "final_tensor_hash_and_file_recheck"],
            "receipt_fields": ["schema", "run_id", "contract_sha256", "loader_source_sha256", "device", "files",
                "tensor_hash_before", "tensor_hash_after", "frozen_eval", "errors"],
            "file_receipt_fields": ["path", "sha256", "bytes", "hash_buffer_sha256", "deserialize_buffer_sha256", "same_buffer"],
            "old_loader_gap": "config.read_text and torch.load(path) reopen files; pre/post hashes and tensor hash alone do not provide same-consumed-buffer file receipts",
            "implementation_required": "new MAIN-approved adapter preserving original remap/instantiation/strict missing-unexpected checks; no sealed loader modification or global monkeypatch",
            "old_runs_recertified": False}


def validate_loader_receipt(receipt, requirement, contract_sha):
    require(set(receipt) == set(requirement["receipt_fields"]), "exact loader receipt fields")
    require(receipt["schema"] == requirement["schema"] and receipt["contract_sha256"] == contract_sha, "loader contract binding")
    require(isinstance(receipt["run_id"], str) and receipt["run_id"], "new run id")
    require(receipt["loader_source_sha256"] == requirement["loader_source"]["sha256"], "loader semantics binding")
    require(receipt["device"] == "cuda:0" and receipt["frozen_eval"] is True and receipt["errors"] == [], "actual CUDA frozen loader")
    require(receipt["tensor_hash_before"] == receipt["tensor_hash_after"] == requirement["expected_tensor_hash"], "frozen expected tensors")
    files = receipt["files"]
    require(isinstance(files, list) and len(files) == len(requirement["expected_files"]), "all model files")
    require(len({f["path"] for f in files}) == len(files), "unique model file receipts")
    for f in files:
        require(set(f) == set(requirement["file_receipt_fields"]), "file receipt fields")
        require(f["path"] in requirement["expected_files"] and f["sha256"] == requirement["expected_files"][f["path"]], "exact model path and bytes digest")
        require(f["same_buffer"] is True and f["hash_buffer_sha256"] == f["deserialize_buffer_sha256"] == f["sha256"], "same consumed bytes required")
        require(type(f["bytes"]) is int and f["bytes"] > 0, "actual positive byte count")
    return {"status": "receipt_schema_bound_not_independent_runtime_proof", "old_runs_recertified": False}


def build(decision=None, decision_binding=None):
    require((decision is None) == (decision_binding is None), "decision external binding")
    for path, sha in SOURCES.items():
        require(v3.sha(ROOT / path) == sha, "frozen source drift " + path)
    prepares = {method: saved.load(path, sha) for method, (path, sha) in PREPARES.items()}
    a, b = prepares.values()
    require(a["slots"] == b["slots"] and [s["source_slot"] for s in a["slots"]] == [0, 1], "same complete two moving tasks")
    require(a["policy"] == b["policy"] and a["expected_model_hash"] == b["expected_model_hash"], "same model")
    require(a["inherited_encoding_report"] == b["inherited_encoding_report"] and
            a["inherited_encoding_report_sha256"] == b["inherited_encoding_report_sha256"], "same inherited evidence")
    inherited = saved.load(a["inherited_encoding_report"], a["inherited_encoding_report_sha256"])
    require(inherited["errors"] == [] and inherited["resources"] == {r: dict(attempted=8, completed=8) for r in ("encoder", "projector")}, "actual shared inherited encoding")
    tasks = []
    model_bindings = None
    for slot in a["slots"]:
        task = saved.load(slot["task_export"], slot["task_sha256"])
        evidence, missing = saved.task_evidence(slot, task, slot["task_export"], slot["task_sha256"], [])
        require(not missing, "complete fixed task fields")
        if model_bindings is None:
            model_bindings = task["task_contract"]["model_bindings"]
        require(model_bindings == task["task_contract"]["model_bindings"], "same checkpoint pair")
        require(task["with_target"] is False and task["history_semantics"] == "actual_moving_raw_between_observed0_5_10", "original off moving task")
        # Full arrays remain in authenticated immutable task exports. Avoid
        # duplicating dense identity192 Q in this small execution contract.
        references = {field: {pointer: {"origin": item["origin"],
            "value_sha256": __import__("hashlib").sha256(saved.canonical(item["value"])).hexdigest()}
            for pointer, item in components.items()} for field, components in evidence.items()}
        tasks.append({"slot": slot, "evidence_references": references})
    fixed = [{"source_slot": s["source_slot"], "task_id": s["task_id"], "method": m, "arm": ARMS[m]}
             for s in a["slots"] for m in MODULES]
    return {"schema": NAME + ".prepare", "status": "prospective_wiring_blocked_until_MAIN_decision_and_instrumented_executor",
            "implementation_sha256": {str(p): v3.sha(p) for p in (Path(__file__), ROOT / "tests" / ("test_" + NAME + ".py"), ROOT / "innerdocs" / (NAME + ".md"))},
            "original_prepares": {m: saved.binding(p, h, "") for m, (p, h) in PREPARES.items()},
            "source_bindings": SOURCES, "tasks": tasks, "fixed_rows": fixed,
            "per_root_arm_resource_vectors": native_vectors(prepares),
            "inherited_shared_once": {"origin": saved.binding(a["inherited_encoding_report"], a["inherited_encoding_report_sha256"], "/resources"),
                "resources": inherited["resources"], "per_root_attribution": {"encoder": 4, "projector": 4},
                "count_once_across_both_methods": True, "other_upstream_costs": None},
            "per_method_job_limits": {"kinodynamic": {"bytes": a["contract"]["output_cap_bytes"], "wall_seconds": a["contract"]["wall_seconds"]},
                "no_composition": {"bytes": b["contract"]["output_cap_bytes"], "wall_seconds": None}},
            "decision": decision_status(decision), "decision_binding": decision_binding,
            "quota_choice": {"native_values_per_root": {"kinodynamic": 144, "no_composition": 88},
                "retain_native": "executable algorithms unchanged; unequal API/samples and AD; joint task comparison only",
                "equal144": "old spectral bank has40 candidates and88 total calls; requires a NEW predeclared algorithm, no padding/best replay",
                "equal88": "old beam needs140 planning+4 replay calls; cannot run unchanged under88; requires a NEW reduced search protocol",
                "common_ceiling144": "possible future MAIN ceiling decision but spectral still uses88, NOT equal actual budget"},
            "interfaces": {m: command_plan(m) for m in MODULES},
            "loader_requirement": loader_requirements(model_bindings, a["expected_model_hash"]),
            "new_execution_evidence": {"all_four_rows": True, "operations": "durable intent before dispatch + returned/failed per API/sample/module, AD phase tagged, 0 encoder/projector/physics guards; preserve partials/no retries",
                "spectral_extra_hooks": "predict API, predictor, pred_proj and batch/sample count independently; existing predict meter is predictor entry, AD2 included in88",
                "inherited": "reference original shared8+8 with SHA; never claim newly repeated encodes or zero total upstream work",
                "wall": "beam900s job dispatch/write checks; spectral no old wall cap, MAIN must predeclare any new watchdog in a NEW executor protocol",
                "loaded_bytes": "new same-buffer load receipts plus before/after tensor hash and frozen guards; only certify NEW invocation"},
            "run_authorized": False, "equal_predict_calls": False, "equal_FLOPs": False,
            "old_runs_recertified": False, "P1_admission": False, "P2_admission": False, "composition_benefit": None}


def prepare(output, decision_path=None, decision_sha=None):
    decision = None
    bound = None
    if decision_path is not None:
        require(decision_sha is not None, "external decision SHA")
        decision = saved.load(Path(decision_path).absolute(), decision_sha)
        bound = saved.binding(Path(decision_path).absolute(), decision_sha, "")
    else:
        require(decision_sha is None, "decision path required")
    value = build(decision, bound)
    data = saved.canonical(value) + b"\n"
    path = Path(output).absolute()
    require(len(data) <= PREP_CAP and not path.exists() and not path.is_symlink(), "new bounded prepare")
    require(shutil.disk_usage(path.parent).free >= FLOOR + len(data), "200MiB floor")
    with path.open("xb") as stream:
        require(stream.write(data) == len(data), "complete write")
        stream.flush()
        os.fsync(stream.fileno())
    return value


def verify(path, expected):
    value = saved.load(Path(path).absolute(), expected)
    binding = value["decision_binding"]
    decision = None if binding is None else v3.evidence_value(binding)
    require(value == build(decision, binding), "full prospective contract reconstruction")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sha")
    parser.add_argument("--decision", type=Path)
    parser.add_argument("--decision-sha")
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.output, args.decision, args.decision_sha)
    else:
        require(args.sha is not None and args.decision is None and args.decision_sha is None, "external prepare SHA; decision bound inside")
        value = verify(args.output, args.sha)
    print(saved.canonical({"path": str(args.output.absolute()), "sha256": v3.sha(args.output),
        "status": value["status"], "rows": len(value["fixed_rows"]), "run_authorized": False}).decode())


if __name__ == "__main__":
    main()
