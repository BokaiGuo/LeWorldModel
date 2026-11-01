"""Saved-only audit of the existing multiround/SRA integration seam.

No producer imports, model loads, physical dispatch, or release path.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = {
    "multiround_source": ("experiments/spectral_reachability/p4_multiround_feedback_runtime_20261027_v1.py", "d106f75800eba62ca8743b77ffba48afcc23bbaac3e4ff3a412c2616240a6786"),
    "root_adapter_source": ("experiments/spectral_reachability/p4_feedback_root_runtime_20261027_v1.py", "b9206b3b3792b1e26f0b364693bc56d8e6e1a2f0edbbe939f3fcf2607635b650"),
    "sra_planning_source": ("experiments/spectral_reachability/p2_no_composition_sra_20261027_v2.py", "6e1402ac8abe3aabfffefd24aa016c341e6eb4f40d3f60e233152460ca70e45a"),
    "sra_physical_source": ("experiments/spectral_reachability/p2_no_composition_sra_physical_20261027_v2.py", "e5633f8f0982e00dfe05b4a056032d7aa877877832c11d75d48a045ccb93f722"),
    "multiround_protocol": ("results/p4_multiround_feedback_runtime_20261027_v1_prepare_v1/protocol.json", "4ed98914804ffb0b5bfb07d0245cbd597c43eb799e78cab6fecbdbb84300c3a3"),
    "multiround_report": ("results/p4_multiround_feedback_runtime_20261027_v1_prepare_v1/report.json", "f97ac090645cc417ea63ede4f9f883e8dba3891125a8d9f14fabe454c95f059e"),
    "sra_physical": ("results/p2_no_composition_sra_physical_20261027_v2_cuda_main_v1/physical_report.json", "fe105e05b73a9bdfd1b602eab740753b2a773b3ac42c27ca5c95156fe524bb03"),
}
OUT = ROOT / "results/codex_full_d2p_physical_preflight_20261030_LOCAL01.json"
MATCHED_ROOT = "23b8de7837c39fb4e0fcffe7678732a9d840c57de7a0c8376e22076770ef6050"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read_pinned():
    result = {}
    for name, (path, sha) in FILES.items():
        raw = (ROOT / path).read_bytes()
        need(hashlib.sha256(raw).hexdigest() == sha, "source hash mismatch: " + name)
        result[name] = raw.decode("utf-8")
    return result


def symbols(source, wanted):
    tree = ast.parse(source)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
            need(node.name not in found, "duplicate symbol")
            found[node.name] = node.lineno
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                name = node.name + "." + getattr(child, "name", "")
                if isinstance(child, ast.FunctionDef) and name in wanted:
                    need(name not in found, "duplicate symbol")
                    found[name] = child.lineno
    need(set(found) == set(wanted), "required seam symbol missing")
    return found


def build(inputs):
    interfaces = {
        "multiround": symbols(inputs["multiround_source"], ["arm_loop", "execute_five", "CudaBackend.compile", "run"]),
        "root_adapter": symbols(inputs["root_adapter_source"], ["bind_sources", "FeedbackAdapter.planning_request", "FeedbackAdapter.accept_selection", "FeedbackAdapter.record_execution"]),
        "sra_planning": symbols(inputs["sra_planning_source"], ["spectral_bank", "run_bank", "compile_cuda", "selected_actions"]),
        "sra_physical": symbols(inputs["sra_physical_source"], ["plans_from_saved", "physical_branch", "run"]),
    }
    protocol = json.loads(inputs["multiround_protocol"])
    report = json.loads(inputs["multiround_report"])
    sra = json.loads(inputs["sra_physical"])
    need(protocol["schema"] == "p4_multiround_feedback_runtime_20261027_v1"
         and protocol["root_sha256"] == MATCHED_ROOT and protocol["root_index"] == 1, "multiround identity")
    contract = protocol["contract"]
    need(contract["planned_steps_per_arm"] == 20 and contract["boundaries"] == [5, 10, 15]
         and contract["primitive_commands_per_packet"] == 5 and contract["candidates_per_compile"] == 10
         and contract["total_compute_matched"] is False and contract["P1_admission"] is False
         and contract["P4_admission"] is False, "multiround contract")
    need(report["schema"] == protocol["schema"] and report["status"] == "calibration_recorded"
         and report["P1_admission"] is False and report["P4_admission"] is False, "multiround report")
    rows = report["rows"]
    need(len(rows) == 2 and {r["arm"] for r in rows} == {"actual", "imagined"}
         and all(r["root_sha256"] == MATCHED_ROOT and r["status"] == "completed"
                 and r["selected_returns"] == 20 for r in rows), "multiround branches")
    srows = sra["rows"]
    need(len(srows) == 4 and len({r["root_sha256"] for r in srows}) == 4
         and all(r["method"] == "joint_h2_no_composition_sra" for r in srows), "SRA roster")
    matches = [r for r in srows if r["root_sha256"] == MATCHED_ROOT]
    need(len(matches) == 1, "missing shared root")
    row = matches[0]
    need(row["root_index"] == 1 and row["context_identity"]["goal"] == protocol["physical_goal"]
         and row["planned_commands"] == 10 and row["corrected_search_horizon"] == 2, "goal or SRA protocol")
    need(sra["planning_accounting"]["per_root"]["total_predict_samples"] == 88
         and report["actual_predict_samples"] == 80, "recorded accounting changed")
    return {
        "schema": "codex_full_d2p_physical_preflight_20261030_v1",
        "sources": {k: {"path": v[0], "sha256": v[1]} for k, v in FILES.items()},
        "static_entrypoints": interfaces,
        "actual_common": {"root_sha256": MATCHED_ROOT, "physical_goal7_equal": True,
                          "identity_scope": "root SHA and goal7 only; complete cross-run source, RGB, task, model and reset closure not established here"},
        "existing_assets": {
            "sra": {"roots": 4, "arms": 1, "physical_planned_steps": 10, "search_horizon_packets": 2,
                    "planning_predict_per_root": 88, "status": row["status"]},
            "multiround": {"roots": 1, "arms": ["imagined", "actual"], "physical_planned_steps": 20,
                           "initial_prefix_steps": 5, "recompile_boundaries": [5, 10, 15],
                           "candidate_count": 10, "combined_predict_samples_two_arms": 80,
                           "planning_horizon": "single five-step suffix at each boundary",
                           "status": report["status"]}},
        "integration_seam": [
            "A NEW orchestrator must connect per-arm observed RGB/history and completed executed packets to a pinned planning callback, then persist selection before guarded dispatch.",
            "Root-adapter planning_request/accept_selection/record_execution provide an existing interface, but require a compatible real source pack and external driver; static symbols do not validate runtime compatibility.",
            "SRA selected_actions exports a previously frozen H2 winner; it does not recompile SRA on live RGB. A same-feedback baseline requires a new explicitly budgeted online adapter or a prospectively specified shared control schedule.",
            "Do not replace the selected common prefix or rebind the one-root source by editing archived files; source/task/reset/RGB/action mapping must be proved independently for every roster slot.",
        ],
        "required_before_execution": [
            {"gate": "method_definition_and_science", "status": "blocked", "requirement": "Explicitly specify components constituting Full D2P and satisfy applicable P1/P2/P3 prerequisites; existing P4 calibration is not Full D2P."},
            {"gate": "prospective_roster_identity", "status": "blocked", "requirement": "Freeze root/goal/history/model/normalizer/reset/renderer identities and complete exclusion authority; current shared root is known, not fresh."},
            {"gate": "same_protocol_driver", "status": "blocked", "requirement": "Implement and test NEW matched orchestration for both methods: identical root and task, observation/guard/action semantics, time horizon, stops, and per-arm own feedback; bind actual source packs."},
            {"gate": "durable_resource_contract", "status": "blocked", "requirement": "Freeze enforceable per-axis limits and count attempts/returns, neural/AD, encoder, physics/setup, IO and wall; account for shared and inherited work once, unknown FLOPs remain null."},
            {"gate": "failure_and_saved_consumer", "status": "blocked", "requirement": "Fixed root/arm/time roster; persist pre-dispatch intent, selection and per-step returns; preserve early stops, failures and null later checkpoints, verify source closure and results."},
            {"gate": "authorization", "status": "blocked", "requirement": "Independent review and explicit future host GPU/physics authorization; this saved-only audit grants none."}],
        "full_d2p_comparison": None, "dispatch_authorized": False, "fair_total_budget": False,
        "new_neural_calls": 0, "new_physical_steps": 0,
        "limitations": ["Static symbol and pinned report checks are not a fullstrict source closure, compatibility test, fresh-root proof, or production readiness.",
                        "88 SRA per-root and 80 multiround two-arm total are different scopes, not a per-arm cost comparison.",
                        "These inputs establish no completed Full D2P physical arm; no claim about all possible workspace assets."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    raw = (json.dumps(build(read_pinned()), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if args.verify:
        need(OUT.read_bytes() == raw, "output mismatch")
        print("verified", OUT)
    else:
        with OUT.open("xb") as f:
            f.write(raw)
        print("created", OUT)


if __name__ == "__main__":
    main()
