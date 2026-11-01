"""Bounded, read-only evidence inventory; never a scientific admission authority.

Run from le-wm-main with .venv/bin/python -m
experiments.spectral_reachability.d2p_downstream_frontier. JSON goes to stdout.
No discovery walk, source execution, model import, hashing, or artifact writes.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

MAX_BYTES = 2_000_000
SOURCES = {
    "admission": "experiments/spectral_reachability/d2p_stage_admission.py",
    "composition_prototype": "experiments/spectral_reachability/reacher_d2p_program_dispatch.py",
    "legacy_sra_horizon": "experiments/spectral_reachability/run_sra_p3_horizon_expansion_cuda.py",
    "sampling_baselines": "experiments/spectral_reachability/d2p_sampling_baselines.py",
    "reported_baselines": "experiments/spectral_reachability/d2p_reported_baselines.py",
    "composition_tests": "tests/test_reacher_d2p_program_dispatch.py",
    "admission_tests": "tests/test_d2p_stage_admission.py",
}
REPORTS = {
    "physical_calibration": "results/pusht_d2p_sealed_program_execution4_v1_20261026_cuda/report.json",
    "official_contract": "results/pusht_d2p_sealed_task_contract4_v1_20261026/report.json",
}
STAGES = {
    "P2": ("multi-program composition", ["P1 admitted", "action witness and predicted postcondition on every edge", "frozen chart transition/search rule", "physical benefit over no-composition SRA and budget-matched flat search"]),
    "P3": ("dynamics-directed horizon and compute", ["P1 admitted", "necessary P2 composition validated", "frozen geometry scheduler", "fixed-horizon/budget and noise/cost ablations"]),
    "P4": ("RGB-only multi-environment closed loop", ["each environment independently admitted", "applicable P1/P2/P3 prerequisites", "root-paired physical success/cost", "PushT approach/contact/push/closure breakdown", "no privileged planner state"]),
    "P5": ("cross-task and independent-model replication", ["P1 and applicable upstream stages admitted", "predeclared frozen independent architecture", "per-model dynamics prerequisites", "same-model multiple tasks separately from cross-model replication", "no training/adapters/repair substitution"]),
}


def bounded_text(root: Path, relative: str) -> str:
    path = root / relative
    # Symlinks outside the supplied tree are not evidence in this inventory.
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("path outside evidence root")
    with path.open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("artifact exceeds bounded-read limit")
    return data.decode("utf-8")


def inspect(root: Path, relative: str, *, source=False) -> dict:
    entry = {"path": str(root / relative), "available": False}
    try:
        raw = bounded_text(root, relative)
        if source:
            tree = ast.parse(raw)
            entry["symbols"] = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
            entry["static_parse_only"] = True
            entry["symbol_lines"] = {n.name: n.lineno for n in tree.body if isinstance(n, ast.FunctionDef)}
            entry["literal_contracts"] = {}
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("P0_INTEGRITY_KEYS", "P1_INTEGRITY_KEYS", "P1_MECHANISM_KEYS"):
                    entry["literal_contracts"][node.targets[0].id] = ast.literal_eval(node.value)
        else:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("report must be a mapping")
            entry["report"] = payload
        entry["available"] = True
    except (OSError, ValueError, SyntaxError, UnicodeError) as exc:
        entry["error"] = str(exc)
    return entry


def task_contract_counts(report: dict) -> dict:
    """Recount saved evaluator labels, not physics, inference, or re-evaluation."""
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("missing contract rows")
    roots, terminals, stages = set(), [], 0
    for row in rows:
        identity = row["root_sha256"]
        if identity in roots:
            raise ValueError("duplicate root")
        roots.add(identity)
        candidates = set()
        for branch in row["branches"]:
            candidate = branch["candidate_id"]
            if candidate in candidates:
                raise ValueError("duplicate candidate")
            candidates.add(candidate)
            values = branch["stages"]
            if not isinstance(values, list) or len(values) != 2:
                raise ValueError("expected two saved stages")
            for value in values:
                if any(type(value.get(key)) is not bool for key in ("official_success", "object_only_success")):
                    raise ValueError("missing boolean success labels")
            stages += len(values)
            terminals.append(values[-1])
        if len(candidates) != 16 or row["selected_candidate_id"] not in candidates:
            raise ValueError("incomplete sealed bank")
    if len(roots) != 4:
        raise ValueError("expected four calibration roots")
    return {"roots": len(roots), "branches": len(terminals), "saved_stages": stages,
            "official_terminal_successes": sum(v["official_success"] for v in terminals),
            "object_only_terminal_successes": sum(v["object_only_success"] for v in terminals)}


def audit(root: Path) -> dict:
    sources = {key: inspect(root, path, source=True) for key, path in SOURCES.items()}
    evidence = {key: inspect(root, path) for key, path in REPORTS.items()}
    contract = evidence["official_contract"]
    if contract["available"]:
        try:
            report = contract["report"]
            if report.get("schema") != "pusht_sealed_task_contract.v1" or report.get("status") != "completed":
                raise ValueError("wrong schema or incomplete task-contract report")
            contract["saved_label_recount"] = task_contract_counts(report)
        except (KeyError, TypeError, ValueError) as exc:
            contract["recount_error"] = str(exc)
    # Avoid copying all saved states into the small frontier artifact.
    for entry in evidence.values():
        payload = entry.pop("report", {})
        entry["reported_metadata"] = {k: payload[k] for k in ("schema", "status", "gates", "scope", "P1_admission", "official_goal_includes_agent_position") if k in payload}
    stages = {}
    for name, (definition, requirements) in STAGES.items():
        stages[name] = {"definition": definition, "prerequisites": requirements,
            "implemented": "prototype only" if name == "P2" and sources["composition_prototype"]["available"] else "not established for full D2P stage",
            "tested": "test source inventoried, execution not inferred",
            "calibration": "P1 supporting calibration only; not this stage",
            "physically_validated": False, "admitted": False,
            "blockers": ["P1 not admitted", "no stage-specific independently validated admission receipt in this bounded inventory"],
            "validator_status": "p2_admission scalar gate exists" if name == "P2" else "no dedicated later-stage validator in d2p_stage_admission.py"}
    return {"schema": "d2p_downstream_frontier.v1", "scope": "explicit allowlist snapshot, not repository-wide proof of absence or a source-seal verification",
        "read_only": True, "neural_work": 0, "new_physical_execution": 0,
        "P1_admitted": False, "stages": stages, "sources": sources, "evidence": evidence,
        "p2_scalar_contract": ["calibration_only must be false", "declared minimum eligible roots >= 2 and achieved", "mechanism gate must be a declared P1_MECHANISM_KEYS member", "all P1 integrity keys plus selected mechanism key exactly true; every supplied gate true", "required comparisons include zero_prefix_two_stage_d2p, greedy_two_stage_d2p, cem_full_horizon and all predeclared extra controls", "every supplied comparison has finite ordered ci95 with upper endpoint strictly below zero"],
        "independent_work_executed": ["bounded source/test interface inventory", "saved official/object-only success-label recount"],
        "safe_runnable_commands": [".venv/bin/python -m pytest -q tests/test_d2p_downstream_frontier.py tests/test_d2p_stage_admission.py", ".venv/bin/python -m experiments.spectral_reachability.d2p_downstream_frontier"],
        "admission_boundary": "Never promotes affirmative artifact flags. Future admission requires a separately authorized source/root/protocol-bound validator; scalar intervals alone are insufficient."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(audit(args.root.resolve()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
