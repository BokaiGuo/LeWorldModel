"""Bind fresh model--physical calibration admission to an immutable damping selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind(selection: dict[str, Any], admission: dict[str, Any], confirmation: dict[str, Any], *, selection_path: Path, admission_path: Path) -> dict[str, Any]:
    if selection.get("schema") != "d2p_reacher_damping_selection.v1" or selection.get("status") != "pass":
        raise ValueError("requires a passing damping-selection receipt")
    if admission.get("schema") != "d2p_reacher_p0_model_physical_calibration_admission.v1" or admission.get("status") != "pass":
        raise ValueError("requires a passing model-physical admission receipt")
    if admission.get("source_report_sha256") != _sha256(Path(admission["source_report"])):
        raise ValueError("admission source report hash does not match")
    if confirmation.get("status") != "calibration_only" or not confirmation.get("protocol", {}).get("calibration_only"):
        raise ValueError("admission source is not calibration-only")
    if not all(confirmation.get("gates", {}).values()):
        raise ValueError("admission source has failed execution gates")
    selection_contract = selection.get("calibration_contract", {})
    checks = {
        "runner": confirmation.get("inputs", {}).get("runner_sha256"),
        "checkpoint": confirmation.get("inputs", {}).get("weights_sha256"),
        "adapter": confirmation.get("inputs", {}).get("adapter_sha256"),
        "action_parameterization": confirmation.get("protocol", {}).get("action_parameterization"),
        "relinearization_steps": confirmation.get("protocol", {}).get("relinearization_steps"),
        "d2p_initialization": confirmation.get("protocol", {}).get("d2p_initialization"),
        "d2p_proposal_samples": confirmation.get("protocol", {}).get("d2p_proposal_samples"),
        "d2p_proposal_elites": confirmation.get("protocol", {}).get("d2p_proposal_elites"),
        "warmup_policy": confirmation.get("protocol", {}).get("warmup_policy"),
        "execution_root_reconstruction": confirmation.get("protocol", {}).get("execution_root_reconstruction"),
    }
    if any(selection_contract.get(key) != value for key, value in checks.items()):
        raise ValueError("damping selection and admission use incompatible contracts")
    selection_seeds = {int(seed) for seed in selection.get("calibration_root_seeds", [])}
    admission_seeds = {int(seed) for seed in admission.get("calibration_root_seeds", [])}
    confirmation_seeds = {int(row["seed"]) for row in confirmation.get("rows", [])}
    if not selection_seeds or not admission_seeds or admission_seeds != confirmation_seeds or selection_seeds & admission_seeds:
        raise ValueError("selection and admission calibration roots must be non-empty, exact, and disjoint")
    result = dict(selection)
    result["calibration_root_seeds"] = sorted(selection_seeds | admission_seeds)
    result["model_physical_admission"] = {
        "path": str(admission_path.resolve()),
        "sha256": _sha256(admission_path),
        "source_confirmation_report": admission["source_report"],
        "source_confirmation_sha256": admission["source_report_sha256"],
        "root_seeds": sorted(admission_seeds),
        "precommitted_rule": admission["precommitted_rule"],
        "strict_order_agreement_count": admission["strict_order_agreement_count"],
        "one_sided_exact_sign_p": admission["one_sided_exact_sign_p"],
    }
    result["damping_selection_source"] = {"path": str(selection_path.resolve()), "sha256": _sha256(selection_path)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable output exists: {args.output}")
    selection_path, admission_path = args.selection.resolve(), args.admission.resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    confirmation_path = Path(admission["source_report"])
    result = bind(selection, admission, json.loads(confirmation_path.read_text(encoding="utf-8")), selection_path=selection_path, admission_path=admission_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "selected": result["selected"], "calibration_root_seeds": result["calibration_root_seeds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
