"""Audit saved-consumer wiring for engineering failure -> scientific-negative leaks.

This is a bounded source audit. It does not execute a runner, model, simulator,
CUDA, or gate. Findings are wiring risks, not scientific results.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/p1_engineering_failure_wiring_audit_20261102_exclusive_RUN01"
TARGETS = (
    ROOT / "experiments/spectral_reachability/p1_dev305_contact_equal_scale_saved_20261102.py",
    ROOT / "experiments/spectral_reachability/p1_dev305_contact_rgb_feedback_saved_20261102.py",
    ROOT / "experiments/spectral_reachability/p1_dev305_contact_model_select_saved_20261102.py",
    ROOT / "experiments/spectral_reachability/p1_dev304_spectral_cp10_saved_20261102.py",
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def source_findings(path):
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    tree = ast.parse(text, filename=str(path))
    joined = "\n".join(line.strip() for line in text.splitlines())
    accepts_nonfailed = '["status"] != "failed"' in text or "['status'] != 'failed'" in text
    requires_fixed = '["status"] == "fixed_horizon_completed"' in text or "['status'] == 'fixed_horizon_completed'" in text
    computes_terminal_from_last = "transitions[-1]" in text
    propagates_status = '"status": entry["status"]' in text or '"status": row["status"]' in text
    requires_report_complete = 'report["status"] == "recorded"' in text and "not report[\"errors\"]" in text
    risks = []
    if accepts_nonfailed and computes_terminal_from_last and not requires_fixed:
        risks.append("nonfailed_censored_or_partial_row_can_reach_terminal_metric")
    if accepts_nonfailed and computes_terminal_from_last and not propagates_status:
        risks.append("terminal_metric_output_does_not_carry_row_status")
    if not requires_fixed and not requires_report_complete:
        risks.append("no_static_complete-horizon_or_complete-report_guard_found")
    return {
        "path": str(path),
        "sha256": sha(raw),
        "syntax_ok": True,
        "signals": {
            "accepts_status_not_failed": accepts_nonfailed,
            "requires_fixed_horizon": requires_fixed,
            "requires_recorded_report_without_errors": requires_report_complete,
            "computes_from_last_transition": computes_terminal_from_last,
            "propagates_row_status": propagates_status,
        },
        "risks": risks,
        "classification": "wiring_risk" if risks else "guarded_for_static_contract",
        "ast_function_count": sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                                  for n in ast.walk(tree)),
        "matched_context": [line for line in text.splitlines()
                            if "status" in line or "transitions[-1]" in line][:20],
    }


def audit():
    findings = [source_findings(path) for path in TARGETS]
    return {
        "schema": "p1_engineering_failure_wiring_audit_20261102.v1",
        "scope": "static saved-consumer wiring only",
        "targets": findings,
        "wiring_risk_count": sum(item["classification"] == "wiring_risk" for item in findings),
        "scientific_negative_result": False,
        "engineering_failure_is_scientific_negative": False,
        "gate_changed": False,
        "long_experiment_run": False,
        "limitations": [
            "Static signals do not prove downstream callers misuse status.",
            "This audit does not repair existing consumers.",
            "A saved report with status recorded can still have semantic or source deficiencies outside this check.",
            "Rows with visibility_guard_stop or other censoring must remain censored/partial in any scientific summary.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    raw = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    output = args.output
    if not output.is_absolute() or output.resolve() != output or output.is_symlink():
        raise SystemExit("canonical absolute output required")
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({
        "output": str(output),
        "sha256": sha(raw),
        "wiring_risk_count": result["wiring_risk_count"],
    }))


if __name__ == "__main__":
    main()
