"""Recompute two fixed legacy P1 panels, retaining every stratum and source gap.

CPU only; neither training, physical execution, nor a changed gate.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from . import run_matched_learned_transfer_benchmark as benchmark
from . import run_cross_environment_transfer_diagnostic as cross

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
NAME = "legacy_p1_failure_attribution_20261102_exclusive"
OLD = RESULTS / "cross_environment_final_strict_20261013/matched_learned_transfer_benchmark.json"
EXTERNAL = RESULTS / "cross_environment_transfer_diagnostic_external_20261015/cross_environment_transfer_diagnostic.json"
OUTPUT = RESULTS / (NAME + "_RUN01")
OLD_ROOTS = (
    RESULTS / "pointmaze_final_quality_matched_20261013/records",
    RESULTS / "pusht_final_quality_matched_20261013/records",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_record(path):
    try:
        data = path.read_bytes()
        return {"path": str(path), "sha256": digest(data), "bytes": len(data)}
    except OSError as exc:
        return {"path": str(path), "error_type": type(exc).__name__, "error": str(exc)}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def check_source(record):
    source = record.get("provenance", {}).get("external_evidence")
    if not source:
        return {"state": "no_external_evidence_path"}
    path = Path(source)
    if not path.is_absolute():
        path = ROOT / path
    meta = file_record(path)
    if "sha256" not in meta:
        return {"state": "external_evidence_missing", **meta}
    try:
        evidence = json.loads(path.read_bytes())
        a = evidence["action_risk"]["action_inverse_risk"]
        risk = evidence["planning_consequence"]["planning_risk"]
        same = (math.isclose(float(a), float(record["diagnostics"]["action_inverse_risk"]),
                             rel_tol=1e-12, abs_tol=1e-12)
                and math.isclose(float(risk), float(record["heldout"]["planning_risk"]),
                                 rel_tol=1e-12, abs_tol=1e-12))
        return {"state": "source_values_match" if same else "source_values_mismatch",
                **meta, "schema": evidence.get("schema"),
                "action_risk": a, "planning_risk": risk,
                "source_protocol": evidence.get("protocol"),
                "planning_protocol": evidence["planning_consequence"].get("protocol"),
                "planning_metric_family": evidence["planning_consequence"].get("metric_family")}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {"state": "external_evidence_invalid", **meta,
                "error_type": type(exc).__name__, "error": str(exc)}


def panel(paths, historical, label, seed):
    hashes = [file_record(path) for path in paths]
    loaded, failures = cross._load_all(paths, require_external=True)
    if len(loaded) < 2:
        return {"label": label, "input": hashes, "load_failures": failures,
                "recomputed": None, "error": "fewer than two loadable records"}
    result = benchmark.audit(loaded, require_external=True, bootstrap_seed=seed)
    by_group = defaultdict(list)
    for item in loaded:
        by_group[benchmark._stratum_key(item)].append(item)
    rows = []
    for row in result["strata"]:
        key = (row["environment"], row["training_stage"], row["latent_dimension"], row["action_distribution"])
        candidates = by_group[key]
        eligible = set(row["eligible_checkpoints"])
        details = []
        for item in candidates:
            d = item["diagnostics"]
            details.append({
                "source_path": item["_source_path"], "checkpoint": item["checkpoint"],
                "architecture": item["_architecture"], "seed": item["_seed"],
                "eligible": item["checkpoint"] in eligible,
                "prediction_mse": d["prediction_mse"],
                "action_inverse_risk": d.get("action_inverse_risk"),
                "geometry": {metric: d.get(metric) for metric in benchmark.METRICS},
                "heldout_planning_risk": item.get("heldout", {}).get("planning_risk"),
                "external_evidence": check_source(item),
                "fit_export": item.get("provenance", {}).get("fit_export"),
                "outcome_export": item.get("provenance", {}).get("outcome_export"),
            })
        rows.append({**row, "candidates": details})
    historical_rows = historical["strata"]
    comparisons = []
    historical_by_key = {(r["environment"], r["training_stage"], r["latent_dimension"],
                          r["action_distribution"]): r for r in historical_rows}
    for row in result["strata"]:
        key = (row["environment"], row["training_stage"], row["latent_dimension"], row["action_distribution"])
        old = historical_by_key.get(key)
        fields = ("candidate_count", "eligible_checkpoints", "identifiable", "reason_if_gated",
                  "selectors", "delta_risk_mse_minus_full")
        comparisons.append({"key": key, "saved_stratum_exists": old is not None,
                            "different_fields": [f for f in fields if old is None or clean(old.get(f)) != clean(row.get(f))]})
    after = [file_record(path) for path in paths]
    return clean({"label": label, "input": hashes, "input_stable": hashes == after,
                  "load_failures": failures, "source_consumer": "existing benchmark._load and audit, require_external=True",
                  "stored_summary": historical["summary"], "recomputed_summary": result["summary"],
                  "historical_stratum_comparisons": comparisons,
                  "historical_key_set_matches": len(historical_rows) == len(result["strata"]) and
                      all(x["saved_stratum_exists"] for x in comparisons),
                  "strata": rows})


def render(report):
    lines = ["# Legacy ATM/MSE P1: per-stratum failure attribution", "",
             "Saved-record recomputation, not training, a new gate, or an independent replication.",
             "Risk and regret are the original selector outputs; normalized by each stratum's train-only scale.",
             "A stratum is not automatically an independent model or root sample.", ""]
    for p in report["panels"]:
        lines += ["## " + p["label"], "",
                  f"Inputs: {len(p['input'])}; load failures: {len(p['load_failures'])}; input stable: {p.get('input_stable')}.",
                  f"Stored versus recomputed row differences: {sum(bool(x['different_fields']) for x in p.get('historical_stratum_comparisons', []))}.",
                  "", "| Environment | stage | pool / eligible | status | MSE checkpoint | full checkpoint | MSE regret | full regret | ΔR |",
                  "|---|---:|---:|---|---|---|---:|---:|---:|"]
        for row in p.get("strata", []):
            s = row["selectors"]
            a, b = s.get("mse_only", {}), s.get("full_mse_action_risk_geometry", {})
            def short(v):
                if not v:
                    return "—"
                path = Path(v)
                return "/".join(path.parts[-3:])
            def number(v):
                return "—" if v is None else f"{v:.6g}"
            lines.append(f"| {row['environment']} | {row['training_stage']} | {row['candidate_count']} / {row['eligible_count']} | "
                         f"{row['reason_if_gated'] or 'identifiable'} | {short(a.get('checkpoint'))} | {short(b.get('checkpoint'))} | "
                         f"{number(a.get('regret'))} | {number(b.get('regret'))} | {number(row.get('delta_risk_mse_minus_full'))} |")
        lines += ["", "Full candidate paths, source hashes, observed source evidence, and exact selectors are in report.json.", ""]
    lines += ["## Interpretation limits", "",
              "- Historical seven-stratum action risk was documented as calibration transition error, not independent ATM.",
              "- External PointMaze evidence describes observed-action horizon consequence, not new closed-loop execution.",
              "- External evidence declarations and values are checked against saved files, not independently replayed.",
              "- A gate failure or unchanged selection does not establish absence of a real effect.",
              "- Shared seeds and outcomes across stages are not proven independent; no significance upgrade is made.", ""]
    return "\n".join(lines)


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError(OUTPUT)
    old_data, external_data = OLD.read_bytes(), EXTERNAL.read_bytes()
    historical, external = json.loads(old_data), json.loads(external_data)
    old_paths = sorted(path for root in OLD_ROOTS for path in root.rglob("checkpoint_record.json"))
    external_paths = [ROOT / path for path in external["input"]["paths"]]
    report = {"schema": NAME, "historical_sources": [file_record(OLD), file_record(EXTERNAL)],
              "panels": [panel(old_paths, historical, "20261013 provisional calibration-risk panel", 20261011),
                         panel(external_paths, external["global_audit"], "20261015 external-evidence panel", 20261013)],
              "no_training": True, "no_new_data": True, "gate_changed": False,
              "scientific_admission": False}
    if digest(OLD.read_bytes()) != digest(old_data) or digest(EXTERNAL.read_bytes()) != digest(external_data):
        raise ValueError("source report changed during recomputation")
    os.mkdir(OUTPUT, 0o700)
    for name, data in (("report.json", (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()),
                       ("report.md", render(report).encode())):
        fd = os.open(OUTPUT / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    print(json.dumps({"output": str(OUTPUT), "panels": [
        {"label": p["label"], "loaded": len(p["input"]) - len(p["load_failures"]),
         "row_differences": sum(bool(x["different_fields"]) for x in p.get("historical_stratum_comparisons", []))}
        for p in report["panels"]]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
