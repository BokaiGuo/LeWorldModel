"""Saved-evidence bridge and bounded arithmetic supplement; never a model runner.

No producer imports, torch, pickle, simulator, training, network or subprocess.
The original reported registry remains authoritative. This is not P1 admission.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path

PREFIX = "chat_baselines_20261027_"
ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = "results/p2_reported_baseline_table_20261027_v3/"
PINS = {
    ARCHIVE + "export/report.json": "52e687811e6d6f728ab3eda60836ea05eff11f36da2bb4af975d1dd19ef380f1",
    "results/p1_moving_baseline_comparison_20261027_v3_main01/comparison.json": "96cf27b143bb1a839edf7bc1e4de02f4a97e433c6475c4c731c78af76300d32e",
    "results/p1_hilewm_bounded_object_fetch_20261027_v2_probe_main01/report.json": "669bf41043dfae01a9bca183fe230d61baad29234c5cc73038c80409844e0b90",
}
DIMENSIONS = (
    "environment", "data_split", "checkpoint", "observations", "action_history",
    "action_mapping", "horizon", "feedback", "budget", "metric", "sample_ids",
    "statistical_unit", "training_and_adaptation", "hardware_and_timing",
)
COUNTS = {"GRASP": 112, "Hi-LeWM": 106, "LeWM": 8, "PlanWAM": 12}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def known(value):
    """Unknown inside a partial nested budget is still unknown, never matched."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and value.strip().lower() not in {
            "unknown", "missing", "unverified", "unmeasured", "n/a", "null"
        }
    if isinstance(value, dict):
        return bool(value) and all(known(v) for v in value.values())
    if isinstance(value, list):
        return all(known(v) for v in value)  # [] is explicit open-loop feedback.
    return type(value) is bool or type(value) in (int, float) and math.isfinite(value)


def compare_contracts(left, right):
    """Fail-closed metadata preflight, not a scientific-comparability certificate.

    Each dimension has a structured value and nonempty source locator. Values
    use identical units/schema; no guessed normalization or horizon conversion.
    Even a full match grants no numerical improvement or paired inference here.
    """
    for obj in (left, right):
        require(type(obj) is dict and set(obj) == set(DIMENSIONS), "complete dimension schema required")
        for entry in obj.values():
            require(type(entry) is dict and set(entry) == {"value", "source"}, "value/source entry required")
            require(isinstance(entry["source"], str) and bool(entry["source"].strip()), "source locator required")
    fields = {}
    for key in DIMENSIONS:
        a, b = left[key]["value"], right[key]["value"]
        fields[key] = ("unknown" if not known(a) or not known(b)
                       else "match" if canonical(a) == canonical(b) else "mismatch")
    return {"fields": fields, "metadata_match_only": all(v == "match" for v in fields.values()),
            "direct_comparison_authorized": False, "improvement_percent": None,
            "paired_inference_allowed": False, "scientific_admission": False}


class Tables(HTMLParser):
    """Read raw primary HTML tables; omit nested-table content from parents."""
    def __init__(self):
        super().__init__()
        self.stack = []
        self.tables = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table":
            self.stack.append({"id": attrs.get("id"), "rows": [], "row": None, "cell": None})
        elif self.stack:
            current = self.stack[-1]
            if tag == "tr":
                current["row"] = []
            elif tag in ("td", "th"):
                current["cell"] = []

    def handle_data(self, value):
        if self.stack and self.stack[-1]["cell"] is not None:
            self.stack[-1]["cell"].append(value)

    def handle_endtag(self, tag):
        if not self.stack:
            return
        current = self.stack[-1]
        if tag in ("td", "th") and current["cell"] is not None:
            require(current["row"] is not None, "cell without row")
            current["row"].append(" ".join("".join(current["cell"]).split()))
            current["cell"] = None
        elif tag == "tr" and current["row"] is not None:
            current["rows"].append(current["row"])
            current["row"] = None
        elif tag == "table":
            self.stack.pop()
            if current["id"]:
                require(current["id"] not in self.tables, "duplicate table id")
                self.tables[current["id"]] = current["rows"]


def weighted_rounding_check(values, weights, average):
    """Feasibility of one-decimal rounded means, NOT a recomputed exact result.

    Closed +/-0.05 intervals conservatively include tie conventions. No CI or
    training-seed uncertainty is inferred from rounding bounds.
    """
    require(len(values) == len(weights) > 0, "aligned nonempty strata required")
    require(all(type(w) is int and w > 0 for w in weights), "positive integer denominators required")
    require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 100
                for v in [*values, average]), "finite 0..100 scores required")
    numbers = [Decimal(str(v)) for v in values]
    require(all(x * 10 == (x * 10).to_integral_value() for x in [*numbers, Decimal(str(average))]),
            "one-decimal reported precision required")
    mean = sum(v * w for v, w in zip(numbers, weights)) / sum(weights)
    epsilon = Decimal("0.05")
    lo = sum(max(Decimal(0), v - epsilon) * w for v, w in zip(numbers, weights)) / sum(weights)
    hi = sum(min(Decimal(100), v + epsilon) * w for v, w in zip(numbers, weights)) / sum(weights)
    reported_lo = max(Decimal(0), Decimal(str(average)) - epsilon)
    reported_hi = min(Decimal(100), Decimal(str(average)) + epsilon)
    return {"denominator": sum(weights), "weighted_rounded_cells": str(mean),
            "weighted_rounding_interval": [str(lo), str(hi)],
            "reported_average_interval": [str(reported_lo), str(reported_hi)],
            "rounding_compatible": max(lo, reported_lo) <= min(hi, reported_hi),
            "is_confidence_interval": False, "new_model_evaluation": False}


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.inputs = {}

    def raw(self, relative, expected):
        path = self.root / relative
        require(not path.is_symlink() and path.resolve().is_relative_to(self.root), "input path escape")
        require(path.stat().st_size <= 8 * 1024 * 1024, "input exceeds 8 MiB")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        require(digest == expected, "input SHA changed: " + relative)
        self.inputs[relative] = digest
        return raw

    def recheck(self):
        for path, digest in list(self.inputs.items()):
            self.raw(path, digest)


def local_projection(saved, arm):
    """Project existing saved receipts; do not invoke P1's ownconsume/assemble."""
    require(saved["status"] == "comparison_recorded_not_admitted", "unexpected saved comparison status")
    require(all(saved[key] is False for key in (
        "official_reproduction", "equal_total_compute", "independent_confirmation", "P1_admission", "P2_admission"
    )), "saved evidence boundary changed")
    rows = [r for r in saved["rows"] if r["arm"] == arm]
    require(len(rows) == 2 and {r["source_slot"] for r in rows} == {0, 1}, "keep both fixed roots")
    summary, = [r for r in saved["summary"] if r["arm"] == arm]
    for column, aggregate in (("model_Q", "mean_model_Q"), ("realQ_terminal10", "mean_realQ_terminal10")):
        require(all(r[column]["status"] == "actual" and type(r[column]["value"]) in (int, float)
                    and math.isfinite(r[column]["value"]) for r in rows), "missing is not measured")
        mean = sum(r[column]["value"] for r in rows) / 2
        require(summary[aggregate]["status"] == "actual" and math.isclose(
            mean, summary[aggregate]["value"], rel_tol=1e-12, abs_tol=1e-12), "saved mean mismatch")
    require(all(r["native_success"]["status"] == "actual" and type(r["native_success"]["value"]) is bool
                and r["full_horizon"]["value"] is True for r in rows), "native/full-horizon evidence missing")
    require(sum(r["native_success"]["value"] for r in rows) == summary["native_successes"]["value"]
            and summary["native_denominator"] == 2, "native denominator mismatch")
    return {"arm": arm, "implementation": "local_variant", "official_reproduction": False,
            "new_independent_roots": 0, "reuse_only": True, "summary": summary,
            "rows": [{k: r[k] for k in ("source_slot", "task_id", "task_sha256", "model_Q",
                      "realQ_terminal10", "native_success", "resources")} for r in rows],
            "validation_scope": "saved JSON SHA + denominator/summary arithmetic; not producer closure audit"}


def reported_contract(row):
    rec, src = row["record"], row["source"]
    locator = rec["source_url"] + "#" + src["table_id"]
    values = {key: None for key in DIMENSIONS}
    values.update(environment=rec["env"], data_split=rec["dataset_split"], checkpoint=rec["checkpoint"],
                  observations=rec["observations"], action_mapping=rec["action_space"],
                  horizon={"reported": rec["horizon"], "unit": None},
                  feedback=({"schedule": "open_loop", "observed_replanning_steps": []}
                            if row.get("feedback") == "open_loop" else None),
                  budget={"reported": rec["budget"], "pipeline": None},
                  metric={"name": rec["metric"], "unit": rec["unit"], "domain": row["outcome_domain"]})
    return {key: {"value": value, "source": locator} for key, value in values.items()}


def local_contract(projected, source):
    # Only facts released by this saved projection; do not invent paper-native
    # bindings or infer uninspected model weight provenance from a method name.
    values = {key: None for key in DIMENSIONS}
    if "rows" in projected:
        values.update(data_split="fixed existing moving two calibration roots",
                      horizon={"model_blocks": 2, "raw_steps": 10, "steps_per_block": 5},
                      feedback={"schedule": "open_loop", "observed_replanning_steps": []},
                      sample_ids=[r["task_sha256"] for r in projected["rows"]],
                      statistical_unit={"fixed_roots": 2, "new_independent_roots": 0},
                      metric={"name": "native_success_count", "unit": "count", "domain": "physical_environment"},
                      budget={"saved_resources": projected["summary"]["resources"], "pipeline": None})
    return {key: {"value": value, "source": source} for key, value in values.items()}


def build(root=ROOT):
    evidence = Evidence(root)
    loaded = {p: json.loads(evidence.raw(p, h)) for p, h in PINS.items()}
    reported = loaded[ARCHIVE + "export/report.json"]
    require(reported["counts"] == COUNTS and len(reported["rows"]) == 238, "reported selection changed")
    require(len({r["id"] for r in reported["rows"]}) == 238, "duplicate reported cells")
    require(sum(r["record"]["value"] == "unknown" for r in reported["rows"]) == 2, "missing-cell census changed")
    for row in reported["rows"]:
        require(row["classification"] == "context_only" and row["record"]["evidence_kind"] == "paper_reported",
                "no promotion of reported evidence")
        src = row["source"]
        require(src["raw_row"][src["column_index_0based"]] == src["reported_token"], "literal cell binding")
        path = ARCHIVE + src["html_file"]
        if path not in evidence.inputs:
            evidence.raw(path, src["html_sha256"])
        require(evidence.inputs[path] == src["html_sha256"], "inconsistent source binding")
        if type(src["pdf_page_1based"]) is int:
            pdf_path = ARCHIVE + "sources/" + row["paper_key"] + "_pdf.pdf"
            if pdf_path not in evidence.inputs:
                evidence.raw(pdf_path, src["pdf_sha256"])
    # One genuinely additional, deterministic supplement: original Table13
    # scenario denominators joined to Table3 rounded scores, never neural CPU.
    html = evidence.raw(ARCHIVE + "sources/planwam_html.html",
                        "67be22fee283aa4bad78578c91d83e6a3085bc8927c63597ee7c9721073bd873").decode()
    tables = Tables()
    tables.feed(html)
    # Table13 caption is an empty figure; actual table is the immediately
    # following paragraph, not an A3.T13.* descendant. Pin its real HTML ID.
    candidates = [row for row in tables.tables["A3.SS4.p2.1"]
                  if row and row[0].lower() == "total"]
    total, = candidates
    require(total == ["Total", "436", "80", "157", "96", "103"], "primary Table13 total changed")
    source_rows = [r for r in tables.tables["A3.SS4.p2.1"]
                   if r and r[0] in {"nuScenes", "Waymo", "KITTI-360", "PandaSet"}]
    require(len(source_rows) == 4 and len({r[0] for r in source_rows}) == 4, "all four scenario sources required")
    require(all(sum(map(int, r[2:])) == int(r[1]) for r in source_rows), "scenario source row sum")
    require([sum(int(r[i]) for r in source_rows) for i in range(1, 6)] == list(map(int, total[1:])),
            "scenario column totals")
    checks = {}
    for metric in ("RC", "HDScore"):
        rows = sorted((r for r in reported["rows"] if r["paper_key"] == "planwam"
                       and r["record"]["metric"] == metric), key=lambda r: r["source"]["column_index_0based"])
        require(len(rows) == 5, "all four difficulty strata and Avg required")
        checks[metric] = weighted_rounding_check([r["record"]["value"] for r in rows[:4]],
                                                list(map(int, total[2:])), rows[4]["record"]["value"])
        checks[metric]["source_cell_ids"] = [r["id"] for r in rows]
        checks[metric]["denominator_source"] = "2610.11382v1 Table13 caption A3.T13; actual table A3.SS4.p2.1"
        checks[metric]["all_four_source_row_and_column_sums_verified"] = True
    moving = loaded["results/p1_moving_baseline_comparison_20261027_v3_main01/comparison.json"]
    probe = loaded["results/p1_hilewm_bounded_object_fetch_20261027_v2_probe_main01/report.json"]
    require(probe["status"] == "failed_preserved" and probe["unpickled"] is False, "probe state changed")
    local = {"GRASP": local_projection(moving, "GRASP_H3_bounded"),
             "LeWM": local_projection(moving, "CEM"),
             "Hi-LeWM": {"implementation": "missing_runnable_paired_checkpoint", "value": None,
                         "asset_probe_status": probe["status"], "first_error": probe["first_error"]},
             "PlanWAM": {"implementation": "missing_official_driving_pipeline", "value": None,
                         "warning": "local filenames containing planwam are not official PlanWAM"}}
    representatives = {
        "GRASP": "grasp:S4.T1.3:5:1",
        "Hi-LeWM": "hilewm:A3.T4.3:15:7",
        "LeWM": "lewm_v3:A7.T10.6:1:1",
        "PlanWAM": "planwam:S4.T1.6:16:11",
    }
    indexed = {r["id"]: r for r in reported["rows"]}
    matrix = {}
    for method, identity in representatives.items():
        row = indexed[identity]
        a = reported_contract(row)
        b = local_contract(local[method], "saved comparison JSON; absent local pipeline for missing methods")
        matrix[method] = {"paper_representative_not_best_sweep": row, "paper_contract": a,
                          "local_contract": b, "preflight": compare_contracts(a, b)}
    evidence.recheck()
    return {"schema": PREFIX + "bridge.v1", "status": "saved_evidence_and_arithmetic_supplement_completed",
            "reported_column": {"authoritative_report": ARCHIVE + "export/report.json",
                                "counts": COUNTS, "numeric_cells": 236, "missing_cells": 2,
                                "classification": "paper_reported/context_only"},
            "local_column": local, "comparability_matrix": matrix, "planwam_arithmetic_supplement": checks,
            "input_sha256": evidence.inputs, "scientific_admission": False,
            "improvement_percent": None, "new_neural_calls": 0, "new_physical_steps": 0,
            "training": False, "network_calls": 0, "new_independent_roots": 0,
            "validation_scope": "No producer reexecution, no model loading, no P1 closure audit"}


def publish(report, directory):
    directory = Path(directory)
    results = ROOT / "results"
    require(directory.parent.resolve() == results.resolve() and directory.name.startswith(PREFIX),
            "exclusive prefixed direct results child required")
    require(not directory.exists(), "output already exists; refuse overwrite")
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    require(len(payload.encode()) < 256 * 1024, "bounded output exceeded")
    directory.mkdir()
    name = PREFIX + "report.json"
    with (directory / name).open("x") as stream:
        stream.write(payload)
    with (directory / (PREFIX + "SHA256SUMS")).open("x") as stream:
        stream.write(hashlib.sha256(payload.encode()).hexdigest() + "  " + name + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new prefixed direct results child; otherwise stdout")
    args = parser.parse_args()
    result = build()
    if args.output:
        publish(result, args.output)
    else:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
