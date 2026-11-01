"""Exact saved metadata contrasts, not experimental or compute equivalence."""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

from experiments.spectral_reachability import agent_p2_control_evidence_resolver_20261028_v1 as resolver

v3 = resolver.contract
NAME = "agent_p2_resolved_controls_comparison_20261028_v1"
ROOT = v3.ROOT
CATALOG = ROOT / "results/agent_p2_control_evidence_resolver_20261028_v1_saved02.json"
CATALOG_SHA = "baa16c6bb9661be94c897077a48824278148c66efc01e5981e69af824b198685"
CONTRACT = ROOT / "results/agent_p2_strong_baseline_saved_summary_20261028_v3_inventory_MAIN01.json"
CONTRACT_SHA = "0683c5d90afb00680f2bcfe45d0fc6bf8309d8c82a4c9448b26960069ebe289f"
RESOLVER_SHA = "d93dd17ea672b31d1a4de122b6fcfa2ab265c0cbadf1c7cccdaf1aefab7a6344"
V3_SHA = "40ed045f8b8109de8c7c70b4b8d63e09958b8ac9773c7854ca29a01f04b93cb5"
require = resolver.require


def cohort(asset):
    """Only a complete task-ID join permits source-local roots to align."""
    tasks = asset["controls"].get("task_identity")
    if not v3.known(tasks) or len(tasks) != len(asset["fixed_rows"]):
        return None
    out = []
    for task in tasks:
        require(set(task) == {"/task_id"}, "task identity pointer schema")
        out.append(task["/task_id"])
    return out


def by_task(asset, field):
    ids = cohort(asset)
    values = asset["controls"].get(field)
    if ids is None or not v3.known(values) or len(values) != len(ids):
        return None
    grouped = {}
    for task_id, value in zip(ids, values):
        # Arms stay in the full census. A repeated task must not disagree on control.
        if task_id in grouped:
            require(grouped[task_id] == value, "within-task control conflict " + field)
        grouped[task_id] = value
    return grouped


def compare(left, right):
    lid, rid = cohort(left), cohort(right)
    same_cohort = lid is not None and rid is not None and set(lid) == set(rid)
    fields = {}
    for field in v3.CONTROL_FIELDS:
        entry = {"status": "unknown", "reason": "missing complete saved control", "by_task": []}
        if field == "root_ids":
            # raw integer slots / root hashes have different namespaces; never compare directly.
            if lid is not None and rid is not None:
                entry.update(status="match" if same_cohort else "mismatch",
                             reason="root alignment through complete task identities; not freshness proof")
        elif field == "task_identity":
            if lid is not None and rid is not None:
                entry.update(status="match" if same_cohort else "mismatch", reason="complete task identity sets")
        elif not same_cohort:
            entry["reason"] = "same-task cohort not closed; no subset substitution"
        else:
            lvalues, rvalues = by_task(left, field), by_task(right, field)
            if lvalues is not None and rvalues is not None:
                entry["by_task"] = [{"task_id": task, "status": "match" if lvalues[task] == rvalues[task] else "mismatch"}
                                    for task in sorted(lvalues)]
                entry.update(status="match" if lvalues == rvalues else "mismatch",
                             reason="exact saved component equality only")
        if field == "checkpoint_identity":
            entry["boundary"] = "declared weights/config identities; original loaded bytes NOT certified"
        if field == "action_support":
            entry["boundary"] = "declared domain/normalizer only; search support, AD and coverage not equated"
        if field == "resource_axes":
            entry["boundary"] = "counts do not certify scoped limits/inherited budget or FLOPs"
        fields[field] = entry
    return {"fields": fields, "left_fixed_rows": left["fixed_rows"], "right_fixed_rows": right["fixed_rows"],
            "same_task_cohort": same_cohort,
            "all_metadata_controls_match": all(f["status"] == "match" for f in fields.values()),
            "matched_experiment_authorized": False, "delta": None,
            "required_next_controls": [{"field": key, "status": val["status"], "reason": val["reason"]}
                                       for key, val in fields.items() if val["status"] != "match"]}


def validate_inputs(catalog, inventory):
    require(catalog["schema"] == resolver.NAME + ".evidence", "catalog schema")
    require(inventory["schema"] == v3.NAME + ".contract", "v3 inventory schema")
    require(set(catalog["assets"]) == set(inventory["assets"]) == set(v3.ASSETS), "full three assets")
    for name, (path, sha) in v3.ASSETS.items():
        a, b = catalog["assets"][name], inventory["assets"][name]
        require(a["asset"] == resolver.binding(path, sha, ""), "asset binding " + name)
        require(a["fixed_rows"] == b["fixed_rows"], "full root/arm denominator " + name)
        require(len(a["rows"]) == len(a["fixed_rows"]), "row denominator")
        require([r["identity"] for r in a["rows"]] == a["fixed_rows"], "row identity order")
        require(set(a["controls"]) == set(v3.CONTROL_FIELDS), "all controls")
    require(catalog["matched"] is False and inventory["run_authorized"] is False, "no authority upgrade")


def build():
    require(v3.sha(Path(resolver.__file__)) == RESOLVER_SHA, "frozen resolver source")
    require(v3.sha(Path(v3.__file__)) == V3_SHA, "frozen v3 source")
    catalog = resolver.load(CATALOG, CATALOG_SHA)
    inventory = resolver.load(CONTRACT, CONTRACT_SHA)
    require(catalog == resolver.build(), "full resolver evidence reconstruction")
    require(inventory == v3.build(), "full v3 inventory reconstruction")
    validate_inputs(catalog, inventory)
    evidence = resolver.catalog(CATALOG, CATALOG_SHA, catalog)
    for name, fields in evidence.items():
        for field, binding in fields.items():
            require(v3.evidence_value(binding) == catalog["assets"][name]["controls"][field], "v3 evidence join")
    assets = catalog["assets"]
    return {"schema": NAME + ".comparison", "status": "saved_control_comparison_not_matched_experiment",
            "source_sha256": v3.sha(Path(__file__)),
            "inputs": {"catalog": resolver.binding(CATALOG, CATALOG_SHA, ""),
                       "v3_contract": resolver.binding(CONTRACT, CONTRACT_SHA, "")},
            "fixed_rows": {name: a["fixed_rows"] for name, a in assets.items()},
            "v3_evidence": evidence,
            "comparisons": {left + "__vs__" + right: compare(assets[left], assets[right])
                            for left, right in combinations(sorted(assets), 2)},
            "unresolved_asset_evidence": {name: a["missing"] for name, a in assets.items()},
            "minimal_acquisition_requests": [
                {"scope": "sampling", "request": "Original prepare path authenticated against report.prepare_sha256, then explicit task/sourcepack path+SHA and all four row joins. Do not infer paths or replace padding history with moving history."},
                {"scope": "all_methods", "request": "Original per-root-per-arm limit/inherited vector for every v3 RESOURCE_AXES entry. If absent, predeclare a new same-task run budget; do not backfill old counts as caps."},
                {"scope": "checkpoint", "request": "Original loader byte-binding receipts if claiming actual loaded checkpoint equivalence; otherwise retain declared-only status. New receipt cannot prove old loaded bytes."},
                {"scope": "execution", "request": "For a future matched execution, freeze all task/root/goal/history/normalizer and domain controls first, preserve all arms, and meter native API/samples/AD/inherited axes separately."}],
            "author_reported_values": [], "local_performance_ranking": None, "composition_benefit": None,
            "run_authorized": False, "equal_budget": False, "equal_FLOPs": False,
            "scientific_admission": False, "P1_admission": False, "P2_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("consume", "verify"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sha")
    args = parser.parse_args()
    value = build()
    if args.command == "consume":
        data = resolver.canonical(value) + b"\n"
        require(len(data) <= resolver.LIMIT, "bounded output")
        with args.output.open("xb") as stream:
            stream.write(data)
    else:
        require(args.sha is not None, "external receipt SHA")
        require(resolver.load(args.output.absolute(), args.sha) == value, "comparison reconstruction")
    print(resolver.canonical({"output": str(args.output.absolute()), "sha256": v3.sha(args.output),
                              "status": value["status"],
                              "field_status": {n: {k: f["status"] for k, f in p["fields"].items()}
                                               for n, p in value["comparisons"].items()}}).decode())


if __name__ == "__main__":
    main()
