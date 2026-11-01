"""Pre-run contract for a matched baseline comparison.

This is an asset/parameter contract only.  It consumes sealed metadata and
never launches a model, simulator, network request, or experiment.  Algorithm
family and algorithm name are descriptive and are intentionally excluded from
the equality key; task/control identity is not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

NAME = "agent_p2_strong_baseline_saved_summary_20261028_v3"
ROOT = Path(__file__).resolve().parents[2]
ASSETS = {
    "kinodynamic": (ROOT / "results/agent_kinodynamic_matched_domain_20261028_v1_cuda_MAIN01/report.json", "d4976aeca357b5324c74be8b2b7e3bfc254e82b834b3acfe1b0ab90a1ef4614c"),
    "no_composition": (ROOT / "results/p1_moving_no_composition_baseline_cuda_20261027_v1_consume_main01.json", "a05aa97f6c2688d7ae7a1fd8b961a23ccba1f4e45b14d95159cca63c5ba23167"),
    "sampling": (ROOT / "results/p1_sampling_physical_20261027_v1_main01/report.json", "c3eef328bee57ce5d42b25fcf5a9935af2d2282c9de3f8caf46f2b35b53a2ef6"),
}
CONTROL_FIELDS = (
    "task_identity", "checkpoint_identity", "action_support", "observation_spec",
    "eval_metric", "horizon", "resource_axes", "root_ids",
    "goal_identity", "source_authority",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular saved asset")
    require(path.stat().st_size < 128 * 1024**2, "bounded saved asset")
    return json.loads(path.read_bytes())


def write(path, value):
    path = Path(path)
    require(not path.exists(), "exclusive output")
    data = canonical(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(data)


def _rows(report):
    rows = report.get("rows")
    require(isinstance(rows, list) and rows, "saved report rows")
    return rows


def _resource_axes(name, report):
    # Preserve native axis names and scopes: API != samples != AD != wall.
    # Missing axes remain null, even when another report explicitly records 0.
    return {
        "planner": report.get("resources"),
        "sampling_search": report.get("sampling_cost"),
        "physical": report.get("counts"),
        "inherited": report.get("inherited_cost") or report.get("inherited_encoding_once"),
        "wall_seconds": report.get("elapsed_seconds"),
        "predictor_samples": report.get("predictor_samples"),
    }


def _asset_inventory(name, path, expected):
    require(sha(path) == expected, "sealed asset drift " + name)
    report = read(path)
    require(report.get("errors", []) == [], "saved asset has errors " + name)
    require(report.get("P1_admission") is False, "saved asset P1 flag " + name)
    rows = _rows(report)
    roots = []
    arms = []
    task_ids = []
    for row in rows:
        root = row.get("source_slot", row.get("root_sha256"))
        arm = row.get("arm")
        require(root is not None, name + " root identity unknown")
        require(arm is not None, name + " arm identity unknown")
        roots.append(root)
        arms.append(arm)
        if row.get("task_id") is not None:
            task_ids.append(row["task_id"])
    contract = report.get("contract") or {}
    controls = {
        "task_identity": sorted(set(task_ids)) or None,
        "checkpoint_identity": None,
        "action_support": contract.get("support_claim") or contract.get("action_map"),
        "observation_spec": None,
        "eval_metric": "native_official_evaluation" if name == "sampling" else ("model_terminal_cost" if name == "no_composition" else None),
        "horizon": contract.get("horizon") or (2 if name == "no_composition" else None),
        "resource_axes": _resource_axes(name, report),
        "root_ids": sorted(set(roots), key=str),
        "arm_ids": sorted(set(arms)),
        "goal_identity": None,
        "source_authority": sorted({row["source_authority_sha256"] for row in rows if row.get("source_authority_sha256")}) or None,
    }
    unknown = [field for field in CONTROL_FIELDS if not known(controls[field])]
    pairs = [{"root": root, "arm": arm} for root, arm in zip(roots, arms)]
    require(len({canonical(pair) for pair in pairs}) == len(pairs), "duplicate root/arm row")
    return {
        "asset": name, "path": str(path), "report_sha256": expected,
        "schema": report.get("schema"), "algorithm": contract.get("algorithm") or "unknown",
        "family": name, "controls": controls, "unknown_controls": unknown,
        "row_count": len(rows), "fixed_rows": pairs,
        "native_resource_evidence": controls["resource_axes"],
        "status": "metadata_inventory_only",
    }


def load_assets():
    return {name: _asset_inventory(name, path, expected) for name, (path, expected) in ASSETS.items()}


def implementation_closure():
    paths = (Path(__file__).resolve(), ROOT / "tests" / ("test_" + NAME + ".py"),
             ROOT / "innerdocs" / (NAME + ".md"))
    return {str(path): sha(path) for path in paths}


def known(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and value.lower() not in {"unknown", "unavailable", "tbd"}
    if isinstance(value, (dict, list)):
        return bool(value) and all(known(v) for v in (value.values() if isinstance(value, dict) else value))
    return isinstance(value, (bool, int, float)) and (not isinstance(value, float) or math.isfinite(value))


def _known(value, field):
    require(field in value, "missing control field " + field)
    require(known(value[field]), "unknown control field " + field)


def compare_controls(left, right):
    """Return a comparison decision; algorithm/family are not control keys."""
    for field in CONTROL_FIELDS:
        _known(left, field)
        _known(right, field)
    mismatches = {field: {"left": left[field], "right": right[field]} for field in CONTROL_FIELDS if left[field] != right[field]}
    return {
        "status": "matched_controls_algorithms_may_differ" if not mismatches else "control_mismatch_no_direct_comparison",
        "algorithm_left": left.get("algorithm", "unknown"),
        "algorithm_right": right.get("algorithm", "unknown"),
        "family_left": left.get("family", "unknown"),
        "family_right": right.get("family", "unknown"),
        "mismatches": mismatches,
        "delta": None,
    }


def build():
    assets = load_assets()
    comparisons = {}
    names = tuple(assets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            try:
                comparisons[f"{left}__vs__{right}"] = compare_controls(assets[left]["controls"], assets[right]["controls"])
            except ValueError as exc:
                comparisons[f"{left}__vs__{right}"] = {"status": "unknown_control_refuse", "reason": str(exc), "delta": None}
    return {
        "schema": NAME + ".contract",
        "implementation_closure": implementation_closure(),
        "status": "prepared_contract_refuses_run_until_controls_complete",
        "assets": assets,
        "comparisons": comparisons,
        "required_control_fields": list(CONTROL_FIELDS),
        "fixed_scope": {"all_saved_roots": True, "all_saved_arms": True, "selection": "none", "padding": "forbidden"},
        "run_authorized": False, "scientific_admission": False, "P1_admission": False,
        "limitations": ["algorithm/family differences alone do not block comparison", "unknown control fields refuse comparison", "no run authorization, fairness claim, or scientific conclusion is produced"],
    }


def pointer(document, path):
    """Resolve a JSON pointer, with no filename/slot-based identity guessing."""
    require(isinstance(path, str) and (path == "" or path.startswith("/")), "JSON pointer")
    value = document
    if path:
        for token in path[1:].split("/"):
            key = token.replace("~1", "/").replace("~0", "~")
            value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def evidence_value(binding):
    require(isinstance(binding, dict) and set(binding) == {"path", "sha256", "pointer"}, "exact evidence binding fields")
    path = Path(binding["path"])
    require(path.is_absolute(), "absolute evidence path")
    require(sha(path) == binding["sha256"], "evidence SHA drift")
    return pointer(read(path), binding["pointer"])


def prepare(request):
    """Validate a MAIN-supplied prospective parameter contract, not execute it.

    Every control and parameter value needs an exact saved JSON witness.  This
    proves byte/field binding only, not the scientific meaning of those fields.
    Failed/missing requests are refused, not repaired or filled with defaults.
    """
    assets = load_assets()
    require(set(request) == {"schema", "methods"}, "exact request fields")
    require(request["schema"] == NAME + ".request", "request schema")
    methods = request["methods"]
    require(isinstance(methods, dict) and set(methods) == set(assets), "all three methods required")
    for name, method in methods.items():
        require(set(method) == {"algorithm", "family", "parameters", "controls", "evidence", "fixed_rows", "resource_budget"}, "method fields")
        require(known(method["algorithm"]) and known(method["family"]), "method description")
        require(method["fixed_rows"] == assets[name]["fixed_rows"], "full fixed root/arm denominator " + name)
        require(set(method["controls"]) == set(CONTROL_FIELDS), "exact controls " + name)
        require(set(method["evidence"]) == set(CONTROL_FIELDS) | {"parameters", "resource_budget"}, "evidence closure " + name)
        for field in CONTROL_FIELDS:
            _known(method["controls"], field)
            require(evidence_value(method["evidence"][field]) == method["controls"][field], "control evidence " + name + "/" + field)
        for field in ("parameters", "resource_budget"):
            require(known(method[field]), "unknown " + field)
            require(evidence_value(method["evidence"][field]) == method[field], "parameter/budget evidence " + name)
        validate_budget(method["resource_budget"])
        require(method["controls"]["resource_axes"] == method["resource_budget"], "comparison must bind the complete budget vector")
    names = tuple(methods)
    comparisons = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            comparisons[f"{left}__vs__{right}"] = compare_controls(methods[left]["controls"], methods[right]["controls"])
    return {"schema": NAME + ".prepared", "status": "parameter_contract_bound_not_run_authorization",
            "implementation_closure": implementation_closure(), "assets": assets, "request": request,
            "comparisons": comparisons, "run_authorized": False, "fairness_claim": False,
            "scientific_admission": False, "P1_admission": False, "freshness_certified": False}


RESOURCE_AXES = ("predict_API", "predictor_samples", "action_encoder", "predictor", "pred_proj",
                 "encoder", "projector", "AD_forward", "VJP", "J", "SVD", "solve",
                 "wall_seconds", "physics_steps")


def validate_budget(budget):
    # No implicit zero: all axes require an explicit known value and scope.
    require(set(budget) == set(RESOURCE_AXES), "complete resource budget axes")
    for axis, entry in budget.items():
        require(isinstance(entry, dict) and set(entry) == {"unit", "scope", "limit", "inherited"}, "budget axis fields " + axis)
        unit = "seconds" if axis == "wall_seconds" else ("samples" if axis == "predictor_samples" else ("steps" if axis == "physics_steps" else "calls"))
        require(entry["unit"] == unit and entry["scope"] == "per_root_per_arm_including_signal_search_replay", "budget unit/scope")
        for key in ("limit", "inherited"):
            value = entry[key]
            require(type(value) in (int, float) and value >= 0, "known nonnegative budget " + axis)
            require(axis == "wall_seconds" or type(value) is int, "integer resource count")
    canonical(budget)  # reject non-finite values


def request_template():
    assets = load_assets()
    return {"schema": NAME + ".request", "methods": {
        name: {"algorithm": asset["algorithm"], "family": asset["family"],
               "fixed_rows": asset["fixed_rows"], "parameters": None,
               "controls": {field: None for field in CONTROL_FIELDS},
               "evidence": {field: None for field in (*CONTROL_FIELDS, "parameters", "resource_budget")},
               "resource_budget": {axis: None for axis in RESOURCE_AXES}}
        for name, asset in assets.items()}}


def verify(path, expected, request_path=None):
    require(sha(path) == expected, "contract SHA")
    value = read(path)
    rebuilt = build() if request_path is None else prepare(read(request_path))
    require(value == rebuilt, "contract reconstruction/input closure")
    return value


def main():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command", required=True)
    inventory = subs.add_parser("inventory")
    inventory.add_argument("--output", type=Path, required=True)
    template = subs.add_parser("template")
    template.add_argument("--output", type=Path, required=True)
    prepared = subs.add_parser("prepare")
    prepared.add_argument("--request", type=Path, required=True)
    prepared.add_argument("--output", type=Path, required=True)
    verifier = subs.add_parser("verify")
    verifier.add_argument("--contract", type=Path, required=True)
    verifier.add_argument("--sha", required=True)
    verifier.add_argument("--request", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        value = verify(args.contract, args.sha, args.request)
        print(json.dumps({"status": value["status"], "run_authorized": False}, sort_keys=True))
    else:
        value = build() if args.command == "inventory" else (request_template() if args.command == "template" else prepare(read(args.request)))
        write(args.output, value)
        print(json.dumps({"output": str(args.output.absolute()), "sha256": sha(args.output), "status": value.get("status", "incomplete_template_not_authorized")}, sort_keys=True))


if __name__ == "__main__":
    main()
