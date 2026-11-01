"""Prospective P2 resource-vector comparison; validation only, never dispatch.

The caller must independently retain a contract digest before any compared run.
Hashes bind bytes, not the time at which they were committed. Receipt values and
coverage are assertions until a separate journal/runtime verifier establishes them.
"""
import argparse
import hashlib
import json
from pathlib import Path

NAME = "p2_preregistered_resource_vector_20261030_codex"
SAVED_SHA = "b99215e0b69322c4cf0e9f3433beb6cdb41d6d114083f57c50ef6ecd1ee0b188"
ROOT = Path(__file__).resolve().parents[2]
SAVED = ROOT / "results/p2_saved_action_fairness_20261030_codex_CODEX01.json"

# Fixed coverage cannot be narrowed after seeing a result. These are separate
# dimensions: nested API/module counters are never added into a "total".
COUNTS = {
    "predict_samples": "sample",
    "vjp_directions": "direction",
    "array_svd_calls": "call",
    "solve_calls": "call",
    "selection_evaluations": "evaluation",
    "replay_calls": "call",
    "physical_steps": "substep",
    "flops": "floating_point_operation",
}
GAUGES = {"planning_wall_ns": "nanosecond", "peak_device_bytes": "byte"}
AXES = {**COUNTS, **GAUGES}
IDENTITY = ("root_sha256", "task_sha256", "model_sha256", "action_domain_sha256",
            "hardware_sha256", "meter_sha256")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def hash_string(value):
    return (type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def integer(value):
    return type(value) is int and value >= 0


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def pairs_unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field: " + key)
        result[key] = value
    return result


def read_bound(path, expected):
    require(hash_string(expected), "external SHA256 required")
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32 * 1024**2,
            "bounded regular input required")
    data = path.read_bytes()
    require(sha(data) == expected, "input SHA256 mismatch")
    def bad_constant(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(data, object_pairs_hook=pairs_unique, parse_constant=bad_constant)


def validate_contract(contract):
    require(type(contract) is dict and set(contract) ==
            {"schema", "arms", "scope", "identity", "axes"}, "exact contract fields")
    require(contract["schema"] == NAME + ".contract.v1", "contract schema")
    arms = contract["arms"]
    require(type(arms) is list and len(arms) >= 2 and
            all(type(x) is str and x.strip() == x and x for x in arms) and
            len(set(arms)) == len(arms), "at least two unique arms")
    require(contract["scope"] == "per_root_per_arm_standalone_planning_and_selected_physical_execution",
            "fixed standalone scope; inherited work is not silently omitted")
    identity = contract["identity"]
    require(type(identity) is dict and set(identity) == set(IDENTITY) and
            all(hash_string(identity[k]) for k in IDENTITY), "complete identity binding")
    axes = contract["axes"]
    require(type(axes) is dict and set(axes) == set(AXES), "fixed full axis roster")
    for axis, unit in AXES.items():
        spec = axes[axis]
        require(type(spec) is dict and set(spec) == {"unit", "aggregation", "cap"},
                "exact axis spec: " + axis)
        aggregation = "max" if axis == "peak_device_bytes" else "sum"
        require(spec["unit"] == unit and spec["aggregation"] == aggregation and integer(spec["cap"]),
                "unit, aggregation or finite cap: " + axis)
    return contract


def validate_receipt(receipt, contract, contract_sha):
    require(type(receipt) is dict and set(receipt) ==
            {"schema", "contract_sha256", "arm", "run_id", "identity", "status", "axes"},
            "exact receipt fields")
    require(receipt["schema"] == NAME + ".receipt.v1" and receipt["contract_sha256"] == contract_sha,
            "contract binding")
    require(receipt["arm"] in contract["arms"] and type(receipt["run_id"]) is str
            and receipt["run_id"].strip() == receipt["run_id"] and receipt["run_id"],
            "arm and run identity")
    require(receipt["identity"] == contract["identity"], "root/task/model/domain/hardware/meter mismatch")
    require(receipt["status"] == "complete", "incomplete or failed run")
    data = receipt["axes"]
    require(type(data) is dict and set(data) == set(AXES), "missing or extra resource axis")
    values = {}
    for axis in AXES:
        entry = data[axis]
        fields = {"unit", "coverage", "evidence_sha256"}
        fields |= {"attempted", "completed"} if axis in COUNTS else {"value"}
        require(type(entry) is dict and set(entry) == fields, "exact receipt axis fields: " + axis)
        require(entry["unit"] == AXES[axis] and entry["coverage"] == "complete"
                and hash_string(entry["evidence_sha256"]), "unknown coverage or evidence: " + axis)
        if axis in COUNTS:
            attempted, completed = entry["attempted"], entry["completed"]
            require(integer(attempted) and integer(completed) and completed <= attempted,
                    "invalid nonrefundable counts: " + axis)
            require(attempted == completed, "failed or pending work: " + axis)
            value = attempted
        else:
            value = entry["value"]
            require(integer(value), "unknown or invalid measured gauge: " + axis)
        require(value <= contract["axes"][axis]["cap"], "common cap exceeded: " + axis)
        values[axis] = value
    return values


def compare(contract, contract_sha, receipts):
    validate_contract(contract)
    require(hash_string(contract_sha) and sha(canonical(contract)) == contract_sha,
            "canonical contract bytes must match external hash")
    require(type(receipts) is list and len(receipts) == len(contract["arms"]), "fixed arm denominator")
    vectors = {}
    run_ids = set()
    for receipt in receipts:
        vector = validate_receipt(receipt, contract, contract_sha)
        arm = receipt["arm"]
        require(arm not in vectors and receipt["run_id"] not in run_ids, "duplicate arm or run")
        vectors[arm] = vector
        run_ids.add(receipt["run_id"])
    require(set(vectors) == set(contract["arms"]), "complete arm roster")
    reference = vectors[contract["arms"][0]]
    differences = {axis: {arm: vectors[arm][axis] for arm in contract["arms"]}
                   for axis in AXES
                   if any(vectors[arm][axis] != reference[axis] for arm in contract["arms"][1:])}
    return {"schema": NAME + ".comparison.v1", "contract_sha256": contract_sha,
            "scope": contract["scope"], "vectors": vectors, "unequal_axes": differences,
            "common_cap_satisfied": True, "actual_vector_equal": not differences,
            "interpretation": "validated receipt assertions only; common cap is not equal actual use; "
                              "no implied equivalence outside these axes",
            "preregistration_timing_verified": False, "runtime_evidence_verified": False,
            "equal_total_compute": False, "P1_admission": False, "P2_admission": False}


def historical_diagnostic():
    """Describe why existing four-way receipts cannot meet this new contract."""
    report = read_bound(SAVED, SAVED_SHA)
    require(report["schema"] == "p2_saved_action_fairness_20261030_codex.report",
            "historical report schema")
    roots = []
    for row in report["roots"]:
        recorded = row["neural_completed"]
        roots.append({"root_slot": row["root_slot"],
                      "predict_samples": recorded["predict_samples"],
                      "vjp_directions": recorded["VJP_directions"],
                      "unsupported_as_complete": ["array_svd_calls", "solve_calls",
                                                   "selection_evaluations", "replay_calls"],
                      "missing": ["flops", "planning_wall_ns", "peak_device_bytes"]})
    return {"schema": NAME + ".historical_diagnostic.v1", "saved_sha256": SAVED_SHA,
            "roots": roots, "prospective_contract_bound": False,
            "common_cap_satisfied": None, "actual_vector_equal": None,
            "reason": "historical source explicitly reports incomplete CPU accounting and unknown FLOPs; "
                      "a new preregistered full-coverage receipt does not exist",
            "P1_admission": False, "P2_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("historical-diagnostic")
    check = sub.add_parser("compare")
    check.add_argument("--contract", required=True, type=Path)
    check.add_argument("--contract-sha256", required=True)
    check.add_argument("--receipt", nargs=2, action="append", metavar=("PATH", "SHA256"), required=True)
    args = parser.parse_args()
    if args.command == "historical-diagnostic":
        result = historical_diagnostic()
    else:
        contract = read_bound(args.contract, args.contract_sha256)
        require(args.contract.read_bytes() == canonical(contract), "contract file must use canonical JSON bytes")
        receipts = [read_bound(path, digest) for path, digest in args.receipt]
        result = compare(contract, args.contract_sha256, receipts)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
