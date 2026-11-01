"""Freeze all selected/zero-prefix pairs from the pinned historical source join.

This is exploratory preparation only, not a physical execution authorization.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

from . import p1_backtrack_physical_contract_20261030 as old_contract
from . import codex_e1_sealed_event_axis_audit_20261030_v1 as util

ROOT = Path(__file__).resolve().parents[2]
JOIN = ROOT / "results/p1_backtrack_physical_source_join_20261030_MAIN02.json"
JOIN_SHA = "cb82a39b8a464f16b7e9f71cd9d072f2d48c0887bf36d57c91ca50baf0f2af84"
NAME = "codex_p1_prefix_null_20261030_v1"
ZERO = bytes(5 * 2 * 4)
ARMS = old_contract.ARMS
ROLES = ("selected_prefix", "zero_prefix")
DESIGN = ROOT.parent / "docs/D2P_P1_PREFIX_NULL_INTERVENTION_DESIGN_20261030.md"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def build(joined):
    """Pure transformation; the caller must first authenticate the source."""
    source = old_contract.plans(joined)
    selected = [x for x in source if x["role"] == "model_selected"]
    need(len(selected) == 8, "selected roster")
    rows, aliases = [], {}
    for logical, source_row in enumerate(selected):
        slot, arm = divmod(logical, 2)[0], ARMS[logical % 2]
        need(source_row["slot"] == slot and source_row["arm"] == arm, "selected order")
        original = bytes.fromhex(source_row["raw15_float32_hex"])
        for role in ROLES:
            raw = original if role == "selected_prefix" else ZERO + original[len(ZERO):]
            task = joined["task_rows"][slot]
            key = (task["task_id"], task["native_root_sha256"], task["condition"], raw.hex())
            alias = aliases.setdefault(key, len(aliases))
            row = copy.deepcopy(source_row)
            row.update(index=len(rows), logical_plan=logical, role=role,
                       raw15_float32_hex=raw.hex(), raw15_sha256=hashlib.sha256(raw).hexdigest(),
                       prefix_sha256=hashlib.sha256(raw[:len(ZERO)]).hexdigest(),
                       suffix_sha256=hashlib.sha256(raw[len(ZERO):]).hexdigest(),
                       duplicate_action_task_group=alias)
            rows.append(row)
    return {"schema": NAME + ".prepare", "join_sha256": JOIN_SHA,
            "task_rows": copy.deepcopy(joined["task_rows"]), "plans": rows,
            "code_and_design_sha256": {
                "prepare": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "source_contract": hashlib.sha256(Path(old_contract.__file__).read_bytes()).hexdigest(),
                "design": hashlib.sha256(DESIGN.read_bytes()).hexdigest(),
            },
            "contract": old_contract.contract(), "logical_plans": 8, "fixed_rows": 16,
            "duplicate_group_scope": "task_id_native_root_condition_raw15_only_not_execution_dedup_authority",
            "exploratory": True, "physical_authorized": False, "P1_admission": False,
            "independent_confirmation": False, "freshness": False,
            "deployment_status": "blocked_pending_native_adapter_and_saved_consumer_certification"}


def load():
    return util.parse(util.read_bound(JOIN, JOIN_SHA, 32 * 1024 * 1024))


def prepare(path):
    path = Path(path)
    need(path.is_absolute() and path.parent == ROOT / "results" and
         path.name.startswith(NAME + "_prepare_") and not path.is_symlink(), "exclusive output path")
    result = build(load())
    data = json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
            "status": result["deployment_status"], "physical_authorized": False}


def verify(path, expected_sha256):
    """Rebuild the pinned roster and reject byte, source, code, or design drift."""
    need(type(expected_sha256) is str and len(expected_sha256) == 64 and
         all(ch in "0123456789abcdef" for ch in expected_sha256), "external SHA required")
    path = Path(path)
    need(path.is_absolute() and path.parent == ROOT / "results" and
         path.name.startswith(NAME + "_prepare_"), "prepare path")
    saved = util.parse(util.read_bound(path, expected_sha256, 32 * 1024 * 1024))
    need(saved == build(load()), "source or code or design drift")
    return saved


def main():
    try:
        need(len(sys.argv) == 2, "expected absolute prepare output path")
        print(json.dumps(prepare(sys.argv[1]), sort_keys=True))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
