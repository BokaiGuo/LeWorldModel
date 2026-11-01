"""Saved-source-only exploratory adapter; no neural calls or physical dispatch."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import zlib

import numpy as np

from experiments.spectral_reachability import codex_p1_null_same_action_error_cuda_20261030_v1 as saved
from experiments.spectral_reachability import p1_constructive_observed_suffix_cuda_20261027_v1 as observed
from experiments.spectral_reachability import agent_p3_actual_cp5_task_builder_20261029_v1 as compact
from experiments.spectral_reachability import p1_moving_overlay_recompile_cuda_20261027_v1 as overlay

NAME = "codex_p1_backtrack_observed_adapter_20261030_v1"
ROOT = saved.ROOT
ARMS = ("projected_ascent", "outward_free_tangent")
ROLES = ("selected_prefix", "zero_prefix")
EXPORT_SCHEMA = "p1_moving_overlay_recompile_cuda_20261027_v1.task_export"
require = saved.require


def export_load(path, expected):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink()
            and path.resolve() == path and saved.sha(path) == expected, "export bytes")
    raw = path.read_bytes()
    require(len(raw) <= 8 * 1024**2, "compressed export cap")
    dec = zlib.decompressobj(31)
    plain = dec.decompress(raw, 32 * 1024**2 + 1)
    require(len(plain) <= 32 * 1024**2 and dec.eof and not dec.unused_data
            and not dec.unconsumed_tail, "single bounded gzip")
    return json.loads(plain)


def adapt(task, export, row, model_before, policy_hashes):
    """Join already authenticated export to compact task and one saved row."""
    compact.task_check(task)
    require(type(export) is dict and set(export) == {
        "schema", "task", "task_contract", "task_id", "P1_admission"
    } and export["schema"] == EXPORT_SCHEMA and export["P1_admission"] is False,
        "export schema")
    full, contract = export["task"], export["task_contract"]
    core = observed.base.encode.core
    overlay.task_interface(export)
    core.validate_task(full)
    require(export["task_id"] == core.task_id(full) == task["binding"]["task_id"]
            and core.digest(contract) == full["task_contract_sha256"], "export task/contract identity")
    for key in ("root_context", "goal", "projection", "action_mean", "action_std"):
        require(full[key] == task[key], "shared field mismatch: " + key)
    require(task["binding"]["model_sha256"] == model_before["original_state_dict"],
            "backtrack model state identity")
    require(full["model_sha256"] == policy_hashes["weights.pt"] and
            full["model_config_sha256"] == policy_hashes["config.json"], "export model files")
    require(contract["blind_sha256"] == task["binding"]["blind_sha256"] and
            contract["condition"] == row["condition"] and contract["root_slot"] == row["root_slot"]
            and contract["with_target"] is (row["condition"] == "on") and
            contract["pixel_sha256"][-1] == full["goal_source_sha256"], "source task semantics")
    require(np.array_equal(np.asarray(full["projection"]), np.eye(192)) and
            full["packet_logit_cap"] == float(.75*np.sqrt(5)) and
            np.array_equal(np.asarray(full["action_mean"], np.float32), observed.base.d2p.ACTION_MEAN) and
            np.array_equal(np.asarray(full["action_std"], np.float32), observed.base.d2p.ACTION_STD),
            "observed compiler Q, chart and normalizer")
    raw, _, _, _, _ = saved.action_inputs(row)
    warm = np.frombuffer(bytes.fromhex(row["warmup_raw_float32_hex"]), dtype="<f4").reshape(10, 2)
    require(np.array_equal(np.asarray(full["root_raw_action_history"], np.float32), warm),
            "actual root raw action history")
    require(row["task_id"] == export["task_id"], "row task ID")
    if row["role"] == "zero_prefix":
        require(np.array_equal(raw[0], np.zeros(10, np.float32)), "literal zero prefix")
    cp5 = row["images"][1]
    require(cp5["step"] == 5, "actual cp5 image")
    saved.rgb(cp5)
    return dict(index=row["index"], slot=row["slot"], root_slot=row["root_slot"],
                condition=row["condition"], arm=row["arm"], role=row["role"],
                original_task_id=export["task_id"], original_export_sha256=task["binding"]["task_sha256"],
                compact_contract_binding_sha256=task["binding"]["task_contract_sha256"],
                original_task_contract_sha256=full["task_contract_sha256"],
                task=deepcopy(full), cp5=deepcopy(cp5), prefix=raw[0].reshape(5, 2).tolist(),
                original_suffix10=raw[1:].reshape(10, 2).tolist())


def build():
    source = saved.material()  # verifies all 16 rows and historical source closure
    replay = saved.replay
    report = replay.one.saved.read_verified(replay.one.source.REPORT, replay.one.source.REPORT_SHA)
    prepared = replay.one.saved.read_verified(report["prepared"], report["prepare_sha256"])
    model_before = report["model_before"]
    policy = Path(prepared["policy"])
    policy_hashes = {name: saved.sha(policy / name) for name in ("weights.pt", "config.json")}
    require(source["model_hash"] == model_before and source["policy"] == str(policy), "model source join")
    rows = []
    for i, row in enumerate(source["rows"]):
        slot = i // 4
        require((row["index"], row["slot"], row["root_slot"], row["condition"], row["arm"], row["role"])
                == (i, slot, slot // 2, ("off", "on")[slot % 2], ARMS[(i % 4)//2], ROLES[i % 2]),
                "fixed sixteen order")
        task = prepared["requests"][slot]["task"]
        paths = [p for p, h in prepared["bindings"].items() if h == task["binding"]["task_sha256"]]
        require(len(paths) == 1, "unique bound task export")
        export = export_load(paths[0], task["binding"]["task_sha256"])
        rows.append(adapt(task, export, row, model_before, policy_hashes))
    for path, expected in source["bindings"].items():
        require(saved.sha(path) == expected, "source drift")
    require(replay.digest(policy / "weights.pt") == policy_hashes["weights.pt"] and
            replay.digest(policy / "config.json") == policy_hashes["config.json"], "model file drift")
    roster = []
    for first, second in zip(rows[::2], rows[1::2], strict=True):
        require(all(first[k] == second[k] for k in ("slot", "arm", "task", "original_task_id")),
                "same logical arm task")
        for role, index in ((ROLES[0], first["index"]), (ROLES[1], second["index"]),
                            ("greedy_prefix", None)):
            roster.append(dict(slot=first["slot"], arm=first["arm"], role=role,
                               row_index=index, status="source_verified" if index is not None else "unknown"))
    return dict(schema=NAME+".prepare", status="exploratory_saved_source_adapter_only",
                source_report_sha256=saved.REPORT_SHA, source_prepare_sha256=saved.PREPARE_SHA,
                original_prepare_sha256=report["prepare_sha256"], model_before=model_before,
                model_files=policy_hashes, adapter_sha256=saved.sha(Path(__file__).resolve()),
                rows=rows, formal_roster=roster, fixed_existing_rows=16, formal_conditions=24,
                missing_greedy=8, neural_calls=0, physical_calls=0, training=False,
                P1_admission=False, physical_authorized=False,
                warning="Not an isolated planner manifest or execution authorization; cp5 latent not yet encoded.")


def verify(path, expected):
    path = Path(path)
    require(path.is_absolute() and path.parent == ROOT/"results" and
            path.name.startswith(NAME+"_prepare_") and path.is_file() and
            not path.is_symlink() and path.resolve() == path and saved.sha(path) == expected,
            "prepared bytes or path")
    data = saved.native.read(path, cap=32 * 1024**2)
    require(data == build() and saved.sha(path) == expected, "prepared or source drift")
    return dict(status="saved_source_checks_passed_exploratory_only", rows=len(data["rows"]),
                formal_conditions=len(data["formal_roster"]), greedy_unknown=data["missing_greedy"],
                physical_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if args.verify is not None:
        require(args.sha256 is not None, "external expected SHA required")
        print(json.dumps(verify(args.verify, args.sha256)))
        return
    require(args.sha256 is None, "SHA is only accepted for verification")
    path = args.output
    require(path.is_absolute() and path.parent == ROOT/"results" and
            path.parent == path.parent.resolve() and path.name.startswith(NAME+"_prepare_") and
            not os.path.lexists(path), "exclusive output path")
    result = build()
    payload = saved.encode_json(result)
    saved.write_exclusive(path, payload)
    print(json.dumps(dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(),
                          rows=len(result["rows"]), greedy_unknown=result["missing_greedy"],
                          physical_authorized=False)))


if __name__ == "__main__":
    main()
