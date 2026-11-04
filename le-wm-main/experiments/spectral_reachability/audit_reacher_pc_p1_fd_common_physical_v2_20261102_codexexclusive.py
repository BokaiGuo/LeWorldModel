"""Read-only verification of sealed FD common-tail physical development output."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np

REPORT_SHA = "4e100e02f8c3feda959f3b25883663e66b063126d4d599fd83a77a40607caa3e"
PLAN_SHA = "6de6f2ed64c80af6dd66eb27a8db920cffeeaf8d0c3bc82c3274dff6e305456a"
MODEL_SHA = "b76a55c6b6f8e13e7c4cbba0887e6ec6babc062c1a3245ddabf3991d39b00401"
SAVED_SHA = "127300623499118458970a433521e5c8a3f2c1e00c55c9d2ba69986c11f7ce9c"
ROOTS = (20263160, 20263161)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def actions(value, shape):
    a = np.asarray(value, dtype=np.float32)
    require(a.shape == shape and np.isfinite(a).all() and np.all(np.abs(a) <= 1.000001),
            "invalid action array")
    return a


def action_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float32).tobytes()).hexdigest()


def rgb(saved):
    shape = saved["shape"]
    require(len(shape) == 3 and all(type(x) is int and x > 0 for x in shape)
            and shape[-1] == 3 and np.prod(shape) <= 50_000_000, "invalid RGB shape")
    payload = zlib.decompress(base64.b64decode(saved["uint8_zlib_base64"], validate=True))
    require(len(payload) == int(np.prod(shape)), "RGB length mismatch")
    digest = hashlib.sha256(payload).hexdigest()
    require(digest == saved["sha256"], "RGB hash mismatch")
    return digest


def qpos_distance(qpos, target):
    delta = (qpos[:2] - target[:2] + np.pi) % (2 * np.pi) - np.pi
    return float(np.linalg.norm(delta))


def verify_point(point):
    q = np.asarray(point["qpos"], dtype=np.float64)
    v = np.asarray(point["qvel"], dtype=np.float64)
    target = np.asarray(point["target_qpos"], dtype=np.float64)
    require(q.ndim == v.ndim == target.ndim == 1 and q.size >= 2 and target.size >= 2
            and v.size >= 1 and np.isfinite(q).all() and np.isfinite(v).all()
            and np.isfinite(target).all(), "invalid state")
    require(np.isfinite(point["qpos_distance"]) and
            np.isclose(qpos_distance(q, target), point["qpos_distance"], atol=1e-12, rtol=1e-12),
            "qpos distance mismatch")
    return rgb(point["rgb"])


def verify(report, plan, model, saved):
    require(report["schema"] == "pc_fd_common_physical_dev_result_v1"
            and plan["schema"] == "pc_fd_common_physical_dev_v1", "schema mismatch")
    require(report["plan_sha256"] == PLAN_SHA and plan["model_sha256"] == MODEL_SHA
            and plan["audit_sha256"] == SAVED_SHA and saved["report_sha256"] == MODEL_SHA
            and saved["plan_sha256"] == plan["model_plan_sha256"]
            and saved["preflight_sha256"] == plan["preflight_sha256"]
            and saved["status"] == "complete_saved_consistency", "sealed lineage mismatch")
    require(report["status"] == "finished" and report["expected_branches"] == 20
            and report["recorded_branches"] == report["complete_branches"] == 20
            and plan["expected_branch_count"] == 20, "branch denominator mismatch")
    for source in (plan["roots"], report["rows"], model["rows"], saved["rows"]):
        require(tuple(x["seed"] for x in source) == ROOTS, "root order mismatch")
    results = []
    for expected, row, source, witness in zip(plan["roots"], report["rows"], model["rows"], saved["rows"], strict=True):
        require(row["status"] == "branches_recorded" and source["status"] == "complete"
                and witness["status"] == "verified_saved_consistency"
                and source["compiler_receipt"]["development_selected_index"] == 4
                and source["compiler_receipt"]["strict_selected_index"] is None
                and witness["development_selected_index"] == 4 and witness["strict_selected_index"] is None,
                "selection or root status mismatch")
        for key in ("root_rgb_sha256", "goal_rgb_sha256"):
            require(row[key] == expected[key] == source[key], "root or goal identity mismatch")
        require(expected["history_index"] == source["history_index"], "history index mismatch")
        warmup = actions(expected["warmup_packets"], (3, 5, 2))
        require(action_hash(warmup) == row["warmup_actions_sha256"] == expected["warmup_sha256"],
                "warmup action binding mismatch")
        snapshot = row["root_snapshot"]
        for key in ("physics_integration_state", "integration_state_spec", "model_binary_sha256",
                    "target_qpos", "target_marker_pos", "rendered_goal_rgb", "qpos", "qvel"):
            require(key in snapshot, "incomplete root snapshot")
        require(rgb(snapshot["rendered_goal_rgb"]) == row["goal_rgb_sha256"], "snapshot goal mismatch")
        prefixes = actions(source["prefix_raw"], (9, 5, 2))
        tails = actions(source["tail_raw"], (5, 10, 2))
        expected_order = [(arm, idx, tail) for arm, idx in (("selected", 4), ("parameter_zero", 0))
                          for tail in range(5)]
        require(len(expected["branches"]) == len(row["branches"]) == 10, "root denominator mismatch")
        branch_results = []
        prefix_reference = {}
        for planned, branch, (arm, idx, tail) in zip(expected["branches"], row["branches"], expected_order, strict=True):
            require((planned["arm"], planned["prefix_index"], planned["tail_index"]) == (arm, idx, tail)
                    and (branch["arm"], branch["prefix_index"], branch["tail_index"]) == (arm, idx, tail),
                    "branch order mismatch")
            intended = np.concatenate((prefixes[idx], tails[tail]))
            actual = actions(branch["raw_actions"], (15, 2))
            declared = actions(planned["raw_actions"], (15, 2))
            require(np.array_equal(actual, intended) and np.array_equal(actual, declared)
                    and action_hash(actual) == branch["actions_sha256"] == planned["actions_sha256"],
                    "branch action binding mismatch")
            require(branch["status"] == "complete" and branch["stop"] is None
                    and branch["attempted_steps"] == branch["returned_steps"] == 15
                    and len(branch["trajectory"]) == 16, "branch stop or length mismatch")
            require(branch["identity"]["reset_rgb_sha256"] == row["root_rgb_sha256"]
                    and branch["identity"]["goal_rgb_sha256"] == row["goal_rgb_sha256"],
                    "branch identity mismatch")
            points = branch["trajectory"]
            for i, point in enumerate(points):
                image_hash = verify_point(point)
                if i == 0:
                    require(image_hash == row["root_rgb_sha256"], "initial RGB mismatch")
                    for key in ("qpos", "qvel", "target_qpos"):
                        require(np.array_equal(np.asarray(point[key]), np.asarray(snapshot[key])),
                                "snapshot initial state mismatch")
                else:
                    require(np.array_equal(np.asarray(point["action"], dtype=np.float32), actual[i-1])
                            and point["terminated"] is False and point["truncated"] is False
                            and np.isfinite(point["reward"]), "trajectory action or stop mismatch")
                    require(np.array_equal(np.asarray(point["target_qpos"]), np.asarray(snapshot["target_qpos"])),
                            "trajectory target mismatch")
            prefix_signature = [(p["qpos"], p["qvel"], p["qpos_distance"]) for p in points[:6]]
            if arm in prefix_reference:
                require(prefix_signature == prefix_reference[arm], "same-prefix replay mismatch")
            else:
                prefix_reference[arm] = prefix_signature
            branch_results.append({"arm": arm, "tail_index": tail, "initial_qpos_distance": points[0]["qpos_distance"],
                                   "prefix_qpos_distance": points[5]["qpos_distance"],
                                   "final_qpos_distance": points[15]["qpos_distance"],
                                   "actions_sha256": branch["actions_sha256"]})
        pairs = [{"tail_index": i, "selected_minus_parameter_zero_final": (
                  branch_results[i]["final_qpos_distance"] - branch_results[i+5]["final_qpos_distance"])}
                 for i in range(5)]
        results.append({"seed": row["seed"], "planned": 10, "complete": 10,
                        "branches": branch_results, "paired_final_deltas": pairs})
    return {"status": "verified_saved_consistency", "model_or_physics_replayed": False,
            "p1_admission": False, "source_independence": False, "expected": 20, "complete": 20,
            "limits": ["self-reported actions/states, not independent physical replay",
                       "rendered RGB hashes and distances checked; historic simulator identity not certified",
                       "strict model selector null; no mechanism or equal-budget claim"], "roots": results}


def main():
    parser = argparse.ArgumentParser()
    for name in ("report", "plan", "model", "saved-audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    files = ((args.report, REPORT_SHA), (args.plan, PLAN_SHA), (args.model, MODEL_SHA), (args.saved_audit, SAVED_SHA))
    for path, expected in files:
        require(sha(path) == expected, "sealed file SHA mismatch: " + str(path))
    report, plan, model, saved = [json.loads(path.read_text()) for path, _ in files]
    source_checks = {path: sha(path) == digest for path, digest in plan["sources"].items()}
    result = verify(report, plan, model, saved)
    result.update(report_sha256=REPORT_SHA, plan_sha256=PLAN_SHA, model_sha256=MODEL_SHA,
                  saved_audit_sha256=SAVED_SHA, consumer_sha256=sha(__file__),
                  plan_source_current_hashes_match=source_checks)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    print(json.dumps({"status": result["status"], "output_sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
