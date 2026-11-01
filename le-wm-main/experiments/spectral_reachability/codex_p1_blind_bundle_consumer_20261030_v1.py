"""Strict standalone consumer for a self-contained, read-only planner input bundle."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path

import numpy as np

NAME = "codex_p1_blind_bundle_20261030_v1"
TASK_KEYS = set("environment task_contract_sha256 root_context root_raw_action_history goal projection model_sha256 model_config_sha256 metric_source_sha256 goal_source_sha256 representation latent_dim horizon packet_steps control_dim action_mean action_std packet_logit_cap feedback".split())
CONTEXT_KEYS = set("layout context_length latent_dim action_dim latent_history action_history".split())
ARMS = ("projected_ascent", "outward_free_tangent")
ROLES = ("selected_prefix", "zero_prefix")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def keys(value, expected):
    need(type(value) is dict and set(value) == expected, "unexpected or missing fields")


def hashvalue(value):
    need(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "SHA-256")


def arr(value, shape):
    a = np.asarray(value)
    need(a.shape == shape and a.dtype.kind in "fi" and np.isfinite(a).all(), "finite array shape")
    return a


def decode(value, limit):
    need(type(value) is str and len(value) <= 4*((limit+2)//3), "base64 size")
    try:
        data = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise ValueError("invalid base64") from exc
    need(len(data) <= limit and base64.b64encode(data).decode() == value, "canonical base64")
    return data


def task_check(t):
    keys(t, TASK_KEYS)
    need(t["environment"] == "swm/PushT-v1" and t["representation"] == "native_projected_cls" and
         t["feedback"] == "open_loop_model_only", "task semantics")
    for key, expected in (("latent_dim", 192), ("horizon", 2), ("packet_steps", 5), ("control_dim", 2)):
        need(type(t[key]) is int and t[key] == expected, "task dimensions")
    for key in ("task_contract_sha256", "model_sha256", "model_config_sha256",
                "metric_source_sha256", "goal_source_sha256"):
        hashvalue(t[key])
    c = t["root_context"]
    keys(c, CONTEXT_KEYS)
    need(c["layout"] == "states_with_between_state_actions_v1" and
         type(c["context_length"]) is int and c["context_length"] == 3 and
         type(c["latent_dim"]) is int and c["latent_dim"] == 192 and
         type(c["action_dim"]) is int and c["action_dim"] == 10, "context ABI")
    arr(c["latent_history"], (3, 192))
    hist = arr(c["action_history"], (2, 10))
    arr(t["goal"], (192,))
    need(np.array_equal(arr(t["projection"], (192, 192)), np.eye(192)), "identity Q")
    mean = arr(t["action_mean"], (2,)).astype(np.float32)
    std = arr(t["action_std"], (2,)).astype(np.float32)
    need((std > 0).all(), "normalization")
    raw = arr(t["root_raw_action_history"], (10, 2))
    need((np.abs(raw) <= 1).all() and np.array_equal(raw, raw.astype(np.float32)), "raw history")
    tokens = (raw.astype(np.float32).reshape(2, 10)-np.tile(mean, 5))/np.tile(std, 5)
    need(np.array_equal(tokens, hist), "raw history normalization")
    need(type(t["packet_logit_cap"]) in (float, int) and
         t["packet_logit_cap"] == float(.75*math.sqrt(5)), "action chart cap")


def validate(bundle):
    keys(bundle, {"schema", "source_prepare_sha256", "model_before", "model_files", "rows",
                  "formal_roster", "producer_sha256", "consumer_sha256",
                  "physical_authorized", "P1_admission"})
    need(bundle["schema"] == NAME and bundle["physical_authorized"] is False and
         bundle["P1_admission"] is False, "not an authorization")
    hashvalue(bundle["source_prepare_sha256"])
    hashvalue(bundle["producer_sha256"])
    hashvalue(bundle["consumer_sha256"])
    keys(bundle["model_before"], {"original_state_dict", "all_buffers"})
    keys(bundle["model_files"], {"weights.pt", "config.json"})
    for h in (*bundle["model_before"].values(), *bundle["model_files"].values()):
        hashvalue(h)
    rows = bundle["rows"]
    need(type(rows) is list and len(rows) == 16, "fixed 16")
    roster = []
    for i, row in enumerate(rows):
        keys(row, {"index", "slot", "root_slot", "condition", "arm", "role", "task_id",
                   "task", "prefix", "history_rgb", "goal_rgb", "cp5_rgb"})
        slot = i//4
        need(type(row["index"]) is int and type(row["slot"]) is int and type(row["root_slot"]) is int and
             (row["index"], row["slot"], row["root_slot"], row["condition"], row["arm"], row["role"]) ==
             (i, slot, slot//2, ("off", "on")[slot%2], ARMS[(i%4)//2], ROLES[i%2]), "row order")
        task = row["task"]
        task_check(task)
        need(digest(task) == row["task_id"], "task identity")
        need(task["model_sha256"] == bundle["model_files"]["weights.pt"] and
             task["model_config_sha256"] == bundle["model_files"]["config.json"], "model identity")
        p = arr(row["prefix"], (5, 2))
        need((np.abs(p) <= 1).all() and np.array_equal(p, p.astype(np.float32)), "prefix")
        if row["role"] == "zero_prefix":
            need(not np.any(p), "zero prefix")
        need(type(row["history_rgb"]) is list and len(row["history_rgb"]) == 3, "history RGB count")
        for frame in [*row["history_rgb"], row["goal_rgb"]]:
            keys(frame, {"rgb_base64", "pixel_sha256"})
            hashvalue(frame["pixel_sha256"])
            pixels = decode(frame["rgb_base64"], 224*224*3)
            need(len(pixels) == 224*224*3 and sha(pixels) == frame["pixel_sha256"], "RGB bytes")
        need(row["goal_rgb"]["pixel_sha256"] == task["goal_source_sha256"], "goal RGB binding")
        image = row["cp5_rgb"]
        keys(image, {"rgb_base64", "source_png_sha256", "pixel_sha256"})
        hashvalue(image["source_png_sha256"])
        hashvalue(image["pixel_sha256"])
        data = decode(image["rgb_base64"], 224*224*3)
        need(len(data) == 224*224*3 and sha(data) == image["pixel_sha256"], "cp5 RGB bytes")
        if i % 2 == 0:
            roster.extend([dict(slot=slot, arm=row["arm"], role=role, row_index=idx,
                                status="source_verified" if idx is not None else "unknown")
                           for role, idx in ((ROLES[0], i), (ROLES[1], i+1), ("greedy_prefix", None))])
        else:
            previous = rows[i-1]
            need(all(row[k] == previous[k] for k in ("task_id", "task", "history_rgb", "goal_rgb")),
                 "same arm baseline input")
    need(bundle["formal_roster"] == roster, "formal 24-condition roster")
    return dict(status="bundle_integrity_only_not_authorized", rows=16, formal_conditions=24,
                greedy_unknown=8, physical_authorized=False)


def unique(pairs):
    result = {}
    for k, v in pairs:
        need(k not in result, "duplicate JSON key")
        result[k] = v
    return result


def read_verified(path, expected):
    hashvalue(expected)
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32*1024**2, "bundle file")
    data = path.read_bytes()
    need(sha(data) == expected, "external bundle SHA mismatch")
    value = json.loads(data, object_pairs_hook=unique, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    status = validate(value)
    return value, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    _, status = read_verified(args.bundle, args.sha256)
    print(json.dumps(status))


if __name__ == "__main__":
    main()
