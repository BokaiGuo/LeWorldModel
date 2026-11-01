"""CPU-only bounded historical identity projection; not a freshness certificate."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]
NAME = "p1_known_history_exclusion_20261101_hypatia"
INPUTS = {
    "prefix_index": ("results/codex_d2p_historical_exclusion_index_20261029_LOCAL01.json",
                     "ac2544512d033ec14fd07808c3abe2dfd96676bb6b82ec89569f498f58256f98"),
    "resolution618": ("results/root_exclusion_authority_resolution_20261027_v1_inventory618_MAIN01.json",
                      "e2afed12a1d8f788ad16d0223482377b5f54d281e67739766475228b53ac2591"),
    "inventory8": ("results/moving_root_exclusion_manifest_20261027_v1_inventory8_main01.json",
                   "a17bc732c9a021a4ca81a7e4da573c752ad1f66eafe08b1a3dad2b64464d7060"),
}
MAX_BYTES = 32 * 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_bound(path, expected):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file() and not path.is_symlink(),
            "canonical regular file required")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(MAX_BYTES + 1)
        after = os.fstat(stream.fileno())
    require(len(raw) <= MAX_BYTES, "input size cap")
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "input changed")
    require(hashlib.sha256(raw).hexdigest() == expected, "input SHA mismatch")
    return raw


def decode(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("invalid JSON constant " + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(state):
    require(type(state) is list and len(state) == 7 and all(
        type(x) in (int, float) and math.isfinite(x) for x in state), "finite state7 required")
    values = [float(x) for x in state]
    pose = values[:5]
    pose[4] %= 2 * math.pi
    pose = [0.0 if x == 0 else x for x in pose]
    return {"state_sha256": sha(struct.pack("<7d", *values)), "pose": pose,
            "pose_sha256": sha(struct.pack("<5d", *pose))}


def build(index, resolution, inventory, bindings, source_sha):
    require(index["schema"] == "codex_d2p_historical_exclusion_index_20261029.v1"
            and index["history_complete"] is False and index["freshness_certified"] is False,
            "historical index authority")
    rows = index["observed_prefix_records"]
    require(len(rows) == 80 and index["counts"]["observed_prefix_records"] == 80, "80 prefix denominator")
    require(resolution["schema"] == "root_exclusion_authority_resolution_20261027_v1.resolution_request"
            and resolution["freshness_certified"] is False and resolution["P1_admission"] is False,
            "resolution authority")
    require(inventory["schema"] == "moving_fresh_source_exclusions.v1"
            and inventory["freshness_certified"] is False and inventory["P1_admission"] is False
            and inventory["source_run_preconditions_met"] is False, "inventory authority")
    poses = []
    seen = set()
    for i, row in enumerate(rows):
        ev = row["evidence"]
        key = (ev["path"], ev["pointer"])
        require(key not in seen and ev["pointer"] == "/prefix_state", "prefix identity pointer")
        seen.add(key)
        ident = identity(row["state7"])
        require(ident["state_sha256"] == row["root_state_sha256"]
                and ident["pose_sha256"] == row["physical_pose_key"]
                and ident["pose"] == row["physical_pose"], "historical identity drift")
        poses.append({"kind": "observed_prefix_not_sampling_root", "state7": row["state7"],
                      "identity": ident, "index_pointer": f"/observed_prefix_records/{i}",
                      "original_evidence": ev, "unresolved": row["unresolved"]})
    for i, bridge in enumerate(resolution["physical_identity_bridges"]):
        meta = bridge["metadata"]
        ident = identity(meta["root_state"])
        require(ident["state_sha256"] == meta["root_state_sha256"], "bridge state mismatch")
        poses.append({"kind": "saved_bridge_metadata_not_new_root", "state7": meta["root_state"],
                      "identity": ident, "index_pointer": f"/physical_identity_bridges/{i}/metadata",
                      "record_id": meta["record_id"], "maps_to": meta["maps_to"],
                      "original_pose_sha256": meta["physical_pose_key"],
                      "unresolved": ["bridge inherited; journal and pixels not revalidated here"]})
    seeds = resolution["stable_identity_projections"]["seed_projections"]
    require(all(type(r["seed"]) is int and r["kind"] in
                ("root_seed", "reset_seed", "calibration_root_seed") for r in seeds),
            "invalid seed projection")
    spans = inventory["reserved_reset_spans"]
    require(all(type(s) is list and len(s) == 2 and all(type(x) is int for x in s)
                and s[0] < s[1] for s in spans), "invalid half-open span")
    return {
        "schema": NAME + ".v1", "inputs": bindings, "implementation_sha256": source_sha,
        "purpose": "bounded known-history exclusion evidence, never a candidate selection",
        "historical_identities": poses,
        "seed_projections": seeds,
        "inventory8_root_seeds": inventory["used_or_reserved_root_seeds"],
        "inventory8_reset_seeds": inventory["used_or_reserved_reset_seeds"],
        "inventory8_seed_authority": inventory["seed_authority"],
        "reserved_half_open_reset_spans": spans,
        "root_goal_groups": resolution["stable_identity_projections"]["root_goal_groups"],
        "inventory8_root_goal_sha256s": inventory["root_goal_sha256s"],
        "source_summary": resolution["summary"],
        "authority_requests": [{k: r[k] for k in ("family", "status", "remaining")}
                               for r in resolution["authority_requests"]],
        "known_gaps": [
            "10 historical dispatches lack an exhaustive reset-attempt ledger or proved execution closure",
            "index is a partial projection; failures, unselected, later development and aliases may be missing",
            "2026-11-01 coherent development root identity is not separately bound in these input snapshots",
            "bridge journals and original referenced files are not revalidated by this projection",
            "seed and root-goal identity do not prove physical root difference",
            "training membership and actual checkpoint training provenance unknown",
            "source distribution, statistical independence, geometry eligibility and full RGB/history unknown",
        ],
        "identity_comparison": {"xy_tolerance": 1e-9, "wrapped_angle_tolerance": 1e-12,
                                "note": "protocol tolerance is a collision warning, not evidence of independence"},
        "history_complete": False, "development_disjoint_certified": False,
        "training_disjoint_certified": False, "freshness_certified": False,
        "source_run_authorized": False, "P1_admission": False,
        "new_candidates_selected": 0, "neural_calls": 0, "physical_calls": 0,
    }


def generate():
    values, bindings = {}, {}
    for name, (rel, expected) in INPUTS.items():
        path = ROOT / rel
        values[name] = decode(read_bound(path, expected))
        bindings[name] = {"path": str(path), "sha256": expected}
    source = Path(__file__).resolve()
    source_sha = sha(source.read_bytes())
    result = build(values["prefix_index"], values["resolution618"], values["inventory8"], bindings, source_sha)
    for name, (rel, expected) in INPUTS.items():
        read_bound(ROOT / rel, expected)
    require(sha(source.read_bytes()) == source_sha, "implementation changed")
    return result


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def create(path):
    path = Path(path)
    require(path.is_absolute() and path.parent.resolve() == path.parent
            and path.name.startswith(NAME + "_") and path.suffix == ".json", "exclusive output namespace")
    data = encode(generate())
    with path.open("xb") as stream:
        require(stream.write(data) == len(data), "short write")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return sha(data)


def verify(path, expected):
    original = read_bound(Path(path), expected)
    require(decode(original) == generate(), "manifest or source drift")
    return {"status": "verified_known_history_projection", "sha256": expected,
            "freshness_certified": False, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("create")
    c.add_argument("--output", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--manifest", required=True)
    v.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if args.command == "create":
        print(json.dumps({"output": args.output, "sha256": create(args.output), "P1_admission": False}))
    else:
        print(json.dumps(verify(args.manifest, args.sha256)))


if __name__ == "__main__":
    main()
