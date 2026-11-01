"""Compare explicit candidate identity metadata against bounded known history.

This does not select roots, validate candidate provenance, certify freshness,
or authorize source execution. No planner outcome fields are accepted.
"""
import argparse
import json
import math
import os
from pathlib import Path

from experiments.spectral_reachability import p1_known_history_exclusion_20261101_hypatia as base
from experiments.spectral_reachability import p1_coherent_development_identity_20261101_hypatia as coherent

NAME = "p1_explicit_collision_check_20261101_hypatia"
HISTORY_PATH = base.ROOT / "results/p1_known_history_exclusion_20261101_hypatia_LOCAL01.json"
HISTORY_SHA = "c89b869189c45c63bb86b2afa09f5ba67f079d0e4ebed84396fc0acd70345136"
COHERENT_PATH = base.ROOT / "results/p1_coherent_development_identity_20261101_hypatia_LOCAL01.json"
COHERENT_SHA = "e084cd3d1ed3e0fd8e216ddc776b238ce8e5b829a165d55d153c789ba69a23f7"
LIMIT = 1024


def validate_candidates(value):
    base.require(type(value) is dict and set(value) == {"schema", "candidates"}
                 and value["schema"] == NAME + ".candidates.v1", "candidate schema")
    rows = value["candidates"]
    base.require(type(rows) is list and 0 < len(rows) <= LIMIT, "candidate count")
    ids = set()
    for row in rows:
        base.require(type(row) is dict and set(row) == {"candidate_id", "root_seed", "reset_seed", "state7"},
                     "exact candidate identity fields only")
        label = row["candidate_id"]
        base.require(type(label) is str and 0 < len(label) <= 128 and label not in ids, "unique candidate id")
        ids.add(label)
        for field in ("root_seed", "reset_seed"):
            value = row[field]
            base.require(value is None or (type(value) is int and 0 <= value < 2**63), "invalid seed")
        if row["state7"] is not None:
            base.identity(row["state7"])
        base.require(any(row[k] is not None for k in ("root_seed", "reset_seed", "state7")),
                     "empty candidate identity")
    return rows


def same_pose(a, b):
    d = a[4] - b[4]
    return all(abs(a[i] - b[i]) <= 1e-9 for i in range(4)) and abs(
        math.atan2(math.sin(d), math.cos(d))) <= 1e-12


def compare(candidates, history, supplement):
    rows = validate_candidates(candidates)
    base.require(history["schema"] == base.NAME + ".v1" and history["history_complete"] is False
                 and history["freshness_certified"] is False, "history authority")
    base.require(supplement["schema"] == coherent.NAME + ".v1"
                 and supplement["freshness_certified"] is False, "coherent authority")
    seed_refs = {}
    for i, row in enumerate(history["seed_projections"]):
        seed_refs.setdefault(row["seed"], []).append("/seed_projections/" + str(i))
    for name in ("inventory8_root_seeds", "inventory8_reset_seeds"):
        for i, seed in enumerate(history[name]):
            seed_refs.setdefault(seed, []).append("/" + name + "/" + str(i))
    seed_refs.setdefault(supplement["record"]["reset_seed"], []).append("coherent:/record/reset_seed")
    identities = [("history:/historical_identities/" + str(i), x["state7"])
                  for i, x in enumerate(history["historical_identities"])]
    identities += [("coherent:/record/state7", supplement["record"]["state7"]),
                   ("coherent:/record/requested_reset_state7", supplement["record"]["requested_reset_state7"])]
    historical = [(ref, base.identity(state)) for ref, state in identities]
    candidate_ids = [base.identity(r["state7"]) if r["state7"] is not None else None for r in rows]
    results = []
    for i, row in enumerate(rows):
        hits = []
        for field in ("root_seed", "reset_seed"):
            seed = row[field]
            if seed is None:
                continue
            for ref in seed_refs.get(seed, []):
                hits.append({"candidate_field": field, "kind": "known_seed", "value": seed, "reference": ref})
            for j, (lo, hi) in enumerate(history["reserved_half_open_reset_spans"]):
                if lo <= seed < hi:
                    hits.append({"candidate_field": field, "kind": "reserved_reset_span", "value": seed,
                                 "span": [lo, hi], "reference": "history:/reserved_half_open_reset_spans/" + str(j)})
        ident = candidate_ids[i]
        if ident is not None:
            for ref, other in historical:
                if ident["state_sha256"] == other["state_sha256"]:
                    hits.append({"candidate_field": "state7", "kind": "exact_state7_bits", "reference": ref})
                if same_pose(ident["pose"], other["pose"]):
                    hits.append({"candidate_field": "state7", "kind": "pose_numeric_alias", "reference": ref})
        peers = []
        for j, other in enumerate(rows):
            if i == j:
                continue
            for field in ("root_seed", "reset_seed"):
                seed = row[field]
                if seed is not None:
                    for other_field in ("root_seed", "reset_seed"):
                        if seed == other[other_field]:
                            peers.append({"candidate_field": field, "kind": "candidate_seed_collision",
                                          "other_id": other["candidate_id"], "other_field": other_field})
            if ident is not None and candidate_ids[j] is not None:
                if ident["state_sha256"] == candidate_ids[j]["state_sha256"]:
                    peers.append({"candidate_field": "state7", "kind": "candidate_exact_state7_bits",
                                  "other_id": other["candidate_id"]})
                if same_pose(ident["pose"], candidate_ids[j]["pose"]):
                    peers.append({"candidate_field": "state7", "kind": "candidate_pose_numeric_alias",
                                  "other_id": other["candidate_id"]})
        missing = [field for field in ("root_seed", "reset_seed", "state7") if row[field] is None]
        results.append({"candidate_id": row["candidate_id"], "provided_identity": row,
                        "computed_identity": ident, "known_history_hits": hits, "batch_hits": peers,
                        "missing_identity_fields": missing,
                        "collision_observed": bool(hits or peers),
                        "development_history_independence": "unknown",
                        "training_disjoint": "unknown", "freshness": "unknown"})
    return results


def generate(path, expected):
    path = Path(path)
    raw = base.read_bound(path, expected)
    candidate = base.decode(raw)
    base.verify(HISTORY_PATH, HISTORY_SHA)
    coherent.verify(COHERENT_PATH, COHERENT_SHA)
    history = base.decode(base.read_bound(HISTORY_PATH, HISTORY_SHA))
    supplement = base.decode(base.read_bound(COHERENT_PATH, COHERENT_SHA))
    source_files = [Path(__file__).resolve(), Path(base.__file__).resolve(), Path(coherent.__file__).resolve()]
    source_hashes = {str(p): base.sha(p.read_bytes()) for p in source_files}
    result = {
        "schema": NAME + ".v1",
        "candidate_input": {"path": str(path), "sha256": expected},
        "history": {"path": str(HISTORY_PATH), "sha256": HISTORY_SHA},
        "coherent_development_root": {"path": str(COHERENT_PATH), "sha256": COHERENT_SHA},
        "implementation_sha256": source_hashes,
        "rows": compare(candidate, history, supplement),
        "scope": "bounded known seed/state/pose and within-batch collisions; no root selection",
        "limitations": [
            "incomplete historical reset and development coverage; no collision is not independence",
            "candidate file is a metadata claim; actual source/identity mapping is not authenticated",
            "shared placement reset seeds are conservative conflicts, not proof of the same physical root",
            "training membership, statistical independence, geometry and complete history are unknown",
            "prefix-state matches and reserved intervals do not establish sampling-root equality",
        ],
        "history_complete": False, "development_disjoint_certified": False,
        "training_disjoint_certified": False, "freshness_certified": False,
        "source_run_authorized": False, "P1_admission": False,
        "selected_candidates": 0, "neural_calls": 0, "physical_calls": 0,
    }
    base.read_bound(path, expected)
    base.verify(HISTORY_PATH, HISTORY_SHA)
    coherent.verify(COHERENT_PATH, COHERENT_SHA)
    base.require(all(base.sha(Path(p).read_bytes()) == h for p, h in source_hashes.items()),
                 "implementation changed")
    return result


def create(candidate, candidate_sha, output):
    output = Path(output)
    base.require(output.is_absolute() and output.parent.resolve() == output.parent
                 and output.name.startswith(NAME + "_") and output.suffix == ".json",
                 "exclusive output namespace")
    data = base.encode(generate(candidate, candidate_sha))
    with output.open("xb") as stream:
        base.require(stream.write(data) == len(data), "short write")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return base.sha(data)


def verify(output, digest):
    data = base.read_bound(Path(output), digest)
    value = base.decode(data)
    ref = value["candidate_input"]
    base.require(value == generate(ref["path"], ref["sha256"]), "report or input drift")
    return {"status": "verified_bounded_collision_report", "sha256": digest,
            "freshness_certified": False, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("--candidates", required=True)
    check.add_argument("--candidates-sha256", required=True)
    check.add_argument("--output", required=True)
    validate = sub.add_parser("verify")
    validate.add_argument("--report", required=True)
    validate.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if args.command == "check":
        print(json.dumps({"output": args.output, "sha256": create(args.candidates, args.candidates_sha256, args.output),
                          "freshness_certified": False, "P1_admission": False}))
    else:
        print(json.dumps(verify(args.report, args.sha256)))


if __name__ == "__main__":
    main()
