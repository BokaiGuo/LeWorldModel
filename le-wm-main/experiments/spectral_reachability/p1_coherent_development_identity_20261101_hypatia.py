"""Append-only CPU projection of the already-used coherent development root."""
import argparse
import json
import math
import os
from pathlib import Path

from experiments.spectral_reachability import p1_known_history_exclusion_20261101_hypatia as base

NAME = "p1_coherent_development_identity_20261101_hypatia"
ROOT = base.ROOT
INPUTS = {
    "history": ("results/p1_known_history_exclusion_20261101_hypatia_LOCAL01.json",
                "c89b869189c45c63bb86b2afa09f5ba67f079d0e4ebed84396fc0acd70345136"),
    "witness": ("results/p1_centered_witness_20261101_main_RUN_MAIN01/report.json",
                "1d7eaa958442e2609a7e5871b68c159021cb20c43603fdd70d9749524d5c6719"),
    "target": ("results/p1_centered_target_cuda_20261101_main_MAIN01.json",
               "801c3b089cb9af71901c638b9197b25aa84573e0befefec8abe8a3c068d62ef3"),
}


def same_pose(a, b):
    delta = a[4] - b[4]
    return all(abs(a[i] - b[i]) <= 1e-9 for i in range(4)) and abs(
        math.atan2(math.sin(delta), math.cos(delta))) <= 1e-12


def build(values, bindings, source_sha, base_sha):
    history, witness, target = (values[k] for k in ("history", "witness", "target"))
    base.require(history["schema"] == base.NAME + ".v1" and history["freshness_certified"] is False,
                 "history schema or authority")
    base.require(history["implementation_sha256"] == base_sha, "base implementation drift")
    base.require(len(witness["rows"]) >= 1 and witness["rows"][0]["step"] == 0
                 and len(target["rows"]) == 1, "source root pointer")
    root = witness["rows"][0]["state"]
    row = target["rows"][0]
    base.require(root == row["root_state"] and row["translated_known_root"] is True, "root identity mismatch")
    history_pixels = [r["pixel_sha256"] for r in witness["history"]]
    base.require([r["step"] for r in witness["history"]] == [0, 5, 10]
                 and history_pixels == row["history_pixel_sha256"], "history identity mismatch")
    witness_path = bindings["witness"]["path"]
    base.require(target["sources"].get(witness_path) == bindings["witness"]["sha256"],
                 "target does not bind witness")
    ident = base.identity(root)
    collisions = [{"kind": r["kind"], "pointer": r["index_pointer"]}
                  for r in history["historical_identities"] if same_pose(ident["pose"], r["identity"]["pose"])]
    return {
        "schema": NAME + ".v1", "inputs": bindings,
        "implementation_sha256": source_sha, "base_implementation_sha256": base_sha,
        "record": {"role": "known_development_root_exclude_not_candidate",
                   "state7": root, "identity": ident, "history_pixel_sha256": history_pixels,
                   "reset_seed": witness["reset_seed"],
                   "requested_reset_state7": witness["requested_reset"],
                   "identity_pointer": "/rows/0/state", "target_pointer": "/rows/0/root_state",
                   "pose_collisions_in_prior_partial_inventory": collisions},
        "limitations": [
            "target copies witness; agreement is not independent replication",
            "source bytes are pinned but frames and simulator provenance are not reverified here",
            "translation of a known development root is not a fresh independent root",
            "absence of collision in this partial inventory does not prove independence",
            "other recent development roots and exhaustive reset history remain unresolved",
            "training membership and actual checkpoint training provenance remain unknown",
        ],
        "history_complete": False, "development_disjoint_certified": False,
        "training_disjoint_certified": False, "freshness_certified": False,
        "source_run_authorized": False, "P1_admission": False,
        "new_candidates_selected": 0, "neural_calls": 0, "physical_calls": 0,
    }


def generate():
    bindings = {k: {"path": str(ROOT / p), "sha256": h} for k, (p, h) in INPUTS.items()}
    values = {k: base.decode(base.read_bound(Path(v["path"]), v["sha256"])) for k, v in bindings.items()}
    base_path = Path(base.__file__).resolve()
    implementation = Path(__file__).resolve()
    source_sha, base_sha = base.sha(implementation.read_bytes()), base.sha(base_path.read_bytes())
    result = build(values, bindings, source_sha, base_sha)
    for binding in bindings.values():
        base.read_bound(Path(binding["path"]), binding["sha256"])
    base.require(source_sha == base.sha(implementation.read_bytes()) and base_sha == base.sha(base_path.read_bytes()),
                 "implementation changed")
    return result


def create(path):
    path = Path(path)
    base.require(path.is_absolute() and path.parent.resolve() == path.parent
                 and path.name.startswith(NAME + "_") and path.suffix == ".json", "exclusive output namespace")
    raw = base.encode(generate())
    with path.open("xb") as stream:
        base.require(stream.write(raw) == len(raw), "short write")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return base.sha(raw)


def verify(path, digest):
    raw = base.read_bound(Path(path), digest)
    base.require(base.decode(raw) == generate(), "supplement or source drift")
    return {"status": "verified_known_development_identity", "sha256": digest,
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
