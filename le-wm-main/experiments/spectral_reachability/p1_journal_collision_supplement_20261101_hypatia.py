"""Supplement bounded collision checks with a pinned historical reset inventory.

This is an exclusion aid, not source authentication or freshness certification.
"""
import argparse
import json
import os
from pathlib import Path

from experiments.spectral_reachability import p1_explicit_collision_check_20261101_hypatia as prior

base = prior.base
NAME = "p1_journal_collision_supplement_20261101_hypatia"
INVENTORY = base.ROOT / "results/codex_p1_original_reset_inventory_20261101_exclusive.json"
INVENTORY_SHA = "68218ed8b88b11f49c404e94d93536a185d52093c0a735e90a61d3c37adde7c5"


def supplement(candidates, inventory):
    rows = prior.validate_candidates(candidates)
    base.require(inventory["history_complete"] is False and inventory["freshness_certified"] is False,
                 "inventory authority")
    events = inventory["events"]
    base.require(type(events) is list and len(events) == 16, "inventory event count")
    answer = []
    for row in rows:
        hits = []
        ident = base.identity(row["state7"]) if row["state7"] is not None else None
        for i, event in enumerate(events):
            for field in ("root_seed", "reset_seed"):
                if row[field] is not None and row[field] == event["seed"]:
                    hits.append({"field": field, "kind": "recorded_reset_seed",
                                 "reference": f"/events/{i}", "operation": event["operation"]})
            if ident is not None:
                other = base.identity(event["state"])
                if ident["state_sha256"] == other["state_sha256"]:
                    hits.append({"field": "state7", "kind": "exact_state7_bits", "reference": f"/events/{i}"})
                if prior.same_pose(ident["pose"], other["pose"]):
                    hits.append({"field": "state7", "kind": "pose_numeric_alias", "reference": f"/events/{i}"})
        answer.append({"candidate_id": row["candidate_id"], "journal_hits": hits})
    return answer


def generate(candidate_path, candidate_sha):
    candidate_path = Path(candidate_path)
    candidates = base.decode(base.read_bound(candidate_path, candidate_sha))
    inventory = base.decode(base.read_bound(INVENTORY, INVENTORY_SHA))
    preceding = prior.generate(candidate_path, candidate_sha)
    hits = supplement(candidates, inventory)
    paths = [Path(__file__).resolve(), INVENTORY]
    hashes = {str(p): base.sha(p.read_bytes()) for p in paths}
    base.require(hashes[str(INVENTORY)] == INVENTORY_SHA, "inventory changed")
    base.read_bound(candidate_path, candidate_sha)
    return {
        "schema": NAME + ".v1", "candidate_input": preceding["candidate_input"],
        "preceding_bounded_check": preceding, "journal_inventory_sha256": INVENTORY_SHA,
        "supplement_implementation_sha256": hashes[str(paths[0])], "journal_rows": hits,
        "limitations": [
            "journal covers only one historical sourcepack, not all development or training",
            "seed 0 placement resets are conservative hits, not proof of equal physical roots",
            "candidate provenance and historical coverage remain unverified",
        ],
        "development_history_independence": "unknown", "training_disjoint": "unknown",
        "freshness": "unknown", "freshness_certified": False, "P1_admission": False,
        "source_run_authorized": False, "selected_candidates": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--candidates-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    base.require(out.is_absolute() and out.parent.resolve() == out.parent
                 and out.name.startswith(NAME + "_") and out.suffix == ".json", "output namespace")
    data = base.encode(generate(args.candidates, args.candidates_sha256))
    with out.open("xb") as stream:
        base.require(stream.write(data) == len(data), "short write")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"output": str(out), "sha256": base.sha(data), "freshness_certified": False}))


if __name__ == "__main__":
    main()
