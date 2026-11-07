"""Post-hoc audit of retention reachability and matched feedback decisions."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pc_tworoom_rgb_timed_prefix_feedback_20261107_codex.json"
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_RETENTION_DECISION_PATH_AUDIT_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_tworoom_retention_decision_path_audit_20261107_codex.json"


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1048576), b""):
            h.update(part)
    return h.hexdigest()


def expected_retained(decision, strategy):
    parent = np.asarray(decision["parent_cost"], dtype=float)
    probe = np.asarray(decision["probe_cost"], dtype=float)
    if parent.shape != (3,) or probe.shape != (3, 2) or not np.isfinite(probe).all():
        raise ValueError("unexpected program costs")
    if strategy == "parent":
        return np.argsort(parent, kind="stable")[:2].tolist()
    order = np.argsort(probe.min(axis=1), kind="stable")
    if strategy == "lookahead":
        return order[:2].tolist()
    vectors = probe - probe.mean(axis=1, keepdims=True)
    if strategy == "shuffled":
        rng = np.random.default_rng(decision["strategy_seed"])
        vectors = np.stack([v[rng.permutation(2)] for v in vectors])
    if strategy not in ("vector", "shuffled"):
        raise ValueError("unknown strategy")
    first = int(order[0])
    distances = np.square(vectors - vectors[first]).sum(axis=1)
    distances[first] = -np.inf
    return [first, int(np.argmax(distances))]


def validate_program(decision, strategy):
    if expected_retained(decision, strategy) != decision["retained"]:
        raise ValueError("retention mismatch")
    values = [(name, index, float(value))
              for name, array in (("parent", decision["parent_cost"]),
                                  ("probe", np.asarray(decision["probe_cost"]).reshape(-1)),
                                  ("tail", decision["tail_cost"]))
              for index, value in enumerate(array)]
    choice = decision["choice"]
    best = min(v[2] for v in values)
    matches = [v for v in values if v[0] == choice["source"] and v[1] == choice["index"]]
    if len(matches) != 1 or matches[0][2] != choice["cost"] or choice["cost"] != best:
        raise ValueError("selected candidate is not recorded minimum")


def match_key(decision):
    receipts = decision["receipts"]
    if len(receipts) < 2:
        raise ValueError("missing scoring receipts")
    first = receipts[0]["inputs"]
    second = receipts[1]["inputs"]
    return (decision["remaining_steps"], decision["prefix_steps"], decision["candidate_seed"],
            decision["old_tail_sha256"], *(first[key]["sha256"] for key in ("pixels", "goal", "action")),
            first["candidate_actions"]["sha256"], second["candidate_actions"]["sha256"])


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = json.loads(SOURCE.read_text())
    counts = Counter()
    matched = Counter()
    detailed = []
    for row in data["rows"]:
        for strategy, arm in row["arms"].items():
            for d in arm["decisions"]:
                clock = d["remaining_steps"]
                counts[f"{strategy}/{clock}/{d['mode']}"] += 1
                if d["mode"] == "program":
                    validate_program(d, strategy)
                    counts[f"{strategy}/{clock}/choice_{d['choice']['source']}"] += 1
                    centered = np.asarray(d["probe_cost"], dtype=float)
                    centered -= centered.mean(axis=1, keepdims=True)
                    if np.any(np.abs(centered) > 1e-10):
                        counts[f"{strategy}/{clock}/descriptor_nonzero"] += 1
                elif d["mode"] != "direct_fallback":
                    raise ValueError("unknown mode")
        vector = row["arms"]["vector"]["decisions"]
        lookahead = row["arms"]["lookahead"]["decisions"]
        for v, l in zip(vector, lookahead):
            if v["iteration"] != l["iteration"] or v["mode"] != "program" or l["mode"] != "program":
                continue
            if match_key(v) != match_key(l):
                continue
            if v["parent_cost"] != l["parent_cost"] or v["probe_cost"] != l["probe_cost"]:
                raise ValueError("matched receipts but differing common scores")
            different_retained = set(v["retained"]) != set(l["retained"])
            different_action = v["actions_sha256"] != l["actions_sha256"]
            different_packet = v["dispatched"] != l["dispatched"]
            matched["pairs"] += 1
            matched["different_retained"] += int(different_retained)
            matched["different_actions"] += int(different_action)
            matched["different_dispatched"] += int(different_packet)
            if different_retained:
                matched["different_retained_and_actions"] += int(different_action)
                matched["different_retained_and_dispatched"] += int(different_packet)
            detailed.append({"root": row["root"], "iteration": v["iteration"],
                             "remaining_steps": v["remaining_steps"],
                             "different_retained": different_retained,
                             "different_actions": different_action,
                             "different_dispatched": different_packet,
                             "vector_choice": v["choice"]["source"],
                             "lookahead_choice": l["choice"]["source"]})
    output = {"scope": "post-hoc exposed RGB development trajectories; not a causal intervention",
              "sha256": {str(p): file_hash(p) for p in (SOURCE, PROTOCOL, Path(__file__))},
              "counts": dict(sorted(counts.items())), "matched": dict(matched), "matched_pairs": detailed}
    with OUTPUT.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps({"counts": output["counts"], "matched": output["matched"]}, indent=2))


if __name__ == "__main__":
    main()
