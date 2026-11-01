"""Independent, saved-index-only historical pose audit; never a freshness certificate."""
import hashlib
import json
import math
import struct
from pathlib import Path

INDEX = Path(__file__).resolve().parents[2] / "results/codex_d2p_historical_exclusion_index_20261029_LOCAL01.json"
INDEX_SHA = "ac2544512d033ec14fd07808c3abe2dfd96676bb6b82ec89569f498f58256f98"


def identity(state):
    if not isinstance(state, list) or len(state) != 7 or any(
        type(x) not in (int, float) or not math.isfinite(x) for x in state
    ):
        raise ValueError("state must contain seven finite numbers")
    state = [float(x) for x in state]
    pose = state[:5]
    pose[4] %= 2 * math.pi
    pose = [0.0 if x == 0 else x for x in pose]
    digest = lambda xs: hashlib.sha256(struct.pack("<" + "d" * len(xs), *xs)).hexdigest()
    return digest(state), pose, digest(pose)


def same_pose(a, b):
    delta = a[4] - b[4]
    return all(abs(a[i] - b[i]) <= 1e-9 for i in range(4)) and abs(
        math.atan2(math.sin(delta), math.cos(delta))
    ) <= 1e-12


def audit(index):
    if index.get("schema") != "codex_d2p_historical_exclusion_index_20261029.v1":
        raise ValueError("unexpected index schema")
    rows = index["observed_prefix_records"]
    if len(rows) != 80 or index["counts"]["observed_prefix_records"] != 80:
        raise ValueError("historical denominator mismatch")
    if index["history_complete"] is not False or index["freshness_certified"] is not False:
        raise ValueError("unexpected authority")
    paths, poses = set(), []
    for row in rows:
        evidence = row["evidence"]
        key = (evidence["path"], evidence["pointer"])
        if key in paths or evidence["pointer"] != "/prefix_state":
            raise ValueError("duplicate or unexpected historical pointer")
        paths.add(key)
        state_hash, pose, pose_hash = identity(row["state7"])
        if (state_hash != row["root_state_sha256"] or pose != row["physical_pose"]
                or pose_hash != row["physical_pose_key"]):
            raise ValueError("historical identity mismatch")
        poses.append(pose)
    edges = [(i, j) for i in range(len(rows)) for j in range(i + 1, len(rows))
             if same_pose(poses[i], poses[j])]
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i, j in edges:
        parent[find(j)] = find(i)
    components = {}
    for i in range(len(rows)):
        components.setdefault(find(i), []).append(i)
    edge_set = set(edges)
    non_cliques = [group for group in components.values() if any(
        (i, j) not in edge_set for n, i in enumerate(group) for j in group[n + 1:]
    )]
    return {
        "schema": "codex_historical_pose_collision_audit_20261030.v1",
        "input_sha256": INDEX_SHA,
        "records": len(rows),
        "distinct_exact_pose_keys": len({r["physical_pose_key"] for r in rows}),
        "direct_pose_collision_pairs": len(edges),
        "connected_components": len(components),
        "component_sizes": sorted((len(g) for g in components.values()), reverse=True),
        "non_clique_components": non_cliques,
        "interpretation": "Connected components are transitive graph summaries, not physical roots or pairwise equivalence classes.",
        "history_complete": False,
        "freshness_certified": False,
        "source_run_authorized": False,
    }


def main():
    with INDEX.open("rb") as f:
        raw = f.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != INDEX_SHA:
        raise ValueError("index SHA or size mismatch")
    result = audit(json.loads(raw))
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
