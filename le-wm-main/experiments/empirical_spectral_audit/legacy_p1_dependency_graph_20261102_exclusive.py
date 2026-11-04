"""Inventory declared identity and reuse for the fixed 240 legacy P1 records.

Edges are evidence of shared identifiers, not proof of independent components.
No inference, training, test re-selection, or gate modification is performed.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import itertools
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/legacy_p1_failure_attribution_20261102_exclusive_RUN01/report.json"
SOURCE_SHA = "64e5ba46abf2aec9c9c8d26194faaab557a5ff7bb33c3b3cf96aaed6a693e287"
OUT = ROOT / "results/legacy_p1_dependency_graph_20261102_exclusive_RUN01"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def resolve_path(value):
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts[0] == "le-wm-main":
        return ROOT.parent / path
    return ROOT / path


@lru_cache(maxsize=None)
def load(path):
    try:
        data = path.read_bytes()
        return {"path": str(path), "sha256": sha(data), "value": json.loads(data)}
    except (OSError, ValueError) as exc:
        return {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}


def ids(values):
    if not isinstance(values, list):
        return None
    result = []
    for value in values:
        if value is None or isinstance(value, (dict, list, bool)):
            return None
        result.append(str(value))
    return sorted(set(result))


def export_identity(path):
    if path is None:
        return {"state": "not_declared"}
    loaded = load(path)
    if "error" in loaded:
        return {"state": "unreadable", "path": str(path), "error": loaded["error"]}
    value = loaded["value"]
    split = value.get("split", {})
    split = split if isinstance(split, dict) else {}
    rows = value.get("enriched_records")
    rows = rows if isinstance(rows, list) else []
    row_ids = ids([r.get("episode_id") for r in rows if isinstance(r, dict)]) if rows else None
    declared_ids = ids(split.get("episode_ids"))
    return {"state": "loaded", "path": str(path), "sha256": loaded["sha256"],
            "dataset_sha256": value.get("dataset_sha256"),
            "base_frozen_export": value.get("protocol", {}).get("base_frozen_export"),
            "split_episode_ids": declared_ids, "row_episode_ids": row_ids,
            "split_row_identity_agree": declared_ids == row_ids if row_ids is not None and declared_ids is not None else None,
            "source_manifest": split.get("source_manifest"),
            "checkpoint_sha256_declared": value.get("checkpoint_sha256")}


def planning_identity(path):
    if path is None:
        return {"state": "not_declared"}
    loaded = load(path)
    if "error" in loaded:
        return {"state": "unreadable", "path": str(path), "error": loaded["error"]}
    value = loaded["value"]
    planning = value.get("planning_consequence", {})
    episodes = planning.get("episodes")
    protocol = planning.get("protocol", {})
    protocol = protocol if isinstance(protocol, dict) else {}
    episode_ids = ids([row.get("episode") for row in episodes if isinstance(row, dict)]) if isinstance(episodes, list) else None
    return {"state": "loaded", "path": str(path), "sha256": loaded["sha256"],
            "schema": value.get("schema"), "planning_seed": protocol.get("planning_seed"),
            "episode_ids": episode_ids, "episodes_declared_count": protocol.get("planning_episodes"),
            "metric_family": planning.get("metric_family"),
            "planning_split_declaration": value.get("protocol", {}).get("planning_split"),
            "root_ids": None,
            "identity_limit": "episode indices plus planning seed are declarations, not globally authenticated roots"}


def lineage(checkpoint, environment, architecture, seed):
    if not isinstance(checkpoint, str) or not checkpoint or not seed or not architecture or not environment:
        return None
    p = Path(checkpoint)
    # Avoid equating seeds alone across architectures or experiments.
    parent = p.parent.parent if p.parent.name == "snapshots" else p.parent
    return [environment, str(parent), architecture, str(seed)]


def relation(left, right):
    """Positive equality only; absence of an edge is never evidence of independence."""
    reasons = []
    if left["checkpoint"] and left["checkpoint"] == right["checkpoint"]:
        reasons.append("same_checkpoint_path")
    if left["training_lineage"] is not None and left["training_lineage"] == right["training_lineage"]:
        reasons.append("same_declared_training_lineage")
    a, b = left["outcome_export"], right["outcome_export"]
    if a.get("sha256") and a.get("sha256") == b.get("sha256"):
        reasons.append("same_outcome_export_bytes")
    if (a.get("dataset_sha256") and a.get("dataset_sha256") == b.get("dataset_sha256")
            and a.get("row_episode_ids") and b.get("row_episode_ids")
            and set(a["row_episode_ids"]) & set(b["row_episode_ids"])):
        reasons.append("same_declared_dataset_and_overlapping_outcome_episode_ids")
    x, y = left["planning"], right["planning"]
    if x.get("sha256") and x.get("sha256") == y.get("sha256"):
        reasons.append("same_external_evidence_bytes")
    if (left["environment"] == right["environment"] and x.get("planning_seed") is not None
            and x.get("planning_seed") == y.get("planning_seed")
            and x.get("episode_ids") and y.get("episode_ids")
            and set(x["episode_ids"]) & set(y["episode_ids"])):
        reasons.append("same_declared_planning_seed_and_episode_indices")
    return reasons


def component_table(strata, edges):
    parent = list(range(len(strata)))
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for edge in edges:
        a, b = root(edge["left"]), root(edge["right"])
        parent[b] = a
    groups = defaultdict(list)
    for i in range(len(strata)):
        groups[root(i)].append(i)
    return sorted(groups.values(), key=lambda x: x[0])


def panel(source):
    strata = []
    records = []
    pinned = {str(Path(x["path"])): x.get("sha256") for x in source["input"]}
    for i, stratum in enumerate(source["strata"]):
        strata.append({"index": i, "environment": stratum["environment"],
                       "training_stage": stratum["training_stage"], "identifiable": stratum["identifiable"],
                       "eligible_count": stratum["eligible_count"],
                       "reason_if_gated": stratum["reason_if_gated"]})
        for candidate in stratum["candidates"]:
            record_path = Path(candidate["source_path"])
            saved = load(record_path)
            if "error" in saved:
                records.append({"stratum": i, "record_path": str(record_path), "error": saved["error"]})
                continue
            if not pinned.get(str(record_path)) or pinned[str(record_path)] != saved["sha256"]:
                raise ValueError(f"record changed since predecessor: {record_path}")
            item = saved["value"]["record"]
            prov = item.get("provenance", {})
            env, architecture, seed = stratum["environment"], candidate["architecture"], candidate["seed"]
            records.append({
                "stratum": i, "eligible": candidate["eligible"], "record_path": str(record_path),
                "record_sha256": saved["sha256"], "environment": env, "architecture": architecture,
                "seed": seed, "checkpoint": item.get("checkpoint"),
                "training_lineage": lineage(item.get("checkpoint"), env, architecture, seed),
                "fit_export": export_identity(resolve_path(prov.get("fit_export"))),
                "outcome_export": export_identity(resolve_path(prov.get("outcome_export"))),
                "planning": planning_identity(resolve_path(prov.get("external_evidence"))),
                "record_protocol_episode_ids": {
                    k: ids(saved["value"].get("protocol", {}).get(k))
                    for k in ("fit_episode_ids", "calibration_episode_ids", "outcome_episode_ids")},
            })
    # Link only candidate members of the original matched selection pools.
    eligible = [r for r in records if r.get("eligible") and strata[r["stratum"]]["identifiable"]]
    pairs = {}
    counts = Counter()
    examples = {}
    for a, b in itertools.combinations(eligible, 2):
        if a["stratum"] == b["stratum"]:
            continue
        why = relation(a, b)
        if not why:
            continue
        pair = tuple(sorted((a["stratum"], b["stratum"])))
        for reason in why:
            counts[reason] += 1
            examples.setdefault((pair, reason), [a["record_path"], b["record_path"]])
        pairs.setdefault(pair, set()).update(why)
    edges = [{"left": i, "right": j, "reasons": sorted(why),
              "examples": {r: examples[((i, j), r)] for r in sorted(why)}}
             for (i, j), why in sorted(pairs.items())]
    identifiable = [s["index"] for s in strata if s["identifiable"]]
    components = [list(filter(lambda i: i in identifiable, component))
                  for component in component_table(strata, edges)]
    components = [c for c in components if c]
    return {"label": source["label"], "strata": strata, "records": records,
            "eligible_identifiable_records": len(eligible), "edges": edges,
            "cross_stratum_candidate_pair_reason_counts": dict(counts),
            "identified_reuse_components": components,
            "components_not_certified_independent": True,
            "unreadable_record_count": sum("error" in r for r in records)}


def render(report):
    lines = ["# Legacy P1: evidence-based reuse graph", "",
             "Components group strata connected by at least one observed reuse relation among eligible candidates.",
             "Different components are not certified independent; this is not an effective sample-size estimate.", ""]
    for p in report["panels"]:
        lines += [f"## {p['label']}", "",
                  f"Records: {len(p['records'])}; eligible records in identifiable strata: {p['eligible_identifiable_records']}.",
                  f"Identified reuse components (stratum indices): {p['identified_reuse_components']}.",
                  f"Cross-stratum edges: {len(p['edges'])}; pair-level reasons: {p['cross_stratum_candidate_pair_reason_counts']}.",
                  "", "| Index | Environment | Stage | Matched | Gate reason | Reuse component |",
                  "|---:|---|---:|---:|---|---|"]
        for s in p["strata"]:
            comp = next((str(c) for c in p["identified_reuse_components"] if s["index"] in c), "not identifiable")
            lines.append(f"| {s['index']} | {s['environment']} | {s['training_stage']} | {s['eligible_count']} | "
                         f"{s['reason_if_gated'] or 'identifiable'} | {comp} |")
        lines += [""]
    lines += ["## Identity constraints", "",
              "- Same declared training lineage is path/architecture/seed identity, not an authenticated training-run manifest.",
              "- Episode IDs are joined only when declared dataset hashes agree; equality of dataset hashes itself is a source declaration.",
              "- Same planning seed and local episode indices are protocol-level reuse evidence, not a verified simulator-root identity.",
              "- External PointMaze outcome roots have no explicit IDs in the cited evidence files; missing identity is unknown, not disjoint.",
              "- Full record, export, and planning source identities and edge examples are in report.json.",
              "- For a new panel, freeze checkpoint lineage, root IDs with dataset namespace and hashes, split assignments, "
              "and separately enumerate intervention and planning roots before outcomes are examined.", ""]
    return "\n".join(lines)


def main():
    raw = SOURCE.read_bytes()
    if sha(raw) != SOURCE_SHA:
        raise ValueError("fixed predecessor report drift")
    source = json.loads(raw)
    if [len(p["input"]) for p in source["panels"]] != [100, 140]:
        raise ValueError("fixed panel denominator drift")
    report = {"schema": "legacy_p1_dependency_graph_20261102_exclusive",
              "predecessor_sha256": SOURCE_SHA,
              "panels": [panel(p) for p in source["panels"]],
              "independent_units_certified": 0,
              "independence_note": "zero certified means unavailable, not zero independent experiments",
              "gate_changed": False, "scientific_admission": False}
    if SOURCE.read_bytes() != raw:
        raise ValueError("predecessor changed")
    if any(len(p["records"]) != n for p, n in zip(report["panels"], (100, 140))):
        raise ValueError("record count mismatch")
    os.mkdir(OUT, 0o700)
    for name, data in (("report.json", (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()),
                       ("report.md", render(report).encode())):
        fd = os.open(OUT / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    print(json.dumps({"output": str(OUT), "panels": [
        {"label": p["label"], "components": p["identified_reuse_components"],
         "edges": len(p["edges"])} for p in report["panels"]]}))


if __name__ == "__main__":
    main()
