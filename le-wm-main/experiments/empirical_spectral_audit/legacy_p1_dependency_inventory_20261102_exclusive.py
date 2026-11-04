"""All-record legacy P1 reuse inventory; no model execution or gate change."""
import hashlib
import itertools
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from experiments.empirical_spectral_audit.legacy_p1_dependency_graph_20261102_exclusive import (
    ROOT, relation,
)

GRAPH = ROOT / "results/legacy_p1_dependency_graph_20261102_exclusive_RUN01/report.json"
GRAPH_SHA = "de2e7394e42a42f6072c821911bd4c132dbfdaa4d2aa942a93c85a3d322ed500"
MANIFEST = ROOT / "results/pointmaze_external_evidence_panel_20261014_v2/manifest.json"
OUT = ROOT / "results/legacy_p1_dependency_inventory_20261102_exclusive_RUN01"


def read(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def root_key(row):
    return f"{int(row['episode_id'])}:{int(row.get('window_start', 0))}"


def conditional_roots(external_rows, fit_rows, outcome_rows, paired_roots):
    """Reconstruct only under unchanged-input/code and reported-count assumptions."""
    if not isinstance(paired_roots, int) or paired_roots <= 0 or paired_roots % 2:
        raise ValueError("invalid paired root count")
    used = {root_key(x) for x in fit_rows + outcome_rows}
    filtered = [root_key(x) for x in external_rows
                if int(x.get("window_start", 0)) != 0 and root_key(x) not in used]
    count = paired_roots // 2
    if len(filtered) < count:
        raise ValueError("current export cannot reconstruct reported count")
    return filtered[:count]


def inventory(panels):
    rows = [(i, r) for i, panel in enumerate(panels) for r in panel["records"]]
    counts = Counter()
    groups = {}
    for (pi, a), (pj, b) in itertools.combinations(rows, 2):
        why = relation(a, b)
        if not why:
            continue
        key = (pi, a["stratum"], pj, b["stratum"])
        group = groups.setdefault(key, {"pair_counts": Counter(), "examples": {}})
        for reason in why:
            counts[("cross_panel" if pi != pj else "within_panel", reason)] += 1
            group["pair_counts"][reason] += 1
            group["examples"].setdefault(reason, [a["record_path"], b["record_path"]])
    return {
        "pair_reason_counts": {f"{k[0]}:{k[1]}": v for k, v in sorted(counts.items())},
        "stratum_groups": [
            {"left_panel": k[0], "left_stratum": k[1], "right_panel": k[2], "right_stratum": k[3],
             "pair_counts": dict(v["pair_counts"]), "examples": v["examples"]}
            for k, v in sorted(groups.items())],
    }


def reconstruct(panel, manifest):
    listed = set(manifest["reports"])
    external_path = Path(manifest["protocol"]["external_export"])
    export, export_sha = read(external_path)
    rows = export["enriched_records"]
    out = []
    for record in panel["records"]:
        if record["environment"] != "PointMaze":
            continue
        path = record["planning"]["path"]
        if path not in listed:
            raise ValueError(f"evidence not listed in manifest: {path}")
        evidence, evidence_sha = read(Path(path))
        if evidence_sha != record["planning"]["sha256"]:
            raise ValueError(f"evidence drift: {path}")
        fit, fit_sha = read(Path(record["fit_export"]["path"]))
        outcome, outcome_sha = read(Path(record["outcome_export"]["path"]))
        if fit_sha != record["fit_export"]["sha256"] or outcome_sha != record["outcome_export"]["sha256"]:
            raise ValueError("diagnostic export drift")
        roots = conditional_roots(rows, fit["enriched_records"], outcome["enriched_records"],
                                  evidence["action_risk"]["paired_roots"])
        if evidence["planning_consequence"]["paired_roots"] != len(roots):
            raise ValueError("planning and intervention root counts disagree")
        out.append({"record_path": record["record_path"], "stratum": record["stratum"],
                    "evidence_path": path, "root_keys": roots})
    overlaps = Counter()
    examples = {}
    for a, b in itertools.combinations(out, 2):
        if a["stratum"] == b["stratum"]:
            continue
        n = len(set(a["root_keys"]) & set(b["root_keys"]))
        pair = tuple(sorted((a["stratum"], b["stratum"])))
        if n:
            overlaps[pair] += 1
            examples.setdefault(pair, {"records": [a["record_path"], b["record_path"]],
                                       "overlap_keys": sorted(set(a["root_keys"]) & set(b["root_keys"]))})
    return {"external_export_path": str(external_path), "current_external_export_sha256": export_sha,
            "conditional_records": out,
            "cross_stratum_overlaps": [{"strata": list(k), "record_pairs": v, **examples[k]}
                                       for k, v in sorted(overlaps.items())],
            "qualification": "Conditional on manifest applying to each listed evidence and unchanged export bytes, "
                             "row order and evaluator implementation at execution; neither input hash nor max_roots "
                             "is pinned in manifest. Count derives from paired_roots / 2, not recorded root IDs."}


def main():
    graph, graph_sha = read(GRAPH)
    if graph_sha != GRAPH_SHA:
        raise ValueError("source graph drift")
    manifest, manifest_sha = read(MANIFEST)
    panels = graph["panels"]
    if [len(p["records"]) for p in panels] != [100, 140]:
        raise ValueError("denominator drift")
    result = {"schema": "legacy_p1_all_record_reuse_inventory_v1",
              "graph_sha256": graph_sha, "manifest_path": str(MANIFEST), "manifest_sha256": manifest_sha,
              "all_record_inventory": inventory(panels),
              "pointmaze_conditional_reconstruction": reconstruct(panels[1], manifest),
              "qualification": "Pair counts are reuse relations, not sample sizes. No edge does not prove independence.",
              "gate_changed": False}
    os.mkdir(OUT, 0o700)
    fd = os.open(OUT / "report.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps({"output": str(OUT), "pair_reason_counts": result["all_record_inventory"]["pair_reason_counts"],
                      "conditional_pointmaze_records": len(result["pointmaze_conditional_reconstruction"]["conditional_records"])}))


if __name__ == "__main__":
    main()
