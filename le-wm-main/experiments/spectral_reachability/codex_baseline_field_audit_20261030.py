"""Read-only, SHA-pinned protocol/resource field audit; never compare scores."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = {
    "paper": ("results/p2_reported_baseline_table_20261027_v3/export/report.json", "52e687811e6d6f728ab3eda60836ea05eff11f36da2bb4af975d1dd19ef380f1"),
    "grasp": ("results/agent_p2_common_resource_grasp_cuda_20261029_v1_consume_MAIN01.json", "d3ee465f2fa6d31345f369f9514165c4817c51a29c3d00ffdd2d681e5829cc1f"),
    "hi_low": ("results/agent_hilewm_low_cem_20261029_v1_consume_MAIN01.json", "96be2b84aadabc4ca23a8e84f2f54f889b86367d41d4d344e0d7d56b46e3b43e"),
    "hi_metadata": ("results/agent_hilewm_normalizer_asset_route_20261029_v2_mirror01/report.json", "f8eeae02c04ef66449243f687c199799c117fb438da9922451bdd1853d6d2256"),
    "lewm": ("results/agent_lewm_bounded_native_cem_h3_20261029_v1_consume_MAIN01.json", "d0e215618c367049c09eed7e93accb90615973f4f8cb322fd19359fef1f7e7a4"),
}
IDS = {
    "GRASP": "grasp:S4.T1.3:5:1",
    "Hi-LeWM": "hilewm:A3.T4.3:15:7",
    "LeWM": "lewm_v3:A7.T10.6:1:1",
    "PlanWAM": "planwam:S4.T1.6:16:11",
}
NAME = "codex_baseline_field_audit_20261030"


def read_inputs():
    result = {}
    for name, (path, sha) in FILES.items():
        raw = (ROOT / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError(f"SHA mismatch: {name}")
        result[name] = json.loads(raw)
    return result


def field(value, pointer, source):
    missing = value is None or value == "unknown"
    return {"status": "unknown_in_source" if missing else "recorded",
            "value": None if missing else value, "source": source, "pointer": pointer}


def missing_leaves(value, pointer=""):
    """Enumerate actual null/unknown leaves, retaining precise source paths."""
    if value is None or value == "unknown":
        return [pointer]
    if isinstance(value, dict):
        return [p for k, v in value.items()
                for p in missing_leaves(v, pointer + "/" + k.replace("~", "~0").replace("/", "~1"))]
    if isinstance(value, list):
        return [p for i, v in enumerate(value) for p in missing_leaves(v, pointer + "/" + str(i))]
    return []


def audit(data):
    paper = data["paper"]
    if paper["cross_paper_ranking_allowed"] is not False:
        raise ValueError("paper comparison boundary changed")
    output = {}
    for method, cell in IDS.items():
        matches = [(i, row) for i, row in enumerate(paper["rows"]) if row["id"] == cell]
        if len(matches) != 1:
            raise ValueError("missing or duplicate paper cell")
        index, row = matches[0]
        r = row["record"]
        if r["paper"] != method or r["evidence_kind"] != "paper_reported":
            raise ValueError("paper identity mismatch")
        fields = ("value", "metric", "unit", "env", "dataset_split", "checkpoint",
                  "action_space", "observations", "horizon", "n_seeds", "uncertainty", "budget", "version")
        author = {k: field(r[k], f"/rows/{index}/record/{k}", "paper") for k in fields}
        author["outcome_domain"] = field(row["outcome_domain"], f"/rows/{index}/outcome_domain", "paper")
        author["full_pipeline_cost"] = field(row["full_pipeline_cost"], f"/rows/{index}/full_pipeline_cost", "paper")
        output[method] = {"paper_cell": cell, "author_reported": author,
                          "author_missing_leaf_pointers": missing_leaves(r, f"/rows/{index}/record")}

    g = data["grasp"]
    order = g["contract"]["order"]
    if g["fixed_denominator"] != 6 or len(order) != 6 or len(g["resources"]) != 6:
        raise ValueError("GRASP roster mismatch")
    indices = [i for i, slot in enumerate(order) if slot[1] == "grasp"]
    if [order[i][0] for i in indices] != [0, 1]:
        raise ValueError("GRASP root mismatch")
    output["GRASP"]["local"] = {
        "scope": "P2 bounded GRASP model-only; two known roots, not official reproduction",
        "status": field(g["status"], "/status", "grasp"),
        "protocol": {k: field(g["contract"][k], "/contract/" + k, "grasp")
                     for k in ("context_length", "raw_steps", "predict_API_cap", "predict_samples_cap",
                               "search_scope", "search_wall_seconds", "official_GRASP_reproduction", "equal_FLOPs")},
        "root_resources": [
            {"root": order[i][0], "fields": {k: field(g["resources"][i][k], f"/resources/{i}/{k}", "grasp")
              for k in ("attempted", "completed", "engine_internal_backward_calls", "independent_JVP_calls", "total_FLOPs")}}
            for i in indices],
        "physical_outcome": {"status": "not_in_selected_receipt", "value": None},
        "missing_resource_pointers": [p for i in indices
                                      for p in missing_leaves(g["resources"][i], f"/resources/{i}")],
    }
    h = data["hi_low"]
    output["Hi-LeWM"]["local"] = {
        "scope": "normalized low-level CEM only, not complete hierarchical policy",
        "fields": {k: field(h[k], "/" + k, "hi_low") for k in
                   ("status", "known_roots", "fresh_roots", "full_policy", "official_baseline", "physical_release", "resources")},
        "raw_inverse_authority": [field(row["raw_inverse_authority"], f"/rows/{i}/raw_inverse_authority", "hi_low")
                                  for i, row in enumerate(h["rows"])],
        "metadata": {k: field(data["hi_metadata"][k], "/" + k, "hi_metadata") for k in
                     ("status", "exhaustive", "author_frozen_scaler_obtained", "raw_inverse_authority", "official_equivalence")},
        "physical_outcome": {"status": "not_in_selected_receipt", "value": None},
        "missing_resource_pointers": missing_leaves(h["resources"], "/resources"),
    }
    l = data["lewm"]
    output["LeWM"]["local"] = {
        "scope": "known-root H3 bounded native CEM model receipt; not original paper evaluation",
        "fields": {k: field(l[k], "/" + k, "lewm") for k in
                   ("status", "official_reproduction", "fair_ranking", "same_support_as_P2",
                    "same_planning_reduction_as_P2", "resources")},
        "planning_scores": [field(row["planning_scores"], f"/rows/{i}/planning_scores", "lewm")
                            for i, row in enumerate(l["rows"])],
        "total_flops": {"status": "not_in_selected_receipt", "value": None},
        "physical_outcome": {"status": "not_in_selected_receipt", "value": None},
        "missing_resource_pointers": missing_leaves(l["resources"], "/resources"),
    }
    output["PlanWAM"]["local"] = {
        "scope": "no local official driving result asserted by this audit",
        "result": {"status": "not_in_selected_receipts", "value": None},
        "resources": {"status": "not_in_selected_receipts", "value": None},
    }
    return {"schema": NAME, "sources": {k: {"path": p, "sha256": s} for k, (p, s) in FILES.items()},
            "methods": output, "comparisons_allowed": False,
            "trust": "pinned saved projections; upstream fullstrict and paper source extraction not rerun",
            "missing_semantics": "unknown or absent in selected sources, not proof of global absence",
            "new_gpu_calls": 0, "new_neural_calls": 0, "new_physics_calls": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    result = audit(read_inputs())
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path = Path(args.output)
    if not path.is_absolute() or path.parent != ROOT / "results" or not path.name.startswith(NAME + "_"):
        raise ValueError("exclusive audit output path required")
    if args.command == "build":
        with path.open("xb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
    else:
        saved = path.read_bytes()
        if args.sha256 is None or hashlib.sha256(saved).hexdigest() != args.sha256 or saved != raw:
            raise ValueError("output SHA or reconstruction mismatch")
    print(json.dumps({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "command": args.command}))


if __name__ == "__main__":
    main()
