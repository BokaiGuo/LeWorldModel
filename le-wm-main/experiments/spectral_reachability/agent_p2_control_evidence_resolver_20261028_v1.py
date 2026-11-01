"""Bounded saved-only evidence projection; never a matched-control certificate."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from experiments.spectral_reachability import agent_p2_strong_baseline_saved_summary_20261028_v3 as contract

NAME = "agent_p2_control_evidence_resolver_20261028_v1"
LIMIT = 8 * 1024**2


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load(path, expected):
    path = Path(path)
    require(path.is_absolute(), "absolute explicit evidence path")
    require(path.is_file() and not path.is_symlink(), "regular evidence file")
    require(path.stat().st_size <= LIMIT, "bounded metadata file")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "evidence SHA drift")
    # Gzip is a transport, never guessed as an identity. Bound decoded bytes too.
    if raw[:2] == b"\x1f\x8b":
        with gzip.open(path, "rb") as stream:
            raw = stream.read(LIMIT + 1)
        require(len(raw) <= LIMIT, "bounded gzip metadata")
    value = json.loads(raw)
    canonical(value)
    return value


def binding(path, sha, pointer):
    return {"path": str(path), "sha256": sha, "pointer": pointer}


def leaf(document, path, sha, pointer, chain):
    value = contract.pointer(document, pointer)
    require(contract.known(value), "unknown evidence " + pointer)
    return {"value": value, "origin": binding(path, sha, pointer), "chain": chain}


def pair(document, path, sha, prefix, path_key, sha_key, chain):
    """Only these explicit sibling path/hash keys authorize traversal."""
    obj = contract.pointer(document, prefix)
    require(isinstance(obj, dict) and path_key in obj and sha_key in obj,
            "missing explicit link " + prefix + "/" + path_key + "+" + sha_key)
    linked_path, linked_sha = obj[path_key], obj[sha_key]
    linked = load(linked_path, linked_sha)
    edge = {"from": binding(path, sha, prefix + "/" + path_key),
            "hash_from": binding(path, sha, prefix + "/" + sha_key),
            "to": binding(linked_path, linked_sha, "")}
    return linked, Path(linked_path), linked_sha, chain + [edge]


TASK_FIELDS = {
    "task_identity": ("/task_id",),
    "checkpoint_identity": ("/task/model_sha256", "/task/model_config_sha256", "/task_contract/model_bindings"),
    "action_support": ("/task/packet_logit_cap", "/task/control_dim", "/task/packet_steps",
                       "/task/action_mean", "/task/action_std", "/task_contract/normalization"),
    "observation_spec": ("/task/representation", "/task/latent_dim", "/task_contract/observed_steps",
                         "/task_contract/pixel_sha256", "/task/root_context", "/task/root_raw_action_history",
                         "/history_semantics", "/with_target"),
    "eval_metric": ("/task/projection", "/task_contract/metric"),
    "horizon": ("/task/horizon", "/task/packet_steps"),
    "goal_identity": ("/task/goal", "/task/goal_source_sha256"),
    "source_authority": ("/blind_source_sha256", "/blind_sha256"),
}


def task_evidence(ref, task, path, sha, chain):
    # Slot alone is NOT identity. Close every identity present in the referring row.
    require(all(key in ref for key in ("source_slot", "task_id", "blind_sha256", "task_contract_sha256")),
            "incomplete referring task identity")
    for key in ("source_slot", "task_id", "blind_sha256"):
        require(ref[key] == task[key], "task join mismatch " + key)
    require(ref["task_contract_sha256"] == task["task"]["task_contract_sha256"], "task contract reference mismatch")
    require(hashlib.sha256(json.dumps(task["task_contract"], sort_keys=True, allow_nan=False).encode()).hexdigest() == ref["task_contract_sha256"],
            "task contract content digest")
    result, missing = {}, []
    for field, pointers in TASK_FIELDS.items():
        components = {}
        for pointer in pointers:
            try:
                components[pointer] = leaf(task, path, sha, pointer, chain)
            except (KeyError, ValueError, IndexError) as exc:
                missing.append({"field": field, "pointer": pointer, "reason": str(exc)})
        result[field] = components
    return result, missing


def resolve_asset(name, path, expected):
    doc = load(path, expected)
    require(doc.get("P1_admission") is False, "P1 admission flag")
    require(doc.get("errors", []) == [], "asset errors")
    rows = doc.get("rows")
    require(isinstance(rows, list) and rows, "all saved rows")
    fixed = [{"root": r.get("source_slot", r.get("root_sha256")), "arm": r["arm"]} for r in rows]
    require(all(r["root"] is not None for r in fixed), "unknown row root")
    require(len({canonical(r) for r in fixed}) == len(fixed), "duplicate root arm")
    result = {"asset": binding(path, expected, ""), "fixed_rows": fixed,
              "rows": [], "missing": [], "controls": {}, "native_evidence": {}}
    refs, refdoc, refpath, refsha, refchain = rows, doc, path, expected, []
    if name == "kinodynamic":
        refdoc, refpath, refsha, refchain = pair(doc, path, expected, "", "prepared", "prepare_sha256", [])
        refs = refdoc["slots"]
        require(len(refs) == len(rows), "all prepare slots")
        require(len({r["source_slot"] for r in refs}) == len(refs), "duplicate prepare slot")
        by_slot = {r["source_slot"]: (i, r) for i, r in enumerate(refs)}
        ordered = []
        for row in rows:
            i, ref = by_slot[row["source_slot"]]
            for key in ("task_id", "blind_sha256"):
                require(row[key] == ref[key], "report/prepare " + key)
            ordered.append((i, ref))
    else:
        ordered = list(enumerate(refs))
    for index, (refindex, ref) in enumerate(ordered):
        root_key = "source_slot" if "source_slot" in rows[index] else "root_sha256"
        out = {"identity": fixed[index], "evidence": {}, "row_identity_evidence": {
            "root": leaf(doc, path, expected, "/rows/" + str(index) + "/" + root_key, []),
            "arm": leaf(doc, path, expected, "/rows/" + str(index) + "/arm", [])}}
        prefix = ("/slots/" if name == "kinodynamic" else "/rows/") + str(refindex)
        try:
            task, tp, ts, tc = pair(refdoc, refpath, refsha, prefix, "task_export", "task_sha256", refchain)
            out["evidence"], absent = task_evidence(ref, task, tp, ts, tc)
            result["missing"].extend({"row": index, **item} for item in absent)
        except (KeyError, ValueError, OSError, IndexError) as exc:
            result["missing"].append({"row": index, "field": "task_closure", "reason": str(exc)})
        result["rows"].append(out)
    # Publish a control only if EVERY saved row has EVERY declared component.
    for field, pointers in TASK_FIELDS.items():
        components = [r["evidence"].get(field, {}) for r in result["rows"]]
        if all(set(c) == set(pointers) for c in components):
            result["controls"][field] = [{p: c[p]["value"] for p in pointers} for c in components]
        else:
            result["controls"][field] = None
            result["missing"].append({"field": field, "reason": "not closed for all fixed rows"})
    # Root IDs are source-local identifiers, not global freshness certificates.
    result["controls"]["root_ids"] = [r["root"] for r in fixed]
    result["controls"]["resource_axes"] = None
    for key in ("resources", "counts", "sampling_cost", "inherited_encoding_once", "inherited_cost", "contract",
                "model_hash_before", "model_hash_after", "model_unchanged"):
        if key in doc:
            if contract.known(doc[key]):
                result["native_evidence"][key] = leaf(doc, path, expected, "/" + key, [])
            else:
                result["missing"].append({"field": key, "reason": "native evidence contains unknown/empty values; not defaulted"})
    if name == "kinodynamic" and "inherited_encoding_report" in refdoc:
        receipt, rp, rs, rc = pair(refdoc, refpath, refsha, "", "inherited_encoding_report",
                                   "inherited_encoding_report_sha256", refchain)
        for key in ("model_hash_before", "model_hash_after", "model_unchanged"):
            if key in receipt:
                result["native_evidence"]["inherited_" + key] = leaf(receipt, rp, rs, "/" + key, rc)
    result["missing"].append({"field": "resource_axes", "reason": "native counts retained; no complete v3 per-root-per-arm limit/inherited budget witness"})
    result["missing"].append({"field": "loaded_checkpoint_bytes", "reason": "model digest declarations/receipts are not independent proof of originally loaded bytes"})
    return result


def build():
    return {"schema": NAME + ".evidence", "v3_source_sha256": contract.sha(Path(contract.__file__)),
            "resolver_source_sha256": contract.sha(Path(__file__)),
            "assets": {n: resolve_asset(n, p, s) for n, (p, s) in contract.ASSETS.items()},
            "status": "partial_saved_evidence_not_matched", "matched": False,
            "run_authorized": False, "P1_admission": False, "P2_admission": False,
            "composition_benefit": None}


def catalog(path, expected, value):
    # v3 reads plain JSON only: projected gzip leaves retain original origin+chain.
    return {name: {field: binding(path, expected, "/assets/" + name + "/controls/" + field)
                   for field, item in asset["controls"].items() if contract.known(item)}
            for name, asset in value["assets"].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("resolve", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sha")
    args = parser.parse_args()
    path = args.output.absolute()
    value = build()
    if args.command == "resolve":
        data = canonical(value) + b"\n"
        require(len(data) <= LIMIT, "output cap")
        with path.open("xb") as stream:
            stream.write(data)
    else:
        require(args.sha is not None, "external output SHA required")
        require(load(path, args.sha) == value, "full evidence reconstruction")
    digest = contract.sha(path)
    print(json.dumps({"path": str(path), "sha256": digest, "status": value["status"],
                      "v3_evidence": catalog(path, digest, value),
                      "missing_counts": {n: len(a["missing"]) for n, a in value["assets"].items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
