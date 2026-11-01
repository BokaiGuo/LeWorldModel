"""Descriptive fixed-roster H3 summary with exact duplicate-input accounting."""
import argparse
import hashlib
import json
from pathlib import Path

from . import p3_raw_h3_roster_dependence_20261031 as roster

ROOT, require = roster.source.ROOT, roster.require
DEPENDENCE = ROOT / "results/p3_raw_h3_roster_dependence_20261031_RUN01.json"
DEPENDENCE_SHA = "599dff4e935f3000dbaa3b16af39203db218690afb2759fa36c8d1fbcb219abc"
RECEIPTS = [
    ("physical", "e6442c26a74167fad45b77c9174f4e55be0d3588579d329d8b9d996f9daa1645"),
    ("partial", "8bcc5311089e98945316ec54b47f2aab017c9ffcc3cc2788f46e13ff252bcf76"),
    ("partial", "3a9d28cbb037f8d2e1935cf972951eb02f3dabe28d06096a5e4cf86333678927"),
    ("physical", "6be5ad6dd1dda82527e508434353a6ff919b98b0bde9e1078cecf645474ba61c"),
    ("partial", "8c1bb24bdfdb0e804bfda6100e32d87841d7e3af50b916c6b7aeba6354918552"),
    ("partial", "ce9a77b4d81730a9b86279fb68bde78c062a16d7c823e94434500a094907fa7c"),
    ("physical", "75f803f6da321f7b7ac43bbb9ed3394512a9c37cb338ccfa6788cc0100043329"),
    ("partial", "4a804804182ce739afaf69a0dc2606de2fa24a3e60c7678ff779708f56ac84e1"),
    ("partial", "b023c3a624fb41735a8f0a0a630d9e976cdaad621aa92d586913eeaf1f788641"),
    ("physical", "470fdaae688cf1aa088de05de464642cff74a7a264c51b5def5ed0113ab808be"),
    ("partial", "061ba6aed0a9e7fc6ff3eb2b7221b4637c7e6ddbf23f3c4f200d7f7ab6d836c3"),
    ("partial", "9846fb63c00518ce2dd88a6bb9036ed95609d9d682fd68734794021fb82ccdf5"),
]
OUTCOME = ("step15_exact", "old_status", "new_status", "old_returned", "new_returned",
           "old_last_success", "new_last_success", "old_endpoint", "new_endpoint",
           "new_minus_old_endpoint")


def checked(path, digest):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, "receipt hash")
    return json.loads(raw)


def summarize(manifest, dependence, receipts):
    require(dependence["manifest_sha256"] == roster.source.MANIFEST_SHA
            and dependence["uses_physical_outcomes"] is False
            and len(receipts) == 12, "roster source contract")
    recomputed = roster.index_roster(manifest["plans"])
    require(all(dependence[key] == value for key, value in recomputed.items()), "roster reconstruction")
    logical = []
    for i, receipt in enumerate(receipts):
        require(receipt["batch_index"] == i and receipt["P3_admission"] is False
                and receipt["independent_confirmation"] is False and len(receipt["pairs"]) == 3,
                "receipt contract")
        for j, pair in enumerate(receipt["pairs"]):
            plan = manifest["plans"][6 * i + 2 * j]
            require(pair["root_slot"] == plan["root_slot"]
                    and pair["decision_index"] == plan["decision_index"]
                    and pair["candidate"] == plan["candidate"], "pair identity")
            logical.append(pair)
    groups = []
    for group in dependence["pair_groups"]:
        ids = group["indices"]
        require(len(ids) == 2, "fixed duplicate multiplicity")
        first, second = [logical[i] for i in ids]
        same = all(first[k] == second[k] for k in OUTCOME)
        groups.append({"logical_pair_indices": ids, "outcomes_exact": same,
                       "records": [{k: record[k] for k in OUTCOME} for record in (first, second)]})
    require(len(groups) == 18, "fixed unique pair count")
    consistent = [g["records"][0] for g in groups if g["outcomes_exact"]]
    valid = [r for r in consistent if r["step15_exact"] is True
             and r["new_minus_old_endpoint"] is not None]
    changes = {}
    for metric in ("object_position_error", "wrapped_object_angle_error", "native_distance"):
        values = [r["new_minus_old_endpoint"][metric] for r in valid]
        changes[metric] = {"negative": sum(v < 0 for v in values), "zero": sum(v == 0 for v in values),
                           "positive": sum(v > 0 for v in values), "values": values}
    return {"logical_pairs": 36, "unique_input_pairs": len(groups),
            "exact_repeat_pairs": len(consistent), "repeat_mismatch_pairs": len(groups) - len(consistent),
            "consistent_full_horizon_pairs_with_exact_prefix": len(valid),
            "consistent_pairs_with_any_guard_failure": sum(
                r["old_status"] == "failed" or r["new_status"] == "failed" for r in consistent),
            "consistent_old_success_observed": sum(
                r["old_last_success"] is True for r in consistent if r["old_status"] != "failed"),
            "consistent_new_success_observed": sum(
                r["new_last_success"] is True for r in consistent if r["new_status"] != "failed"),
            "consistent_old_success_status_eligible": sum(r["old_status"] != "failed" for r in consistent),
            "consistent_new_success_status_eligible": sum(r["new_status"] != "failed" for r in consistent),
            "endpoints_new_minus_old": changes, "groups": groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    require(output.is_absolute() and output.parent == ROOT / "results"
            and output.name.startswith("p3_raw_h3_panel_summary_20261031_")
            and not output.exists() and not output.is_symlink(), "exclusive output")
    manifest = checked(roster.source.MANIFEST, roster.source.MANIFEST_SHA)
    dependence = checked(DEPENDENCE, DEPENDENCE_SHA)
    receipts, sources = [], {str(DEPENDENCE): DEPENDENCE_SHA,
                            str(roster.source.MANIFEST): roster.source.MANIFEST_SHA}
    for i, (kind, digest) in enumerate(RECEIPTS):
        path = ROOT / f"results/p3_raw_h3_{kind}_pair_20261031_BATCH{i:02d}_RUN01.json"
        receipts.append(checked(path, digest))
        sources[str(path)] = digest
    result = {"schema": "p3_raw_h3_panel_summary_20261031", **summarize(manifest, dependence, receipts),
              "sources": sources, "analyzer_sha256": roster.source.host.sha(Path(__file__)),
              "independent_confirmation": False, "equal_budget_certified": False, "P3_admission": False,
              "limitations": "Descriptive post-outcome known-root panel; duplicate inputs are not independent "
                             "samples; guard failures are censored; no statistical inference."}
    require(all(roster.source.host.sha(path) == digest for path, digest in sources.items()), "source drift")
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"sha256": hashlib.sha256(raw).hexdigest(), "unique_pairs": result["unique_input_pairs"],
                      "repeat_mismatches": result["repeat_mismatch_pairs"]}))


if __name__ == "__main__":
    main()
