"""Consume the fixed known-root RGB-abstention panel without imputing endpoints."""
import argparse
import json
from pathlib import Path

from . import p3_rgb_guarded_h3_physical_20261031 as run
from . import p3_h3_real_context_physical_summary_mixed_20261031 as pairs

NAME = "p3_rgb_guarded_h3_summary_20261031"
MANIFEST = run.ROOT / "results/p3_rgb_guarded_h3_physical_20261031_prepare_RUN01.json"
MANIFEST_SHA = "2574ba88bdf69a52bdce48cdba1e73b0ca32638f43228d54b920274d83b19afd"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    run.need(out.is_absolute() and out.parent == run.ROOT / "results"
             and out.name.startswith(NAME + "_") and not out.exists() and not out.is_symlink(),
             "exclusive output")
    run.need(run.host.sha(MANIFEST) == MANIFEST_SHA, "manifest hash")
    manifest, data = run.build()
    run.need(json.loads(MANIFEST.read_bytes()) == manifest, "manifest reconstruction")
    all_pairs, batches = [], []
    for index in range(18):
        directory = run.ROOT / f"results/p3_rgb_guarded_h3_physical_20261031_batch{index:02d}_RUN01"
        report_path, prepare_path = directory / "report.json", directory / "prepare.json"
        run.need(report_path.is_file() and prepare_path.is_file(), "missing batch")
        report_sha, prepare_sha = run.host.sha(report_path), run.host.sha(prepare_path)
        expected = run.prepare_batch(manifest, index)
        checked = run.guarded_saved.consume(directory, report_sha, prepare_sha, expected, data["references"])
        report = json.loads(report_path.read_bytes())
        all_pairs.extend(pairs.pair_batch({"prepare": expected, "rows": report["rows"]}))
        batches.append({"index": index, "report_sha256": report_sha, "prepare_sha256": prepare_sha,
                        "saved_status": checked["status"]})
        print(json.dumps({"batch": index, "saved_status": checked["status"]}), flush=True)
    run.need(len(all_pairs) == 54 and sum(x["identity"]["padding"] for x in all_pairs) == 2,
             "fixed pair roster")
    real = [pair for pair in all_pairs if not pair["identity"]["padding"]]
    run.need(len(real) == 52 and len({(p["identity"]["logical_pair"], p["identity"]["source_variant"])
                                     for p in real}) == 52, "unique logical identities")
    unique = {}
    for pair in real:
        key = pair["pair_key"]
        signature = {k: pair[k] for k in pair if k not in ("identity",)}
        if key in unique:
            run.need(unique[key] == signature, "repeated action pair differs")
        else:
            unique[key] = signature
    results = list(unique.values())
    result = {"schema": NAME, "manifest_sha256": MANIFEST_SHA, "batches": batches, "rows": real,
              "fixed_pairs": len(real), "unique_action_pairs": len(results),
              "unique_both_full": sum(x["old_endpoint"] is not None and x["new_endpoint"] is not None
                                      for x in results),
              "unique_both_abstained": sum(x["old_status"] == x["new_status"] == "rgb_preflight_abstain"
                                           for x in results),
              "unique_mixed_status": sum(x["old_status"] != x["new_status"] for x in results),
              "P1_admission": False, "P3_admission": False, "equal_budget_certified": False,
              "independent_confirmation": False,
              "limitations": "Known roots; rule and batch16 pilot follow observed incidents; "
                             "no independent safety or causal performance estimate."}
    with out.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"sha256": run.host.sha(out), "unique": len(results),
                      "both_full": result["unique_both_full"],
                      "both_abstained": result["unique_both_abstained"]}), flush=True)


if __name__ == "__main__":
    main()
