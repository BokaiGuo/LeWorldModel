"""Validate every fixed fallback batch and compare with sealed offline predictions."""
import argparse
from collections import Counter
import json
from pathlib import Path

from . import p3_rgb_fallback_h3_physical_20261031 as run
from . import p3_h3_real_context_physical_summary_mixed_20261031 as pairs

NAME = "p3_rgb_fallback_h3_summary_20261031"
MANIFEST = run.ROOT / "results/p3_rgb_fallback_h3_physical_20261031_prepare_RUN01.json"
MANIFEST_SHA = "816d381a4bde7df9f694846ad3d60813f6fe2818e30681d5a518511f08bbb574"
PREDICTED = run.ROOT / "results/p3_rgb_fallback_exploration_20261031_RUN01.json"
PREDICTED_SHA = "23a640d4fb7f1bcdb3ebbc906aeabae2dbafa2e2c82bbd88397803779ab9f155"


def summarize(rows, expected_rows):
    run.need(len(rows) == len(expected_rows) == 52, "fixed real pair roster")
    expected = {(x["identity"]["logical_pair"], x["identity"]["source_variant"]): x
                for x in expected_rows}
    run.need(len(expected) == 52, "prediction identities")
    unique = {}
    matches = 0
    for row in rows:
        identity = (row["identity"]["logical_pair"], row["identity"]["source_variant"])
        predicted = expected.pop(identity)
        run.need(row["pair_key"] == predicted["pair_key"], "action pair binding")
        # This is a check of exact saved physical outcomes, not evidence of independence.
        matches += int(row["new_endpoint"] == predicted["endpoint"]
                       and row["old_endpoint"] == predicted["old_endpoint"])
        signature = {k: row[k] for k in row if k != "identity"}
        if row["pair_key"] in unique:
            run.need(unique[row["pair_key"]] == signature, "repeated action mismatch")
        else:
            unique[row["pair_key"]] = signature
    run.need(not expected and len(unique) == 26, "unique physical pairs")
    records = list(unique.values())
    complete = [r for r in records if r["new_minus_old"] is not None]
    metrics = ("object_position_error", "object_angle_error_wrapped",
               "pusher_position_error", "native_joint_xy_error")
    return {"fixed_pairs": 52, "unique_action_pairs": len(records),
            "unique_complete_pairs": len(complete), "offline_endpoint_pairs_exact": matches,
            "native_success_new": sum(bool(x["new_native_success"]) for x in records),
            "native_success_old": sum(bool(x["old_native_success"]) for x in records),
            "new_suffix_selections": dict(Counter(x["new_suffix_selection"] for x in records)),
            "delta_directions": {m: dict(Counter(
                "lower" if x["new_minus_old"][m] < -1e-6 else
                "higher" if x["new_minus_old"][m] > 1e-6 else "tie"
                for x in complete)) for m in metrics}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    run.need(out.is_absolute() and out.parent == run.ROOT / "results"
             and out.name.startswith(NAME + "_") and not out.exists() and not out.is_symlink(),
             "exclusive output")
    run.need(run.host.sha(MANIFEST) == MANIFEST_SHA and run.host.sha(PREDICTED) == PREDICTED_SHA,
             "sealed inputs")
    manifest, data = run.build()
    run.need(json.loads(MANIFEST.read_bytes()) == manifest, "manifest reconstruction")
    predictions = json.loads(PREDICTED.read_bytes())
    all_pairs, batches = [], []
    for index in range(18):
        suffix = f"{index:02d}" if index in (0, 16) else str(index)
        directory = run.ROOT / f"results/p3_rgb_fallback_h3_physical_20261031_run_BATCH{suffix}_RUN01"
        report_path, prepare_path = directory / "report.json", directory / "prepare.json"
        run.need(report_path.is_file() and prepare_path.is_file(), "missing batch")
        report_sha, prepare_sha = run.host.sha(report_path), run.host.sha(prepare_path)
        expected = run.prepare_batch(manifest, index)
        checked = run.guarded_saved.consume(directory, report_sha, prepare_sha, expected, data["references"])
        report = json.loads(report_path.read_bytes())
        observed = pairs.pair_batch({"prepare": expected, "rows": report["rows"]})
        for i, pair in enumerate(observed):
            pair["old_suffix_selection"] = report["rows"][2 * i].get("suffix_selection")
            pair["new_suffix_selection"] = report["rows"][2 * i + 1].get("suffix_selection")
        all_pairs.extend(observed)
        batches.append({"index": index, "report_sha256": report_sha, "prepare_sha256": prepare_sha,
                        "saved_status": checked["status"]})
        print(json.dumps({"batch": index, "saved_status": checked["status"]}), flush=True)
    run.need(len(all_pairs) == 54 and sum(x["identity"]["padding"] for x in all_pairs) == 2,
             "padding roster")
    real = [x for x in all_pairs if not x["identity"]["padding"]]
    summary = summarize(real, predictions["rows"])
    run.need(run.host.sha(MANIFEST) == MANIFEST_SHA and run.host.sha(PREDICTED) == PREDICTED_SHA,
             "source drift")
    result = {"schema": NAME, "manifest_sha256": MANIFEST_SHA, "prediction_sha256": PREDICTED_SHA,
              "batches": batches, "rows": real, "summary": summary,
              "P1_admission": False, "P3_admission": False, "equal_budget_certified": False,
              "independent_confirmation": False,
              "limitations": "Known roots, dependent action pairs; rule was designed after observing "
                             "incidents. No independent safety or equal-budget claim."}
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"sha256": run.host.sha(out), "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
