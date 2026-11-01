"""Consume the fixed cp15 panel with verified guard-censored partial batches."""
import argparse
import json
from pathlib import Path

from . import p3_h3_real_context_physical_summary_20261031 as full
from . import p3_h3_partial_full_audit_20261031 as partial

NAME = "p3_h3_censored_panel_summary_20261031"
PARTIAL = full.ROOT / "results/p3_h3_partial_full_audit_20261031_RUN01.json"
PARTIAL_SHA = "1e1a7f624b000d8fc89c0062b3473926e1e63ec096d8fe9e87f54e46f019de14"
need, host = full.need, full.host


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    need(out.is_absolute() and out.parent == full.ROOT / "results"
         and out.name.startswith(NAME + "_") and not out.exists() and not out.is_symlink(),
         "exclusive output")
    need(host.sha(full.MANIFEST) == full.MANIFEST_SHA and host.sha(PARTIAL) == PARTIAL_SHA,
         "source hashes")
    manifest, data = full.source.build()
    need(json.loads(full.MANIFEST.read_bytes()) == manifest, "manifest reconstruction")
    partial_result = json.loads(PARTIAL.read_bytes())
    need(partial_result["schema"] == partial.NAME and partial_result["P3_admission"] is False
         and [x["batch"] for x in partial_result["batches"]] == [16, 17], "partial roster")
    rows, digests = [], {}
    for i in range(18):
        path = full.ROOT / f"results/p3_h3_real_context_physical_20261031_batch{i:02d}_RUN01"
        prepared = full.source.prepare_batch(manifest, i)
        report_path, prepare_path = path / "report.json", path / "prepare.json"
        report_sha, prepare_sha = host.sha(report_path), host.sha(prepare_path)
        if i < 16:
            full.source.old.saved.consume(path, report_sha, prepare_sha, prepared, data["references"])
            report = json.loads(report_path.read_bytes())
            events = full.source.batch.kernel.journal(path / "journal.jsonl", report["journal_chain_tail"])
            full.validate_guards(prepared, report, events)
            pairs = full.pair_batch(prepared, report)
        else:
            entry = partial_result["batches"][i - 16]
            need(report_sha == entry["report_sha256"], "partial report identity")
            report, checked = partial.inspect(path, report_sha, prepared, data)
            pairs = full.pair_batch(prepared, report)
            need(pairs == entry["pairs"] and checked["statuses"] == entry["statuses"],
                 "partial audit reproduction")
        rows.extend(pairs)
        need(host.sha(report_path) == report_sha and host.sha(prepare_path) == prepare_sha, "batch drift")
        digests[str(report_path)] = report_sha
        digests[str(prepare_path)] = prepare_sha
        print(json.dumps({"batch": i, "verified": True}), flush=True)
    value = full.aggregate(rows)
    value["schema"] = NAME
    value["limitations"] += (" Guard-censored endpoints are null; failed-branch incident checks do not "
                             "independently certify every pre-failure frame.")
    value["manifest_sha256"], value["partial_sha256"] = full.MANIFEST_SHA, PARTIAL_SHA
    value["batch_sha256"], value["code_sha256"] = digests, host.sha(__file__)
    need(host.sha(full.MANIFEST) == full.MANIFEST_SHA and host.sha(PARTIAL) == PARTIAL_SHA
         and all(host.sha(path) == digest for path, digest in manifest["bindings"].items()), "source drift")
    with out.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"sha256": host.sha(out), "unique_action_pairs": value["unique_action_pairs"],
                      "complete_unique_pairs": value["complete_unique_pairs"]}), flush=True)


if __name__ == "__main__":
    main()
