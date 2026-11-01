"""Verify all rows of cp15-replan partial batches, preserving guard censoring."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p3_h3_replan_guard_incident_20261031 as incident
from . import p3_raw_h3_partial_pair_20261031 as prior

NAME = "p3_h3_partial_full_audit_20261031"
need, host = incident.need, incident.host
kernel = incident.summary.source.batch.kernel


def inspect(directory, digest, prepared, data):
    failures = incident.inspect_failed(directory, digest, prepared, data)
    report = json.loads((directory / "report.json").read_bytes())
    events = kernel.journal(directory / "journal.jsonl", report["journal_chain_tail"])
    counts = {k: {"attempted": 0, "completed": 0} for k in prepared["contract"]["caps"]}
    for event in events:
        if event["kind"] in ("operation_attempted", "operation_completed"):
            need(event["resource"] in counts, "unknown resource")
            counts[event["resource"]]["attempted" if event["kind"] == "operation_attempted"
                                      else "completed"] += 1
    need(counts == report["counts"], "operation census")
    frames = [e["receipt"]["png"] for e in events if e["kind"] == "fresh_observation"]
    statuses = []

    def frame_reader(name, frame_sha):
        path = directory / name
        need(Path(name).name == name and host.sha(path) == frame_sha, "frame hash")
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB")).copy()

    for i, (row, plan) in enumerate(zip(report["rows"], prepared["plans"], strict=True)):
        arm = f"d{plan['decision_index']}_c{plan['candidate']}_{plan['variant']}"
        need(row["branch_index"] == i and row["selected_plan"] == {**plan, "arm": arm},
             "branch identity")
        if row["status"] == "failed":
            need(any(f["branch"] == i for f in failures), "guard failure coverage")
        else:
            prior.validate_complete(row, plan, data["references"][plan["root_slot"]],
                                    events, frame_reader)
        statuses.append(row["status"])
    need(len(frames) == len(set(frames)) and {p.name for p in directory.iterdir()} ==
         set(frames) | {"prepare.json", "report.json", "journal.jsonl"}, "artifact census")
    need(host.sha(directory / "report.json") == digest, "report drift")
    return report, {"statuses": statuses, "guard_incidents": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    need(out.is_absolute() and out.parent == incident.ROOT / "results"
         and out.name.startswith(NAME + "_") and not out.exists() and not out.is_symlink(),
         "exclusive output")
    need(host.sha(incident.summary.MANIFEST) == incident.summary.MANIFEST_SHA, "manifest hash")
    manifest, data = incident.summary.source.build()
    need(json.loads(incident.summary.MANIFEST.read_bytes()) == manifest, "manifest reconstruction")
    result = []
    for i, digest in incident.REPORTS.items():
        folder = incident.ROOT / f"results/p3_h3_real_context_physical_20261031_batch{i:02d}_RUN01"
        prepared = incident.summary.source.prepare_batch(manifest, i)
        report, checked = inspect(folder, digest, prepared, data)
        result.append({"batch": i, "report_sha256": digest, **checked,
                       "pairs": incident.summary.pair_batch(prepared, report)})
    need(all(host.sha(path) == digest for path, digest in manifest["bindings"].items()), "source drift")
    value = {"schema": NAME, "batches": result, "P1_admission": False, "P3_admission": False,
             "limitations": "Partial batch evidence; missing horizon outcomes remain null. "
                            "Known roots, no equal-budget or independent confirmation."}
    with out.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"sha256": host.sha(out), "batches": len(result)}), flush=True)


if __name__ == "__main__":
    main()
