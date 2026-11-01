"""Verify fixed cp15 replan guard incidents without admitting partial endpoints."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p3_h3_real_context_physical_summary_20261031 as summary

NAME = "p3_h3_replan_guard_incident_20261031"
ROOT, host, need = summary.ROOT, summary.host, summary.need
REPORTS = {16: "8b10d6413e0dbf3ffe893ffa4342deb51cc450272a50ade14deec2e2e50e3225",
           17: "e74d9a2c88754c98a515bfadfa0ffdf8f3eba95a3f161580e873910385a59ed1"}


def inspect_failed(directory, expected, p, data):
    report_path = directory / "report.json"
    need(host.sha(report_path) == expected, "report hash")
    report = json.loads(report_path.read_bytes())
    need(report["status"] == "partial_or_failed" and len(report["rows"]) == 6
         and report["fixed_denominator"] == 6 and report["new_neural_calls"] == 0,
         "partial report")
    need(json.loads((directory / "prepare.json").read_bytes()) == p, "prepared batch")
    need(host.sha(directory / "journal.jsonl") == report["journal_sha256"], "journal hash")
    kernel = summary.source.batch.kernel
    events = kernel.journal(directory / "journal.jsonl", report["journal_chain_tail"])
    need(events[0]["kind"] == "run_started" and events[0]["prepare"] == p
         and events[-1]["kind"] == "run_finished" and events[-1]["status"] == report["status"]
         and [e["receipt"] for e in events if e["kind"] == "branch_closed"] == report["rows"],
         "journal and row closure")
    summary.validate_guards(p, report, events)
    findings = []
    for i, row in enumerate(report["rows"]):
        if row["status"] != "failed":
            continue
        need(all(row["selected_plan"].get(k) == v for k, v in p["plans"][i].items())
             and set(row["selected_plan"]) == set(p["plans"][i]) | {"arm"}
             and row["selected_plan"]["arm"] == row["arm"] and not row["full_horizon"]
             and row["first_error"]["message"] == "RGB/canvas/state/pure-render guard; failed image retained",
             "failed row identity")
        local = [event for event in events if event.get("branch") == i]
        trans = [e for e in local if e["kind"] == "returned_transition" and e["phase"] == "plan"]
        guards = [e for e in local if e["kind"] == "guard_receipt" and e["phase"] == "plan"]
        images = [e["receipt"] for e in local if e["kind"] == "fresh_observation"
                  and e["receipt"]["phase"] == "plan"]
        n = row["own_returned"]
        need(n == row["own_attempted"] == len(trans) == 17
             and [t["step"] for t in trans] == list(range(1, n + 1))
             and all(t["command"] == p["plans"][i]["commands"][j] for j, t in enumerate(trans))
             and trans[-1]["state"] == row["final_state"], "physical action sequence")
        rejected = [g for g in guards if not g["accepted"]]
        need(len(rejected) == 1 and rejected[0]["step"] == n and images[-1]["step"] == n
             and images[-1]["state"] == trans[-1]["state"] == images[-1]["observation_state"],
             "failed guard identity")
        image = images[-1]
        path = directory / image["png"]
        need(path.name == image["png"] and host.sha(path) == image["file_sha256"], "frame hash")
        with Image.open(path) as im:
            pixels = np.asarray(im).copy()
        need(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8
             and hashlib.sha256(pixels.tobytes()).hexdigest() == image["pixel_sha256"],
             "frame pixels")
        count = int(np.count_nonzero(np.all(pixels == [65, 105, 225], axis=-1)))
        xy = np.asarray(image["state"][:4])
        inside = bool(((xy >= 0) & (xy <= 512)).all())
        need(count == rejected[0]["canonical_agent_pixels"]
             and rejected[0]["render_preserved_dynamics"] is True
             and not (count > 0 and inside), "guard recomputation")
        need(not any(e["kind"] == "operation_attempted" and e.get("resource") == "public_step"
                     and e.get("phase") == "plan" and e.get("step", 0) > n for e in local),
             "no post-guard dispatch")
        findings.append({"branch": i, "identity": p["identities"][i], "returned": n,
                         "position": image["state"][:2], "in_canvas": inside,
                         "canonical_agent_pixels": count, "fixed_horizon_endpoint": None})
    need(findings and host.sha(report_path) == expected, "missing incidents or report drift")
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    need(output.is_absolute() and output.parent == ROOT / "results"
         and output.name.startswith(NAME + "_") and not output.exists()
         and not output.is_symlink(), "exclusive output")
    need(host.sha(summary.MANIFEST) == summary.MANIFEST_SHA, "manifest hash")
    manifest, data = summary.source.build()
    need(json.loads(summary.MANIFEST.read_bytes()) == manifest, "manifest rebuild")
    results = []
    for i, digest in REPORTS.items():
        folder = ROOT / f"results/p3_h3_real_context_physical_20261031_batch{i:02d}_RUN01"
        results.append({"batch": i, "report_sha256": digest,
                        "incidents": inspect_failed(folder, digest, summary.source.prepare_batch(manifest, i), data)})
    need(all(host.sha(path) == digest for path, digest in manifest["bindings"].items()), "source drift")
    result = {"schema": NAME, "batches": results, "P3_admission": False, "P1_admission": False,
              "limitations": "Incident verification only; other branches not fully consumed; "
                             "known roots, no imputed endpoints or equal-budget claim."}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"sha256": host.sha(output), "incidents": sum(len(x["incidents"]) for x in results)}))


if __name__ == "__main__":
    main()
