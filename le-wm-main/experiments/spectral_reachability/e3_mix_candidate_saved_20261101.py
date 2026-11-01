"""Validate fixed candidate actions and summarize observed physical outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_four_prefix_saved_audit_20261030 as audit
from . import e3_initial_goal_qualification_20261031 as qualification
from . import e3_mix_candidate_physical_20261101 as experiment

NAME = "e3_mix_candidate_saved_20261101"


def check(prepared, prep_sha, directory, report_sha):
    prepared, directory = Path(prepared), Path(directory)
    need, sha = audit.need, audit.sha
    need(sha(prepared) == prep_sha and sha(directory / "report.json") == report_sha, "hash")
    prep = json.loads(prepared.read_bytes())
    report = json.loads((directory / "report.json").read_bytes())
    need(report["status"] == "recorded_not_independently_consumed"
         and report["prepare_sha256"] == prep_sha and not report["errors"]
         and not report["pending"] and not report["secondary_errors"]
         and len(report["rows"]) == len(prep["schedule"]) == 24, "report contract")
    need(all(sha(path) == digest for path, digest in prep["bindings"].items()), "source drift")
    journal = directory / "journal.jsonl"
    need(sha(journal) == report["journal_sha256"], "journal hash")
    events = audit.journal(journal, report["journal_chain_tail"])
    rows = []
    for index, (plan, row) in enumerate(zip(prep["schedule"], report["rows"], strict=True)):
        need(plan["index"] == row["branch"] == index and plan["candidate"] == index // 3
             and plan["suffix_role"].endswith(("two_packets", "first_packet_zero", "zero"))
             and row["initial_exact"] and row["prefix_verified"] and not row["first_error"]
             and row["status"] in ("completed15", "native_stopped", "censored_out_of_domain"),
             "branch identity")
        n = row["control_returned"]
        need(np.array_equal(np.asarray(row["executed_raw"], np.float32),
                            np.asarray(plan["commands"][:n], np.float32)), "actions")
        scope = [e for e in events if e.get("episode_id") == row["episode_id"]]
        frames = [e["receipt"] for e in scope if e["kind"] == "fresh_observation"
                  and e["receipt"]["phase"] == "plan"]
        need([f["step"] for f in frames] == list(range(n + 1))
             and all(frames[5][key] == value for key, value in plan["expected_prefix_checkpoint"].items()),
             "frame sequence / prefix")
        for frame in frames:
            path = Path(frame["path"])
            need(path.parent == directory and not path.is_symlink()
                 and sha(path) == frame["file_sha256"], "image hash")
            with Image.open(path) as image:
                pixels = np.asarray(image)
            need(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8
                 and hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()
                 == frame["pixel_sha256"], "pixel hash")
        goal = np.asarray(frames[0]["goal"])
        accepted = row["status"] != "censored_out_of_domain"
        last = frames[-1]["state"] if accepted else None
        metric = qualification.native(goal, last) if last is not None else None
        if row["status"] == "native_stopped":
            need(row["transitions"][-1]["terminated"] or row["transitions"][-1]["truncated"],
                 "stop flag")
        rows.append({"branch": index, "candidate": plan["candidate"], "winner": plan["winner"],
                     "variant": plan["suffix_role"], "model_h15_score": plan["model_h15_score"],
                     "status": row["status"], "returned_steps": n,
                     "last_accepted_metrics": metric,
                     "observed_h10_metrics": metric if accepted and n == 15 else None})
    return {"schema": NAME, "status": "saved_checks_passed", "rows": rows,
            "prepare_sha256": prep_sha, "report_sha256": report_sha,
            "physical_event_chain_independently_replayed": False, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "prep-sha", "directory", "report-sha", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    audit.need(output.is_absolute() and output.parent == experiment.ROOT / "results"
               and output.name.startswith(NAME + "_") and not output.exists()
               and not output.is_symlink(), "exclusive output")
    data = check(args.prepared, args.prep_sha, args.directory, args.report_sha)
    payload = json.dumps(data, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    with output.open("xb") as stream:
        audit.need(stream.write(payload) == len(payload), "write")
    print(json.dumps({"status": data["status"], "rows": len(data["rows"]),
                      "sha256": audit.sha(output)}))


if __name__ == "__main__":
    main()
