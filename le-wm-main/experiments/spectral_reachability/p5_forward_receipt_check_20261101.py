"""Read-only, CPU-only revalidation of a saved P5 supervised acceptance."""
import argparse
import json
import re

from experiments.spectral_reachability import p5_rgb_forward_probe_supervision_20261030_codex_v1 as s


def check(suffix, preflight_sha, supervisor_sha, journal_sha, acceptance_sha):
    for value in (preflight_sha, supervisor_sha, journal_sha, acceptance_sha):
        s.need(re.fullmatch(r"[0-9a-f]{64}", value) is not None, "external SHA")
    supervision, run = s.paths(suffix)
    s.need(supervision.is_dir() and not supervision.is_symlink(), "supervisor path")
    base, prepared = s.preflight()
    raw = s.wiring.acc.read_new_file(supervision / "supervisor.jsonl", supervisor_sha)
    s.need(raw.endswith(b"\n"), "supervisor truncated")
    lines = raw.splitlines()
    events = [s.bridge.read_json(line) for line in lines]
    s.need(all(s.bridge.canonical(e).rstrip(b"\n") == line
               for e, line in zip(events, lines, strict=True)), "supervisor canonical")
    s.need(events and events[0]["preflight_sha256"] == preflight_sha, "preflight external SHA")
    report = s.hostgate.load_report(events[0]["preflight_name"], preflight_sha)
    expected_review = s.validate_supervisor(events, suffix, s.source_sha(), report)
    review = s.checked_review(run, base, prepared, journal_sha)
    s.need(review == expected_review, "review mismatch")
    path = s.RESULTS / (s.NAME + "_accept_" + suffix + ".json")
    acceptance_raw = s.wiring.acc.read_new_file(path, acceptance_sha)
    acceptance = s.bridge.read_json(acceptance_raw)
    expected = {"schema": s.NAME + ".accept", "status": "saved_self_report_consistency_not_scientific",
                "producer_sha256": s.source_sha(), "probe_sha256": s.PROBE_SHA,
                "contract_sha256": s.PREP_SHA, "preflight_sha256": preflight_sha,
                "supervisor_sha256": supervisor_sha, "review": review, "P5_admission": False,
                "boundary": "self-reported execution; finite differences not theoretical causality or fresh confirmation"}
    s.need(acceptance_raw == s.bridge.canonical(expected) and acceptance == expected,
           "acceptance content mismatch")
    return {"status": "saved_receipt_consistent_not_scientific", "suffix": suffix,
            "acceptance_sha256": acceptance_sha, "journal_sha256": journal_sha,
            "gpu_queried": False, "P5_admission": False,
            "boundary": "does not independently establish service or writer termination"}


def main():
    parser = argparse.ArgumentParser()
    for key in ("suffix", "preflight-sha", "supervisor-sha", "journal-sha", "acceptance-sha"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.suffix, args.preflight_sha, args.supervisor_sha,
                           args.journal_sha, args.acceptance_sha), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
