"""Pair Q/L2 feedback runs by actual cp5 RGB, action history and fixed prefix."""
from __future__ import annotations

import argparse
import json

import numpy as np

from experiments.spectral_reachability import reacher_d2p_context as context
from experiments.spectral_reachability import reacher_pc_p1_fd_cp5_host_20261102_codexexclusive as host
from experiments.spectral_reachability import reacher_pc_p1_fd_cp5_saved_20261102_codexexclusive as saved


def paired(l2_events, q_events, l2_results, q_results):
    keys = [(seed, f"prefix_{index:02d}") for seed in (20263160, 20263161) for index in range(13)]
    contexts = []
    for events in (l2_events, q_events):
        entries = [e for e in events if e["kind"] == "context" and e["role"] == "successor"]
        host.require([(e["seed"], e["arm"]) for e in entries] == keys, "successor roster")
        contexts.append(entries)
    expected = [(seed, index) for seed in (20263160, 20263161) for index in range(13)]
    host.require([(r["seed"], r["prefix"]) for r in l2_results] == expected
                 and [(r["seed"], r["prefix"]) for r in q_results] == expected, "result roster")
    out = []
    for left, right, l2, q in zip(contexts[0], contexts[1], l2_results, q_results, strict=True):
        lv, rv = left["value"], right["value"]
        left_frames = [context.unpack_rgb_observation(x) for x in lv["frames"]]
        right_frames = [context.unpack_rgb_observation(x) for x in rv["frames"]]
        host.require(len(left_frames) == len(right_frames) == 3
                     and all(np.array_equal(a, b) for a, b in zip(left_frames, right_frames, strict=True))
                     and np.array_equal(context.unpack_rgb_observation(lv["goal"]),
                                        context.unpack_rgb_observation(rv["goal"]))
                     and np.array_equal(np.asarray(lv["history"], dtype=np.float32),
                                        np.asarray(rv["history"], dtype=np.float32)), "cp5 context mismatch")
        host.require(l2["initial"] == q["initial"] and l2["cp5"] == q["cp5"], "paired physical state mismatch")
        out.append({"seed": l2["seed"], "prefix": l2["prefix"],
                    "cp5_rgb_sha256": context.rgb_digest(left_frames[-1]),
                    "initial": l2["initial"], "cp5": l2["cp5"],
                    "l2_final": l2["final"], "q_final": q["final"],
                    "q_minus_l2_final": q["final"] - l2["final"],
                    "same_raw_suffix": l2["suffix_raw"] == q["suffix_raw"]})
    return {"status": "same_saved_cp5_rgb_and_history", "rows": out,
            "independent_confirmation": False, "P1_admission": False,
            "scope": "two exposed roots; saved bytes only; no independent physical replay"}


def main():
    parser = argparse.ArgumentParser()
    for key in ("l2-journal", "l2-journal-sha", "q-journal", "q-journal-sha",
                "l2-saved", "l2-saved-sha", "q-saved", "q-saved-sha", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    for path, sha in ((args.l2_journal, args.l2_journal_sha), (args.q_journal, args.q_journal_sha),
                      (args.l2_saved, args.l2_saved_sha), (args.q_saved, args.q_saved_sha)):
        host.require(host.bridge.sha_file(path) == sha, "source SHA mismatch")
    l2 = json.loads(open(args.l2_saved).read())
    q = json.loads(open(args.q_saved).read())
    host.require(l2["status"] == q["status"] == "complete_saved_consistency"
                 and l2["journal_sha256"] == args.l2_journal_sha
                 and q["journal_sha256"] == args.q_journal_sha, "saved receipt binding")
    report = paired(saved.verify_journal(args.l2_journal), saved.verify_journal(args.q_journal),
                    l2["results"], q["results"])
    host.write_new(args.output, report)
    print(json.dumps({"status": report["status"], "rows": len(report["rows"]),
                      "sha256": host.bridge.sha_file(args.output)}))


if __name__ == "__main__":
    main()
