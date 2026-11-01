"""Recheck saved fixed-prefix CUDA successors against their original same-action panel."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import codex_p1_fixed_prefix_suffix_cuda_20261030_main as run

REPORT_SHA = "b4a46a861ce850603261ce7c52f55f3ff57b83873d2ccbb20c4f16ec9c871762"
DIRECTORY = run.ROOT / "results/codex_p1_fixed_prefix_suffix_cuda_20261030_main_HOST01"
NAME = "codex_p1_fixed_prefix_suffix_recheck_20261030_main"


def check():
    report = run.source.saved.read_verified(DIRECTORY / "report.json", REPORT_SHA)
    original, _, fixed = run.inputs()
    run.spec.require(report["status"] == "complete" and len(report["rows"]) == len(fixed) == 8,
                     "full roster")
    result = []
    for i, (summary, row) in enumerate(zip(report["rows"], fixed, strict=True)):
        path = DIRECTORY / f"node_{i:02d}/trace.jsonl"
        run.spec.require(run.source.host.F.sha(path) == summary["trace_sha256"], "trace hash")
        events = [json.loads(line) for line in path.read_text().splitlines()]
        previous = None
        for j, event in enumerate(events):
            body = {key: event[key] for key in ("index", "previous", "kind", "payload")}
            run.spec.require(event["index"] == j and event["previous"] == previous and
                             event["sha256"] == run.journal.digest(body), "trace chain")
            previous = event["sha256"]
        run.spec.require([e["kind"] for e in events] == ["inputs"] + ["optimized"] * 4 + ["selection"],
                         "trace event roster")
        run.spec.require(summary["indices"] == row["indices"] and summary["arm"] == row["arm"] and
                         summary["slot"] == row["slot"], "row identity")
        errors = []
        for j, candidate_index in enumerate(row["indices"]):
            candidate = original["rows"][i]["result"]["candidates"][candidate_index]
            actual = events[j + 1]["payload"]
            run.spec.require(actual["index"] == j and
                             np.array_equal(np.asarray(events[0]["payload"]["prefixes"][j], np.float32).reshape(5, 2),
                                            np.asarray(candidate["prefix"], np.float32)), "prefix binding")
            got = np.asarray(actual["successor"], np.float32).reshape(3, 192)
            expected = np.asarray(candidate["mid"]["latent_history"], np.float32)
            got_history = np.asarray(actual["history"], np.float32).reshape(2, 10)
            expected_history = np.asarray(candidate["mid"]["action_history"], np.float32)
            run.spec.require(np.allclose(got, expected, rtol=1e-5, atol=1e-6) and
                             np.allclose(got_history, expected_history, rtol=1e-5, atol=1e-6),
                             "same-action successor or history mismatch")
            errors.append({"index": candidate_index, "latent_max_abs": float(np.max(np.abs(got - expected))),
                           "history_max_abs": float(np.max(np.abs(got_history - expected_history)))})
        final = events[-1]["payload"]
        run.spec.require(final["scores"] == summary["scores"] and final["index"] == summary["selected_index"]
                         and final["index"] == min(range(4), key=lambda j: (final["scores"][j], j)),
                         "saved selection binding")
        result.append({"slot": row["slot"], "arm": row["arm"], "candidates": errors})
    return {"schema": NAME, "report_sha256": REPORT_SHA, "rows": result, "candidate_checks": 32,
            "physical_execution": False, "P1_admission": False,
            "limitations": "Saved same-action model successor check, not independent model execution or physics."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    path = Path(parser.parse_args().output)
    run.spec.require(path.is_absolute() and path.parent == run.ROOT / "results" and
                     path.name.startswith(NAME + "_") and not path.exists() and not path.is_symlink(),
                     "exclusive output")
    result = check()
    digest = run.write_new(path, result)
    print(json.dumps({"candidate_checks": result["candidate_checks"], "sha256": digest}))


if __name__ == "__main__":
    main()
