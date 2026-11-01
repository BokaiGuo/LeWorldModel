"""Compare every forced-prefix RGB-feedback diagnostic without promoting one."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_candidate_feedback_20261101_main as host
from . import p1_horizon_q_saved_20261101_main as saved
from . import p1_candidate_intervention_analysis_20261101_main as physical


def analyze():
    engine = host.physical.engine
    reference = json.loads((host.physical.ROOT /
        "results/p1_candidate_intervention_20261101_main_run_MAIN01/report.json").read_bytes())
    initial_report = json.loads((host.physical.DIRECTORY / "report.json").read_bytes())
    first = initial_report["rows"][0]
    receipt = next(n["receipt"] for n in initial_report["nodes"][first["episode_id"]] if n["boundary"] == 0)
    events = [json.loads(line) for line in Path(receipt["trace"]).read_bytes().splitlines()]
    bank = np.asarray(next(e["payload"]["raw"] for e in events if e["kind"] == "optimized_candidates"), np.float32)
    sources, rows = {}, []
    for index in range(8):
        directory = host.ROOT / f"results/{host.NAME}_c{index}_RUN_MAIN01"
        report = json.loads((directory / "report.json").read_bytes())
        sources[str(directory / "report.json")] = engine.sha(directory / "report.json")
        engine.require(engine.sha(directory / "journal.jsonl") == report["journal_sha256"],
                       "physical journal bytes")
        journal = engine.journal_reader.journal((directory / "journal.jsonl").read_bytes(),
                                                 report["journal_chain_tail"])
        image_count = 0
        for event in journal:
            if event["kind"] != "fresh_observation":
                continue
            cp_image = event["receipt"]
            path = Path(cp_image["path"])
            engine.require(path.parent == directory and engine.sha(path) == cp_image["file_sha256"],
                           "physical image file")
            with Image.open(path) as image:
                engine.require(engine.old.array_sha(np.asarray(image)) == cp_image["pixel_sha256"],
                               "physical image pixels")
            image_count += 1
        engine.require(len(report["rows"]) == 1, "one forced candidate per shard")
        row = report["rows"][0]
        count = min(5, row["control_returned"])
        engine.require(np.array_equal(np.asarray(row["executed_raw"][:count], np.float32),
                                      bank[index, 0, :count]), "same frozen prefix actions")
        nodes = report["nodes"][row["episode_id"]]
        node0 = next(n for n in nodes if n["boundary"] == 0)
        trace = Path(node0["receipt"]["trace"])
        engine.require(engine.sha(trace) == node0["receipt"]["trace_sha256"], "intervention trace")
        trace_events = [json.loads(line) for line in trace.read_bytes().splitlines()]
        forced = [e["payload"] for e in trace_events if e["kind"] == "forced_prefix_intervention"]
        engine.require(len(forced) == 1 and forced[0]["candidate"] == index
                       and forced[0]["source_trace_sha256"] == receipt["trace_sha256"],
                       "declared forced selection")
        cp = {c["step"]: c for c in row["checkpoints"]}
        goal = cp[0]["goal"]
        expected = next(r for r in reference["rows"] if r["candidate"] == index and r["suffix_role"] == "own")
        if 5 in cp:
            previous = next(c for c in expected["observations"] if c["step"] == 5)
            engine.require(all(cp[5][k] == previous[k] for k in ("state", "snapshot", "pixel_sha256")),
                           "actual cp5 equality across physical interventions")
        complete = report["status"] == "recorded_known_root_not_admitted"
        validation = saved.check(directory) if complete else None
        partial_validation = None
        if not complete:
            engine.require(row["first_error"] == dict(message="fresh RGB/canvas guard",
                                                      stage="episode", type="Rejected")
                           and row["control_returned"] <= 5, "recognized prefix domain failure")
            last = row["last_observation"]
            engine.require(all(last[k] == expected["last_observation"][k]
                               for k in ("state", "snapshot", "pixel_sha256"))
                           and any(not 0 <= value <= 512 for value in last["state"][:4]),
                           "same independently replayed out-of-domain prefix")
            partial_validation = "same_action_state_RGB_domain_failure_as_prior_physical_intervention"
        rows.append(dict(candidate=index, report_status=report["status"], status=row["status"],
                         physical_steps=row["control_returned"], native_success=row["native_success"],
                         first_error=row["first_error"], original_model_index=forced[0]["original_model_index"],
                         forced_selection=True, full_saved_check=validation,
                         checked_images=image_count, partial_validation=partial_validation,
                         prefix_delta=physical.delta(physical.physical(cp[5]["state"], goal),
                                                     physical.physical(cp[0]["state"], goal)) if 5 in cp else None,
                         checkpoints=[dict(step=k, **physical.physical(c["state"], goal)) for k, c in sorted(cp.items())],
                         missing_cp5_reason=None if 5 in cp else "prefix stopped or failed; do not impute"))
    return dict(rows=rows, sources=sources, fixed_roster=8, P1_admission=False,
                autonomous_method_success_claim=False, independent_confirmation=False,
                caution="Forced candidates, known root, own carried tails. Partial shards are not full saved-check passes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(rows=len(result["rows"]), output=str(args.output))))
