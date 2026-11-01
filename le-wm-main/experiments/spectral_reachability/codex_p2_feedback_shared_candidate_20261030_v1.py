"""Build saved same-feedback candidate pairs; no neural or physical execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import agent_p2_full30_control15_feedback_summary_20261029_v1 as source

NAME = "codex_p2_feedback_shared_candidate_20261030_v1"
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
PATHS = {
    "prepare": RESULTS / "agent_p2_full30_control15_feedback_20261029_v2_prepare_MAIN01.json",
    "report": RESULTS / "agent_p2_full30_control15_feedback_20261029_v2_cuda_MAIN01/report.json",
    "completion": RESULTS / "agent_p2_full30_control15_feedback_20261029_v2_cuda_MAIN01/completion.json",
    "receipt": RESULTS / "agent_p2_full30_control15_feedback_20261029_v2_consume_MAIN01.json",
}
SUMMARY = RESULTS / "agent_p2_full30_control15_feedback_summary_20261029_v1_MAIN01.json"
SUMMARY_SHA = "f2559fe07093b084ca0529a0cc458048183e5d0be7f2a7053a8ec451359586e6"


def make_pair(job, row, summary_job):
    require = source.require
    require(job["policy"] == row["policy"] == summary_job["policy"] == "F", "actual feedback job")
    require(job["base_index"] == row["base_index"] == summary_job["base_index"], "base identity")
    result = row["result"]
    require(result["input_context_sha256"] == source.digest(job["input_context"]) ==
            summary_job["input_context_sha256"], "same full feedback context")
    es = result["evaluations"]
    require(len(es) == 5 and [e["index"] for e in es] == list(range(5)), "fixed candidates")
    selected = result["selected"]
    require(type(selected) is int and 0 <= selected < 5 and selected == summary_job["original_selected"],
            "source selection")
    require(selected == min(range(5), key=lambda i: (es[i]["Q"], i)), "stable source selection")
    prefix = source.array(job["prefix_raw"], (15, 2), True).astype("<f4")
    values = []
    for label, index in (("selected", selected), ("fixed_index_1", 1)):
        e = es[index]
        require(e["logits"] == result["proposals"]["logits"][index], "candidate witness")
        require(source.scores(e["contexts"][-1], job["goal"])[1] == e["Q"] ==
                summary_job["candidates"][index]["Q_numpy_original"], "same-context saved score")
        suffix = source.chart(e["logits"]).reshape(15, 2)
        raw = np.concatenate((prefix, suffix))
        require(np.isfinite(raw).all() and np.all(np.abs(raw) <= 1), "raw action domain")
        values.append(dict(arm=label, candidate_index=index, model_Q=e["Q"],
                           raw30=raw.tolist(), raw_float32_hex=raw.tobytes().hex(),
                           raw_float32_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                           physical_outcome=None))
    require(values[0]["raw_float32_hex"] == result["raw_float32_bits"], "selected literal bits")
    return dict(base=job["base_index"], root=job["root_slot"], source_arm=job["arm"],
                task_id=job["task_id"], task_sha256=job["task_sha256"],
                parent_witness_sha256=job["parent_witness_sha256"],
                feedback_context_sha256=result["input_context_sha256"],
                inherited_model_resources=summary_job["resources"], arms=values,
                equal_actions=values[0]["raw_float32_hex"] == values[1]["raw_float32_hex"])


def build():
    paths = {k: str(v) for k, v in PATHS.items()}
    summary = source.analyze(paths, source.PINS)
    source.same(source.read(str(SUMMARY), SUMMARY_SHA), summary, "existing full saved summary")
    prep = source.read(str(PATHS["prepare"]), source.PINS["prepare"])
    report = source.read(str(PATHS["report"]), source.PINS["report"])
    slots = [source.read(str(PATHS["report"].parent / f"slot{i}.json.gz"),
                         report["artifact_sha256"][f"slot{i}.json.gz"], True) for i in (1, 3, 5, 7)]
    pairs = [make_pair(prep["jobs"][2*i+1], slots[i], summary["jobs"][2*i+1]) for i in range(4)]
    return dict(schema=NAME, status="saved_candidate_recipes_not_physical",
                comparison="selected versus fixed candidate index 1 from same observed cp15 context and candidate set",
                pairs=pairs, source_summary_sha256=SUMMARY_SHA, source_pins=source.PINS,
                after_existing_results_known=True, preregistered=False,
                shared_model_work_not_two_independent_budgeted_runs=True,
                new_neural_calls=0, new_physical_calls=0, total_compute_equal=False,
                engine_backward=None, FLOPs=None, independent_confirmation=False, P2_admission=False,
                missing="candidate-1 physical replay and matching cp15 feedback guard not implemented here")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-sha")
    args = parser.parse_args()
    path = Path(args.output)
    source.require(path.is_absolute() and path.parent.resolve() == path.parent and
                   path.name.startswith(NAME + "_"), "own canonical output")
    data = source.raw(build())
    if args.verify_sha:
        source.same(source.read(str(path), args.verify_sha), source.decode(data), "whole recipe rebuilt")
        print(json.dumps(dict(status="saved_recipe_verified_not_physical", sha256=args.verify_sha)))
        return
    source.require(not os.path.lexists(path), "exclusive output")
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status="saved_candidate_recipes_not_physical", sha256=hashlib.sha256(data).hexdigest())))


if __name__ == "__main__":
    main()
