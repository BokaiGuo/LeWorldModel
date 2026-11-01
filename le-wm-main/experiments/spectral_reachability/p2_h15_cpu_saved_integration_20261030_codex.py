"""Historical H15 CPU integration diagnostic; never runs neural or physics."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import p2_durable_hook_driver_20261030_codex as durable
from experiments.spectral_reachability import p2_h15_cpu_metered_backend_20261030_codex as adapter
from experiments.spectral_reachability import p2_hook_contract_bridge_20261030_codex as upstream
from experiments.spectral_reachability import p2_preregistered_resource_vector_20261030_codex as spec

NAME = "p2_h15_cpu_saved_integration_20261030_codex"
ROOT = Path(__file__).resolve().parents[2]


def diagnostic_contract():
    # Only test data for an already observed run, not a future commitment.
    axes = {k: {"unit": unit, "aggregation": "max" if k == "peak_device_bytes" else "sum", "cap": 1000}
            for k, unit in spec.AXES.items()}
    axes["array_svd_calls"]["cap"] = 2
    axes["selection_evaluations"]["cap"] = 40
    return {"schema": spec.NAME + ".contract.v1", "arms": ["diagnostic", "unused"],
            "scope": "per_root_per_arm_standalone_planning_and_selected_physical_execution",
            "identity": {key: "a" * 64 for key in spec.IDENTITY}, "axes": axes}


def run(output):
    output = Path(output)
    core.require(output.is_absolute() and output.parent == ROOT / "results" and
                 output.name.startswith(NAME + "_") and not output.exists() and not output.is_symlink(),
                 "exclusive diagnostic output")
    host = spec.read_bound(upstream.HOST / "report.json", upstream.HOST_SHA)
    core.require(host["status"] == "recorded_not_admitted" and len(host["slots"]) == 2, "host roster")
    output.mkdir()
    fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    c = diagnostic_contract()
    digest = spec.sha(spec.canonical(c))
    results = []
    for root in (0, 1):
        slot = host["slots"][root]
        core.require(slot["root"] == root and slot["status"] == "complete", "slot")
        directory = upstream.HOST / f"root{root}"
        report = spec.read_bound(directory / "report.json", slot["report_sha256"])
        raw = (directory / "events.jsonl").read_bytes()
        core.require(hashlib.sha256(raw).hexdigest() == report["journal_sha256"], "source journal hash")
        events = durable.read_chain(raw)
        attempts = [x["payload"] for x in events if x["payload"].get("kind") == "native_call_attempt"]
        returns = [x["payload"] for x in events if x["payload"].get("kind") == "native_call_return"]
        core.require(len(attempts) == len(returns) == 43 and attempts[0]["method"] == "signal", "signal roster")
        goal, signal = attempts[0]["task"]["goal"], returns[0]["result"]
        d = adapter.make_driver(output / f"root{root}", c, digest, dict.fromkeys(core.AXES, 1000),
                                "diagnostic", f"historical-root-{root}")
        b = adapter.Backend(d)
        bank = b.compile(signal["jacobian"], signal["nominal"], goal)
        expected = report["result"]["bank"]
        core.require(bank["labels"] == expected["labels"] and
                     np.allclose(bank["logits"], expected["logits"], rtol=1e-11, atol=1e-11)
                     and np.allclose(bank["raw15"], expected["raw15"], rtol=0, atol=1e-6),
                     "saved bank mismatch")
        scores = []
        for i, row in enumerate(report["result"]["candidates"]):
            score = b.score_candidate(i, row["path"][-1]["latent_history"][-1], goal)
            core.require(np.isclose(score, row["score"], rtol=1e-13, atol=1e-13), "saved score mismatch")
            scores.append(score)
        winner = int(np.argmin(scores))
        core.require(winner == report["result"]["winner"], "saved winner mismatch")
        execution = b.finish()
        verification = durable.verify(execution["journal"], execution["sha256"], digest)
        core.require(verification["status"] == "complete" and
                     verification["completed"]["array_SVD_calls"] == 2 and
                     verification["completed"]["solve_calls"] == 1 and
                     verification["completed"]["metric_calls"] == 40, "CPU accounting")
        results.append({"root": root, "source_child_sha256": slot["report_sha256"],
                        "source_journal_sha256": report["journal_sha256"], "winner": winner,
                        "bank_and_40_saved_scores_match": True, "execution": execution, "verification": verification})
    result = {"schema": NAME + ".report.v1", "historical_host_sha256": upstream.HOST_SHA,
              "diagnostic_contract_sha256": digest, "contract_is_synthetic_and_retrospective": True,
              "rows": results, "no_neural_or_physical_calls": True,
              "whole_planning_coverage": False, "fairness": False, "P1_admission": False, "P2_admission": False}
    data = spec.canonical(result) + b"\n"
    path = output / "report.json"
    with path.open("xb") as stream:
        core.require(stream.write(data) == len(data), "complete report write")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return {"report": str(path), "sha256": hashlib.sha256(data).hexdigest(), "roots": len(results)}


def verify(output, expected_sha):
    output = Path(output)
    value = spec.read_bound(output / "report.json", expected_sha)
    digest = spec.sha(spec.canonical(diagnostic_contract()))
    core.require(value["schema"] == NAME + ".report.v1" and
                 value["historical_host_sha256"] == upstream.HOST_SHA and
                 value["diagnostic_contract_sha256"] == digest and
                 value["contract_is_synthetic_and_retrospective"] is True and
                 value["P1_admission"] is value["P2_admission"] is False and len(value["rows"]) == 2,
                 "saved report bindings")
    host = spec.read_bound(upstream.HOST / "report.json", upstream.HOST_SHA)
    adapter_sha = spec.sha(Path(adapter.__file__).read_bytes())
    for root, row in enumerate(value["rows"]):
        ex = row["execution"]
        path = output / f"root{root}/events.jsonl"
        core.require(row["root"] == root and row["source_child_sha256"] == host["slots"][root]["report_sha256"]
                     and ex["journal"] == str(path) and ex["adapter_sha256"] == adapter_sha,
                     "root, source and adapter binding")
        saved = spec.read_bound(upstream.HOST / f"root{root}/report.json", row["source_child_sha256"])
        core.require(saved["journal_sha256"] == row["source_journal_sha256"] and
                     saved["result"]["winner"] == row["winner"] and row["bank_and_40_saved_scores_match"] is True,
                     "saved source binding")
        checked = durable.verify(path, ex["sha256"], digest)
        core.require(checked == row["verification"] and checked["status"] == "complete" and
                     checked["completed"]["array_SVD_calls"] == 2 and
                     checked["completed"]["solve_calls"] == 1 and checked["completed"]["metric_calls"] == 40,
                     "journal recount")
        events = durable.read_chain(path.read_bytes())
        ops = [e["payload"]["entry"] for e in events if e["kind"] == "resource"
               and e["payload"]["kind"] == "resource_attempt"]
        core.require(all(x["scope"]["adapter_sha256"] == adapter_sha for x in ops), "adapter scopes")
        core.require(type(ex["cpu_adapter_wall_ns"]) is int and ex["cpu_adapter_wall_ns"] >= 0 and
                     type(ex["cpu_process_ns"]) is int and ex["cpu_process_ns"] >= 0 and
                     ex["FLOPs"] is None and ex["coverage_verified"] is False, "timing boundary")
    return {"verified": True, "report_sha256": expected_sha, "roots": len(value["rows"]),
            "bank_match_recomputed": False, "P1_admission": False, "P2_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-sha256")
    args = parser.parse_args()
    print(json.dumps(verify(args.output, args.verify_sha256) if args.verify_sha256 else run(args.output), sort_keys=True))


if __name__ == "__main__":
    main()
