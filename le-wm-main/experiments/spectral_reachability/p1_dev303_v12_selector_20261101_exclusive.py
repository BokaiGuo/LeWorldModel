"""Host-CUDA, outcome-free V12 selector for the sealed dev303 bank."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import p1_dev302_mixed_bank_cuda_20261101_codex as kernel
from . import p1_dev303_v12_adapter_20261101_exclusive as adapter
from . import p1_dev303_physical_prefix_bridge_20261101_exclusive as bridge

ROOT = adapter.ROOT
NAME = "p1_dev303_v12_selector_20261101_exclusive"
PREP = ROOT / "results/p1_dev303_v12_adapter_20261101_exclusive_PREP_MAIN01.json"
PREP_SHA = "f21eb1abe6517bf272f58e3a96af2291f829a4ebf6379add6c02d0fa09e2f1e0"
TASK = adapter.TASK
TASK_SHA = adapter.TASK_SHA


def require(value, message):
    if not value:
        raise ValueError(message)


def protocol():
    raw = adapter.prepare(
        ROOT / "results/p1_dev303_blind_planners_20261101_codex_RUN_MAIN02/report.json",
        adapter.REPORT_SHA,
    )
    require(adapter.PREP if hasattr(adapter, "PREP") else True, "adapter loaded")
    require(raw["schema"] == "p1_dev303_v12_adapter_20261101_exclusive.prepare.v1",
            "V12 preparation schema")
    value = dict(raw)
    value["schema"] = NAME + ".protocol.v1"
    value["algorithm"]["selected"] = "minimum robust score among RGB-feasible candidates"
    value["algorithm"]["same_bank_greedy"] = "minimum predicted cp5 latent goal MSE"
    value["bindings_sha256"][str(Path(__file__).resolve())] = kernel.sha(Path(__file__))
    return value


def run(prepared: Path, prepared_sha: str, output: Path) -> str:
    value = protocol()
    require(kernel.sha(prepared) == prepared_sha and json.loads(prepared.read_bytes()) == value,
            "selector preparation")
    require(value["candidate_status"] == "ready" and value["feasible_candidate_count"] > 0,
            "abstain before CUDA: no RGB-feasible candidate")
    old_name, old_protocol = kernel.NAME, kernel.protocol
    original_rng = kernel.np.random.default_rng
    kernel.NAME, kernel.protocol = NAME, protocol
    kernel.np.random.default_rng = lambda seed=None: original_rng(kernel.SEED)
    try:
        return kernel.run(prepared, prepared_sha, output)
    finally:
        kernel.np.random.default_rng = original_rng
        kernel.NAME, kernel.protocol = old_name, old_protocol


def normalize(output: Path, kernel_report_sha: str) -> str:
    report = json.loads((output / "report.json").read_bytes())
    require(report["schema"] == NAME + ".report.v1" and report["status"] == "model_only_complete",
            "sealed kernel report")
    rows = sorted(report["rows"], key=lambda row: row["index"])
    require([row["index"] for row in rows] == list(range(17)), "complete selector roster")
    scores = [{
        "robust_score": float(row["tail_selected"]["robust_score"]),
        "cp5_q": float(row["predicted_cp5_q"]),
    } for row in rows]
    feasible = [i for i, row in enumerate(rows) if row["contact_feasible_prior"] is True]
    require(feasible, "no feasible RGB candidate")
    role_indices = {
        "selected": min(feasible, key=lambda i: (scores[i]["robust_score"], i)),
        "zero": 0,
        "task_greedy_direct": 4,
        "random": 7,
    }
    source = bridge.verified_source()
    final = {
        "schema": bridge.REPORT_SCHEMA,
        "status": "model_only_sealed",
        "prepare_sha256": PREP_SHA,
        "source": source,
        "selection_input": "rgb_only",
        "simulator_state_used": False,
        "physical_outcomes_used": False,
        "P1_admission": False,
        "errors": [],
        "scores": scores,
        "role_indices": role_indices,
        "kernel_report_sha256": kernel_report_sha,
        "resources": report["attempts"],
        "hooks": report["hooks"],
        "model_before": report["model_before"],
        "model_after": report["model_after"],
    }
    payload = (json.dumps(final, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path = output / "selector_report.json"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return kernel.sha(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.tag.isalnum(), "tag")
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    require(not os.path.lexists(prepared) and (not args.run or not os.path.lexists(output)),
            "exclusive output")
    payload = (json.dumps(protocol(), sort_keys=True, indent=2) + "\n").encode()
    with prepared.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    digest = kernel.sha(prepared)
    print(json.dumps({"prepared": str(prepared), "sha256": digest}), flush=True)
    if args.run:
        require(json.loads(prepared.read_bytes())["candidate_status"] == "ready",
                "abstain before CUDA: no RGB-feasible candidate")
        with kernel.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            kernel_report_sha = run(prepared, digest, output)
        selector_sha = normalize(output, kernel_report_sha)
        print(json.dumps({"output": str(output), "kernel_report_sha256": kernel_report_sha,
                          "selector_report_sha256": selector_sha}), flush=True)


if __name__ == "__main__":
    main()
