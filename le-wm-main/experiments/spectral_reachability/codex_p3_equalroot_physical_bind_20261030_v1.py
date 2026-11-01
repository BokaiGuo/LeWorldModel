"""Saved-only binding of six no-shadow winners to the original moving source.

No model, environment or physical dispatch. This is not a physical authorization.
"""
from collections import Counter
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np

from experiments.spectral_reachability import codex_p3_equalroot_host_20261030_v1 as host
from experiments.spectral_reachability import codex_p3_quota_journal_20261030_v1 as journal
from experiments.spectral_reachability import moving_history_budget_scheduler_cuda_20261027_v1 as m
from experiments.spectral_reachability import p1_moving_baseline_physical_20261027_v1 as physical

NAME = "codex_p3_equalroot_physical_bind_20261030_v1"
RUN = host.ROOT / "results/codex_p3_equalroot_host_20261030_v1_RUN_01"
REPORT_SHA = "7c1e389681172f897a28463760fe64cbb670fe5c31e6d74a4ecd20aed02534e6"
PREP_SHA = "9cdfff674690bce149eef8a4d09329f0f666be02e1b9be8a7ede40c8490fd252"
POLICIES = ("uniform", "signal", "reverse")
AXES = ("action_encoder", "predict", "predictor", "pred_proj")


def summarize(events, slots, references):
    require = host.require
    require(len(slots) == len(references) == 2, "two roots")
    by_policy = [e for e in events if e.get("kind") == "arm_complete"]
    require([e["policy"] for e in by_policy] == list(POLICIES), "three arms")
    attempts = [e for e in events if e.get("kind") == "resource_attempt"]
    returns = [e for e in events if e.get("kind") == "resource_return"]
    metrics = [e for e in events if e.get("kind") == "metric_return"]
    result = []
    for root in range(2):
        require(slots[root]["root_sha256"] == references[root]["root_sha256"], "source root")
        for e in by_policy:
            policy = e["policy"]
            winners = e["winners"]
            require(len(winners) == 2 and [w["group"] for w in winners] == [0, 1], "winner order")
            w = winners[root]
            require(w["policy"] == policy and w["eligible"] is True, "winner policy")
            raw = host.prior.prior.raw_commands(w["commands"])
            require(raw.shape == (10, 2) and raw.dtype == np.dtype("<f4"), "literal raw10")
            scope = lambda x: x["scope"][:2] == [policy, root]
            counts = Counter(x["entry"]["resource"] for x in attempts if scope(x))
            done = Counter(x["entry"]["resource"] for x in returns if scope(x))
            nmetric = sum(1 for x in metrics if scope(x))
            require(set(counts) == set(AXES) and counts == done and
                    all(counts[k] == 20 for k in AXES) and nmetric == 10,
                    "separate actual per root axis census")
            result.append(dict(root=root, policy=policy, task_id=slots[root]["task_id"],
                root_sha256=slots[root]["root_sha256"], candidate_index=w["index"],
                candidate_sha256=w["candidate_sha256"], raw_float32_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                commands=raw.astype(float).tolist(), model_cost=w["cost"],
                actual_resource_attempts=dict(counts), actual_resource_returns=dict(done),
                actual_metric_returns=nmetric, physical_authorized=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    host.require(output.is_absolute() and output.parent == host.ROOT / "results" and
                 output.name.startswith(NAME + "_") and not output.exists() and not output.is_symlink(),
                 "exclusive new output")
    host.require(host.prior.sha(RUN / "report.json") == REPORT_SHA and
                 host.prior.sha(RUN / "prepare.json") == PREP_SHA, "pinned actual producer")
    subprocess.run([sys.executable, "-B", "-m",
                    "experiments.spectral_reachability.codex_p3_equalroot_host_20261030_v1",
                    "consume", "--run", str(RUN), "--sha256", REPORT_SHA],
                   check=True, stdout=subprocess.DEVNULL)
    report = host.prior.prior.read(RUN / "report.json")
    prepared = host.prior.prior.read(host.PREVIOUS)
    data = physical.kernel.load_source(m.SOURCEPACK, m.SOURCEPACK_SHA, m.SOURCE_PREPARE, m.SOURCE_PREPARE_SHA)
    slots = prepared["slots"]
    bindings = {}
    for root, slot in enumerate(slots):
        physical.task_bridge(slot, data["references"][root], bindings)
    events = journal.read(RUN / "events.jsonl", report["journal_tail"], host.CAP)
    rows = summarize(events, slots, data["references"])
    result = dict(schema=NAME, status="saved_source_join_only_not_execution_ready",
                  producer_report_sha256=REPORT_SHA, producer_prepare_sha256=PREP_SHA,
                  sourcepack_sha256=m.SOURCEPACK_SHA, original_task_export_bindings=bindings,
                  rows=rows, signal_accounting="historically acquired once, shared across arms; independent per-arm acquisition not measured",
                  signal_allocation="not invented; per-root and per-arm full acquisition cost unknown",
                  full_dynamic_source_closure="unknown", physical_factory_consumer="new six-slot adapter missing",
                  physics_step_budget="not yet fixed for new six-slot adapter", physical_execution=False,
                  physical_authorized=False, scientific_pass=False)
    print(host.write_new(output, result))


if __name__ == "__main__":
    main()
