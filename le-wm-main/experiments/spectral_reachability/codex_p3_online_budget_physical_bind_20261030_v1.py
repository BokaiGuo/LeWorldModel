"""Bind all twelve online-budget winners to physical task sources; no dispatch."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
from experiments.spectral_reachability import codex_p3_online_budget_host_20261030_v1 as host
from experiments.spectral_reachability import moving_history_budget_scheduler_cuda_20261027_v1 as source
from experiments.spectral_reachability import p1_moving_baseline_physical_20261027_v1 as physical

NAME = "codex_p3_online_budget_physical_bind_20261030_v1"
RUN = host.ROOT / "results/codex_p3_online_budget_host_20261030_v1_RUN_01"
REPORT_SHA = "7d2ff7ef0c554077cf0a2461ab733d80f0f14a5658bdf1c554fd58ac6f4dacde"
PREP_SHA = "bfab85393221afe02f8ac804130040c4f493d5d3deffdeb5486e14cc6e5cb305"
RECEIPT = host.ROOT / "results/codex_p3_online_budget_host_20261030_v1_CONSUME_01.json"
RECEIPT_SHA = "03a3eaa5f7cce26298ab578440c02dac87b7d984ab0e2cb3eccfdc07c28adc67"


def reconstruct(checked, slots, references):
    host.require(len(slots) == len(references) == 2 and
                 [(x["point"], x["arm"]) for x in checked["rows"]] ==
                 [(p, a) for p in (20, 30) for a in ("uniform", "signal", "reverse")],
                 "complete fixed two-point roster")
    rows = []
    for item in checked["rows"]:
        for root in (0, 1):
            slot, ref, winner = slots[root], references[root], item["winners"][root]
            host.require(slot["root_sha256"] == ref["root_sha256"], "source root identity")
            raw = host.prior.prior.raw_commands(winner["commands"])
            host.require(raw.shape == (10, 2), "raw10")
            rows.append(dict(point=item["point"], policy=item["arm"], root=root,
                             root_sha256=slot["root_sha256"], task_id=slot["task_id"],
                             candidate_index=winner["index"], model_cost=winner["cost"],
                             raw_float32_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                             commands=raw.astype(float).tolist(),
                             arm_resource_attempts=item["attempted"], physical_authorized=False))
    return rows


def build():
    host.require(host.prior.sha(RUN / "report.json") == REPORT_SHA and
                 host.prior.sha(RUN / "prepare.json") == PREP_SHA, "producer hashes")
    report = host.prior.prior.read(RUN / "report.json")
    host.require(report["status"] == "panel_completed_unverified" and report["error"] is None,
                 "complete CUDA producer")
    # Verify in a clean interpreter: importing this bridge changes loaded-source inventory.
    program = ("from pathlib import Path; "
               "from experiments.spectral_reachability import codex_p3_online_budget_host_20261030_v1 as h; "
               "o=Path(%r); r=h.prior.prior.read(o/'report.json'); "
               "h.require(h.prior.sha(o/'report.json')==%r,'report SHA'); "
               "p,t,b=h.verify_prepare(o/'prepare.json',r['prepare_sha256']); "
               "v=h.consumer.consume(o/'events.jsonl',r['journal_tail'],h.CAP,t,b); "
               "print(h.journal.digest(v))") % (str(RUN), REPORT_SHA)
    digest = subprocess.check_output([sys.executable, "-B", "-c", program], text=True).strip()
    host.require(host.prior.sha(RECEIPT) == RECEIPT_SHA, "pinned consumer receipt")
    host.require(digest == RECEIPT_SHA, "receipt equals fresh independent reconstruction")
    checked = host.prior.prior.read(RECEIPT)
    host.require(checked["status"] == "semantic_saved_checked_not_scientifically_admitted",
                 "consumer receipt status")
    prepared = host.prior.prior.read(host.PREVIOUS)
    data = physical.kernel.load_source(source.SOURCEPACK, source.SOURCEPACK_SHA,
                                       source.SOURCE_PREPARE, source.SOURCE_PREPARE_SHA)
    bindings = {}
    for root, slot in enumerate(prepared["slots"]):
        physical.task_bridge(slot, data["references"][root], bindings)
    rows = reconstruct(checked, prepared["slots"], data["references"])
    return dict(schema=NAME, status="saved_source_join_only_not_execution_ready",
                producer_report_sha256=REPORT_SHA, producer_prepare_sha256=PREP_SHA,
                sourcepack_sha256=source.SOURCEPACK_SHA, original_task_export_bindings=bindings,
                rows=rows, physical_execution=False, physical_authorized=False, scientific_pass=False,
                physical_reuse_or_repetition="not fixed", source_closure="unknown")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    host.require(path.is_absolute() and path.parent == host.ROOT / "results" and
                 path.name.startswith(NAME + "_") and not path.exists() and not path.is_symlink(),
                 "exclusive binding output")
    print(host.write_new(path, build()))


if __name__ == "__main__":
    main()
