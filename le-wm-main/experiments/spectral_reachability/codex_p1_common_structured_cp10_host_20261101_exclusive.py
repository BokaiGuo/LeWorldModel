"""Known-root cp10-goal structured host, isolated from the previously sealed host."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from . import codex_p1_common_structured_host_20261101_exclusive as previous
from . import e3_cp10_four_arm_root0_20261031 as cp10
from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as locking

host = previous.host
NAME = "codex_p1_common_structured_cp10_host_20261101_exclusive"
CEM_REPORT = host.ROOT / "results/p1_common_budget_cem_host_20261101_leibniz_root0_RUN_LEIBNIZ01/report.json"
CEM_REPORT_SHA = "28d1df63a0aa883256229d1a7fb5a9d2b78e96086f79b29ef9205703dee8a6c2"


def checked_target(prepared):
    """Rebind before protocol creation; compare pinned CEM and cp10 identities."""
    host.require(host.sha(CEM_REPORT) == CEM_REPORT_SHA, "CEM report SHA drift")
    cem = json.loads(CEM_REPORT.read_bytes())
    entry, target, sources = cp10.inputs()
    data = host.p3.source_data(prepared["source"])
    record = cp10.rebind(prepared, data)
    host.require(record == cem["cp10_goal_binding"], "CEM cp10 identity mismatch")
    checkpoints = cem["rows"][0]["checkpoints"]
    zero = [row for row in checkpoints if row["step"] == 0]
    host.require(len(zero) == 1 and zero[0]["goal"] == entry["goal_state"]
                 and zero[0]["goal_rgb_sha256"] == entry["pixel_sha256"]
                 and target["goal_state"] == entry["goal_state"], "CEM target state/RGB mismatch")
    host.require(prepared["source"]["expected_model_hash"] == cem["model_before"]["original_state_dict"]
                 == cem["model_after"]["original_state_dict"], "CEM model identity mismatch")
    host.require(host.digest(prepared["source"]["tasks"][0]["goal"]) == record["goal_latent_sha256"]
                 and prepared["source"]["tasks"][0]["binding"]["root_sha256"] == record["root_sha256"]
                 and np.array_equal(np.asarray(data["references"][0]["states"][-1], dtype=np.float64),
                                    np.asarray(entry["root_state"], dtype=np.float64)),
                 "source latent/root identity mismatch")
    return record, sources


def configure():
    original_build = host.build

    def build_target_first():
        prepared = original_build()
        binding, sources = checked_target(prepared)
        prepared["sources"].update(sources)
        prepared["sources"][str(CEM_REPORT)] = CEM_REPORT_SHA
        for module in (previous, cp10, locking):
            path = Path(module.__file__).resolve()
            prepared["sources"][str(path)] = host.sha(path)
        own = Path(__file__).resolve()
        prepared["sources"][str(own)] = host.sha(own)
        prepared["cp10_identity"] = binding
        return prepared

    host.build = build_target_first
    previous.configure()
    host.NAME = NAME
    old_build, old_protocol, old_execute = host.build, host.protocol, host.execute

    def build():
        value = old_build()
        identity = value["cp10_identity"]
        common = value["common_node_protocol"]["base_protocol"]["identity"]
        host.require(common["root"] == identity["root_sha256"]
                     and common["goal"] == identity["goal_latent_sha256"], "prebound protocol identity mismatch")
        return value

    def protocol():
        value = old_protocol()
        value.update(goal="root0_previously_observed_cp10", target_rebinding_before_protocol=True,
                     paired_identity_only_not_paired_outcome=True)
        return value

    def execute(prepared, data, store, visual, runtime, factory):
        # Source task was bound before freezing; reproduce reference changes on a
        # temporary task, then ensure the actual task and common protocol agree.
        check = deepcopy(prepared)
        observed = cp10.rebind(check, data)
        host.require(observed == prepared["cp10_identity"], "runtime cp10 identity drift")
        task = prepared["source"]["tasks"][0]
        common = prepared["common_node_protocol"]["base_protocol"]["identity"]
        host.require(host.digest(task["goal"]) == observed["goal_latent_sha256"] == common["goal"]
                     and task["binding"]["root_sha256"] == observed["root_sha256"] == common["root"],
                     "runtime protocol target mismatch")
        store.put("cp10_goal_binding.json", {**observed, "cem_report": str(CEM_REPORT),
                                             "cem_report_sha256": CEM_REPORT_SHA,
                                             "goal_state": data["references"][0]["goal"].tolist()})
        result = old_execute(prepared, data, store, visual, runtime, factory)
        result["cp10_goal_binding"] = observed
        return result

    host.build, host.protocol, host.execute = build, protocol, execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if not args.tag.isalnum():
        raise ValueError("alphanumeric tag required")
    configure()
    prefix = host.ROOT / "results" / NAME
    prepared = Path(str(prefix) + "_PREP_" + args.tag + ".json")
    output = Path(str(prefix) + "_RUN_" + args.tag)
    if prepared.exists() or output.exists():
        raise ValueError("exclusive outputs")
    seal = host.prepare(str(prepared))
    print(json.dumps({"prepared": seal}), flush=True)
    with locking.gpu_lock(host.ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        result = host.run(str(prepared), seal["sha256"], str(output), host.AUTH)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "recorded_known_root_not_admitted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
