"""CPU-only join of the actual bank-bound plans to historical physical inputs.

No physical factory is invoked. The output is a retrospective engineering
binding, not a physical authorization or preregistration.
"""
import argparse
import json
import os
from pathlib import Path

from experiments.spectral_reachability import codex_p2_knownroot_host_cuda_20261030_v1 as host
from experiments.spectral_reachability import p3_horizon_residual_physical_20261027_v4 as physical_source

NAME = "codex_p2_knownroot_physical_join_20261030_v1"
REFERENCE_SHA = "c5eee78ded753d1c2b955335b97e218d420671e1d2d35636aae8303a8811622b"
need = host.need


def join(*, run_dir, run_sha, prepared_dir, prepared_sha, reference_dir,
         source_validator=None, consumer=None):
    """Strict planner consume plus exact context, goal-vector and RGB join."""
    run_dir = Path(run_dir).resolve()
    need(host.frontier.file_sha(run_dir / "run_report.json") == run_sha, "run SHA")
    report = host.frontier.read(run_dir / "run_report.json")
    manifest, protocol, prepared = host.validate_prepare(prepared_dir, prepared_sha)
    need(report["status"] == "exploratory_complete" and report["prepare"] == manifest and
         report["physical_authorized"] is False and report["P1_admission"] is False and
         report["P2_admission"] is False and len(report["roots"]) == 4, "host run")
    refs = (source_validator or physical_source.validate_prepared)(reference_dir, REFERENCE_SHA)
    need(refs["root_seed"] == 202753010 and refs["root_oversample_factor"] == 4,
         "physical root collection recipe")
    packets = {p["root_sha256"]: p for p in refs["packets"]}
    need(set(packets) == set(host.frontier.ROOTS) == set(refs["references"]), "physical root census")
    compiled = {x["root_sha256"]: x["compiled"] for x in
                host.frontier.read(protocol["context"]["compilation"])["rows"]}
    results = []
    consume = consumer or host.saved.consume
    for index, case in enumerate(protocol["context"]["cases"]):
        root = case["root_sha256"]
        receipt = report["roots"][index]
        need(receipt["root_index"] == case["root_index"] == index and
             receipt["root_sha256"] == root and
             Path(receipt["directory"]).resolve() == run_dir / ("root_" + str(index)),
             "root receipt")
        c, packet, ref = compiled[root], packets[root], refs["references"][root]
        need(packet["root_context"] == case["root"]["context"] and
             packet["goal"] == c["goal_vectors"][0] and
             packet["projection"] == c["scoring"]["metric"]["projection"], "context/goal/Q binding")
        need(ref["root_sha256"] == root and
             ref["history_rgb_sha256"] == case["root"]["root_rgb_sha256"] and
             ref["goal_rgb_sha256"] == case["root"]["goal_rgb_sha256"], "root/goal RGB binding")
        domain = case["action_domain"]
        need(domain["name"] == "raw" and domain["low"] == [-1., -1.] and
             domain["high"] == [1., 1.] and domain["action_block"] == 5, "action domain")
        result = consume(directory=run_dir / ("root_" + str(index)),
                         expected_report_sha256=receipt["report_sha256"], ready=case,
                         compiled=c, protocol=protocol, prepared=prepared,
                         source_bindings=manifest["source_bindings"])
        plans = result["plans"]
        need([p["arm"] for p in plans] == list(host.frontier.ARMS), "two arm census")
        for p in plans:
            need(p["root_sha256"] == root and p["action_domain"] == domain and
                 len(p["commands"]) == 10 and p["selection_uses_physical_outcome"] is False,
                 "selected raw10 binding")
        ordered = plans if index % 2 == 0 else plans[::-1]
        for p in ordered:
            results.append(dict(root_index=index, root_sha256=root, arm=p["arm"],
                                commands=p["commands"], selection_sha256=p["selection_sha256"],
                                history_rgb_sha256=ref["history_rgb_sha256"],
                                goal_rgb_sha256=ref["goal_rgb_sha256"],
                                action_domain=domain, root_context_sha256=host.frontier.digest(packet["root_context"]),
                                goal_vector_sha256=host.frontier.digest(packet["goal"]),
                                host_report_sha256=receipt["report_sha256"]))
    need(len(results) == 8, "eight fixed plans")
    return dict(schema=NAME, status="joined_not_physical_ready", plans=results,
                run_report_sha256=run_sha, host_prepare_sha256=prepared_sha,
                physical_reference_sha256=REFERENCE_SHA, root_seed=refs["root_seed"],
                root_oversample_factor=4, warmup_public_steps_per_branch=15,
                renderer="swm/PushT-v1 rgb_array; RoyalBlue guard; runtime verification required",
                physical_factory_bound=False, physical_authorized=False,
                physical_steps=0, new_neural_calls=0, P1_admission=False, P2_admission=False,
                caveat="retrospective known-root join; no execution or scientific admission")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("run-dir", "run-sha", "prepared-dir", "prepared-sha", "reference-dir", "output"):
        parser.add_argument("--" + option, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    need(output.is_absolute() and output.name.startswith(NAME + "_") and
         not os.path.lexists(output), "exclusive output")
    result = join(run_dir=args.run_dir, run_sha=args.run_sha, prepared_dir=args.prepared_dir,
                  prepared_sha=args.prepared_sha, reference_dir=args.reference_dir)
    with output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"output": str(output), "sha256": host.frontier.file_sha(output),
                      "status": result["status"]}))


if __name__ == "__main__":
    main()
