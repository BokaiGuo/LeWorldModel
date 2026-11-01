"""Known cp10 root: zero first packet, then actual-RGB structured carry replanning.

This is an intervention on the archived spectral8 common-budget controller,
not coherent initialization, not a three-arm confirmation.
"""
import argparse
import json
from pathlib import Path

import torch

from . import codex_p1_common_structured_cp10_host_20261101_exclusive as base

NAME = "codex_p1_cp10_zero_carry_20261101_exclusive"


def intervene(result):
    plan = result["selected_raw"]
    if plan.shape != (3, 5, 2) or not bool(torch.isfinite(plan).all()):
        raise ValueError("expected finite H3 plan")
    changed = plan.detach().clone()
    changed[0].zero_()
    updated = dict(result)
    updated["selected_raw"] = changed
    return updated


def configure():
    base.configure()
    host = base.host
    adapter = base.previous.adapter
    old_plan, old_build, old_protocol = adapter.plan, host.build, host.protocol

    def plan(leaf, task, protocol, protocol_sha, boundary, *, device, emit):
        result = old_plan(leaf, task, protocol, protocol_sha, boundary, device=device, emit=emit)
        if boundary == 0:
            result = intervene(result)
            emit("zero_prefix_intervention", {
                "rule": "zero_first_packet_keep_original_selected_tail_for_metered_carry_rescore",
                "original_selected_index": result["selected_index"],
                "raw": result["selected_raw"].detach().cpu().tolist(),
                "incremental_neural_cost": 0,
            })
        return result

    def build():
        value = old_build()
        source = Path(__file__).resolve()
        value["sources"][str(source)] = host.sha(source)
        return value

    def protocol():
        value = old_protocol()
        value.update(prefix_intervention="zero_first_packet_keep_selected_tail",
                     suffix="own_real_RGB_spectral8_carry_rescore_at_5_10_15",
                     meaning="known_root_development_not_coherent_not_independent_confirmation")
        return value

    adapter.plan, host.build, host.protocol, host.NAME = plan, build, protocol, NAME
    return host


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if not args.tag.isalnum():
        raise ValueError("alphanumeric tag required")
    host = configure()
    prefix = host.ROOT / "results" / NAME
    prepared = Path(str(prefix) + "_PREP_" + args.tag + ".json")
    output = Path(str(prefix) + "_RUN_" + args.tag)
    if prepared.exists() or output.exists():
        raise ValueError("exclusive outputs")
    seal = host.prepare(str(prepared))
    print(json.dumps({"prepared": seal}), flush=True)
    with base.locking.gpu_lock(host.ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        result = host.run(str(prepared), seal["sha256"], str(output), host.AUTH)
    print(json.dumps({"status": result["status"], "output": str(output)}), flush=True)
    return 0 if result["status"] == "recorded_known_root_not_admitted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
