"""Eight-arm fixed old-Q/geometric-Q H4 comparison on the same return goal."""
import argparse
import json
from pathlib import Path

from . import p1_reverse_h4_parfit_20261101 as runtime
from . import p1_normalized_exchange_q_20261101_main as search
from . import p1_q_return_shard_20261101_main as q

NAME = "p1_tangent_control_20261101_main"
ROOT = q.ROOT
DOC = ROOT.parent / "docs/D2P_P1_RGB_TANGENT_PROTOCOL_20261101_MAIN.md"
METRICS = {
    "original": (ROOT/"results/p1_goal_cometric_cuda_20261101_main_MAIN01.json",
                 "3ce857733d7ac413aa322e96fa4955f6db94bea53b55add7a5b0527148a21925"),
    "geometric": (ROOT/"results/p1_rgb_tangent_cuda_20261101_main_MAIN01.json",
                  "ae45085cdfe44f756d9344cf9b8e389c7210c09c2d1b5f0f49f13c956cd85dc2"),
}


def setup(variant, arm):
    if variant not in METRICS or arm not in q.parent.ARMS:
        raise ValueError("fixed eight-arm roster")
    path, digest = METRICS[variant]

    def return_setup(which):
        q.METRIC, q.METRIC_SHA = path, digest
        # Retain original return-task input adapter, never call reverse.setup.
        return search.setup(which)

    runtime.reverse.setup = return_setup
    experiment = runtime.setup(arm)
    runner = experiment.runner
    control = q.carry.control
    control.base.CONFIG.update(goal_metric_definition=variant, goal_metric_sha256=digest)
    old_build, old_protocol = runner.build, runner.protocol

    def protocol():
        value = old_protocol()
        value.update(goal_metric_definition=variant, metric_sha256=digest,
                     objective="goal_RGB_geometric_tangent_Q" if variant=="geometric"
                               else "goal_encoder_full_input_cometric_Q",
                     legacy_metric_policy_tag="goal_encoder_cometric_v1_is_shared_Q_runtime_ABI",
                     comparison_variants=list(METRICS), scientific_panel_denominator=8,
                     goal="original_root0_push_and_return",
                     metric_construction_unamortized=dict(
                         encoder_forwards=21 if variant=="geometric" else 13,
                         input_vjp_rows=0 if variant=="geometric" else 192),
                     P1_admission=False, independent_confirmation=False)
        return value

    def build():
        value = old_build()
        value["protocol"] = protocol()
        # Shared H4 runtime schema; task identity comes from the original target adapter.
        for source in (Path(__file__).resolve(), DOC, path):
            value["sources"][str(source)] = runner.sha(source)
        return value

    runner.build, runner.protocol = build, protocol
    runner.NAME = f"{NAME}_{variant}_{arm}"
    return experiment


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant",choices=METRICS,required=True)
    parser.add_argument("--arm",choices=q.parent.ARMS,required=True)
    parser.add_argument("--tag",required=True)
    parser.add_argument("--authorize-host-cuda",action="store_true",required=True)
    args=parser.parse_args()
    if not args.tag.isalnum():
        raise ValueError("simple tag")
    experiment=setup(args.variant,args.arm)
    prefix=ROOT/"results"/f"{NAME}_{args.variant}_{args.arm}"
    prepared=Path(str(prefix)+f"_PREP_{args.tag}.json")
    output=Path(str(prefix)+f"_RUN_{args.tag}")
    if prepared.exists() or output.exists():
        raise ValueError("exclusive output")
    receipt=experiment.runner.prepare(str(prepared))
    print(json.dumps(receipt),flush=True)
    with experiment.extra.gpu.locking.gpu_lock(experiment.extra.gpu.LOCK):
        result=experiment.runner.run(str(prepared),receipt["sha256"],str(output),
                                    "MAIN_AUTHORIZES_COMMON_SUFFIX_RGB_CUDA_AND_PHYSICS")
    print(json.dumps(result),flush=True)
    raise SystemExit(0 if result["status"]=="recorded_known_root_not_admitted" else 1)
