"""Consume the complete fixed eight-arm panel without endpoint imputation."""
import argparse
import json
from pathlib import Path

from . import p1_tangent_control_20261101_main as control
from . import p1_saved_h4_parfit_20261101 as saved
from . import p1_return_variants_analysis_20261101_main as metrics
from . import p1_goal_cometric_cuda_20261101_main as goal_source


def analyze():
    sources={}
    def read(path):
        sources[str(path)]=saved.sha(path)
        return json.loads(Path(path).read_bytes())
    target=read(goal_source.TARGET)["rows"][0]
    if sources[str(goal_source.TARGET)]!=goal_source.TARGET_SHA:
        raise ValueError("fixed goal")
    rows=[]
    for variant in control.METRICS:
        for arm in control.q.parent.ARMS:
            directory=control.ROOT/"results"/f"{control.NAME}_{variant}_{arm}_RUN_MAIN01"
            path=directory/"report.json"
            report=read(path)
            if len(report["rows"])!=1 or report["rows"][0]["arm"]!=arm:
                raise ValueError("fixed arm")
            row=report["rows"][0]
            prepared=read(Path(report["prepared"]))
            if (sources[report["prepared"]]!=report["prepare_sha256"]
                or report["protocol"]!=prepared["protocol"]
                or report["protocol"]["goal_metric_definition"]!=variant
                or report["protocol"]["metric_sha256"]!=control.METRICS[variant][1]):
                raise ValueError("variant protocol")
            if any(saved.sha(p)!=h for p,h in prepared["sources"].items()):
                raise ValueError("source drift")
            if row["full_horizon"] and row["control_returned"]==20:
                verification=saved.check(directory,sources[str(path)])
            else:
                verification=saved.status_only(directory,sources[str(path)])
            points=[]
            for cp in row["checkpoints"]:
                if cp["goal"]!=target["goal_state"] or cp["goal_rgb_sha256"]!=target["pixel_sha256"]:
                    raise ValueError("task binding")
                if cp["step"]==0 and cp["state"]!=target["root_state"]:
                    raise ValueError("same initial state")
                points.append(dict(step=cp["step"],**metrics.metrics(cp["state"],cp["goal"]),
                                   joint_xy=metrics.qualification.native(cp["goal"],cp["state"])["joint_xy_error"]))
            cp0=next(r for r in points if r["step"]==0)
            cp5=next((r for r in points if r["step"]==5),None)
            fields=("object_xy","object_angle","pusher_xy","joint_xy")
            final=row.get("final_state")
            terminal=None
            if final is not None:
                native=metrics.qualification.native(target["goal_state"],final)
                terminal=dict(step=row["control_returned"],**metrics.metrics(final,target["goal_state"]),
                              joint_xy=native["joint_xy_error"],native_success=native["initial_success"])
                if row["native_success"]!=native["initial_success"]:
                    raise ValueError("native success recomputation")
            cost={}
            for node in report["nodes"][row["episode_id"]]:
                if node["status"]=="sealed":
                    for key,value in node["receipt"]["account"]["charged_upper_bounds"].items():
                        cost[key]=cost.get(key,0)+value
            rows.append(dict(variant=variant,arm=arm,status=row["status"],success=row["native_success"],
                first_error=row["first_error"],terminal=terminal,points=points,
                prefix_delta=metrics.delta({k:cp5[k] for k in fields},{k:cp0[k] for k in fields}) if cp5 else None,
                sealed_node_costs=cost,verification=verification))
    return dict(rows=rows,sources=sources,development_roots=1,P1_admission=False,
                independent_confirmation=False,
                caution="No endpoint padding; non-full20 status checks are not full integrity verification.",
                metric_costs={v:dict(encoder_forwards=21 if v=="geometric" else 13,
                                    input_vjp_rows=0 if v=="geometric" else 192) for v in control.METRICS})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=analyze()
    with args.output.open("x") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps([dict(variant=r["variant"],arm=r["arm"],status=r["status"],
                          success=r["success"],terminal=r["terminal"]) for r in result["rows"]]),flush=True)
