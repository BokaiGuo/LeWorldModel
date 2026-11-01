"""Same-action old/new metric decomposition; diagnostic only, no selection."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import p1_candidate_intervention_analysis_20261101_main as physics
from . import p1_tangent_control_20261101_main as control
from . import p1_reverse_target_cuda_20261101_main as helpers

ROOT=control.ROOT
DOC=ROOT.parent/"docs/D2P_P1_TANGENT_FIXED_BANK_PROTOCOL_20261101_MAIN.md"
DIRECTORY=ROOT/"results/p1_candidate_intervention_20261101_main_run_MAIN01"
RGB=ROOT/"results/p1_candidate_rgb_cuda_20261101_main_MAIN01.json"


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    base=helpers.encoder.base
    output=Path(output)
    if not output.is_absolute() or output.parent!=ROOT/"results" or output.exists():
        raise ValueError("exclusive output")
    checked=physics.analyze(DIRECTORY)
    rgb=json.loads(RGB.read_bytes())
    if any(base.sha(p)!=h for p,h in rgb["sources"].items()):
        raise ValueError("RGB bindings")
    plan=json.loads((DIRECTORY/"prepared.json").read_bytes())
    prefix=ROOT/"results/p1_equal_candidate_q_20261101_main_structured"
    prepared=Path(str(prefix)+"_PREP_MAIN01.json")
    report_path=Path(str(prefix)+"_RUN_MAIN01")/"report.json"
    report=json.loads(report_path.read_bytes())
    if base.sha(prepared)!=report["prepare_sha256"]:
        raise ValueError("source prepared")
    task=json.loads(prepared.read_bytes())["source"]["tasks"][0]
    own=[b for b in plan["schedule"] if b["suffix_role"]=="own"]
    if [b["candidate"] for b in own]!=list(range(8)):
        raise ValueError("fixed bank")
    metric={}
    sources={**rgb["sources"],**plan["bindings"],**checked["sources"],**base.bindings()}
    for variant,(path,digest) in control.METRICS.items():
        if base.sha(path)!=digest:
            raise ValueError("metric binding")
        metric[variant]=json.loads(path.read_bytes())
        sources[str(path)]=digest
        if base.spec.digest(metric[variant]["goal_latent"]) != report["cp10_goal_binding"]["goal_latent_sha256"]:
            raise ValueError("runtime-bound goal identity")
    # The prepared source is rebound by the original task adapter at execution.
    # Never silently use its historical pre-rebind target.
    task["goal"] = metric["original"]["goal_latent"]
    for p in (prepared,report_path,RGB,DOC,Path(__file__).resolve()):
        sources[str(p)]=base.sha(p)
    with base.locking.gpu_lock(base.LOCK):
        host=require_cuda_allocation("cuda:0")
        device=torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(str(base.MODEL),device=device).eval().requires_grad_(False)
        before=base.model_hash(model)
        if before!=base.MODEL_HASH:
            raise ValueError("checkpoint")
        tensor=lambda x:torch.as_tensor(x,device=device,dtype=torch.float32)
        mean,std=tensor(task["action_mean"]),tensor(task["action_std"])
        goal=np.asarray(task["goal"],np.float32)
        def score(vector,variant):
            residual=(np.asarray(vector,np.float32)-goal).astype(float)
            matrix=np.asarray(metric[variant]["sqrt_matrix"],float)
            return float(np.mean((residual@matrix.T)**2))
        rows=[]
        with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH),torch.autocast("cuda",enabled=False):
            for branch in own:
                z=tensor(task["root_context"]["latent_history"])[None]
                history=tensor(task["root_context"]["action_history"])[None]
                raw=tensor(branch["commands"]).reshape(3,5,2)
                for packet in raw:
                    context=torch.cat((history,((packet-mean)/std).reshape(1,1,10)),dim=1)
                    predicted=model.predict(z,model.action_encoder(context))
                    z=torch.cat((z,predicted[:,-1:]),dim=1)[:,-3:]
                    history=context[:,-2:]
                vector=z[0,-1].cpu().numpy()
                i=branch["candidate"]
                old_score=score(vector,"original")
                if not np.isclose(old_score,checked["predicted_scores"][i],rtol=1e-5,atol=1e-7):
                    raise ValueError("same-action original-score replay")
                physical=next(r for r in checked["rows"] if r["candidate"]==i and r["suffix"]=="own")
                actual=None
                if physical["endpoint15"] is not None:
                    frame=next(f for f in rgb["frames"] if f["candidate"]==i and f["suffix"]=="own" and f["step"]==15)
                    actual=rgb["encoded"][frame["pixel_sha256"]]["latent"]
                rows.append(dict(candidate=i,status=physical["status"],predicted=vector.tolist(),
                    actual=actual,physical_endpoint=physical["endpoint15"],
                    predicted_scores={v:score(vector,v) for v in metric},
                    actual_scores={v:score(actual,v) for v in metric} if actual is not None else None))
        torch.cuda.synchronize()
        if base.model_hash(model)!=before or any(base.sha(p)!=h for p,h in sources.items()):
            raise ValueError("model/source drift")
        pairs=[]
        for a in rows:
            for b in rows[a["candidate"]+1:]:
                if a["actual"] is None or b["actual"] is None:
                    continue
                variants={}
                for v in metric:
                    pm=a["predicted_scores"][v]-b["predicted_scores"][v]
                    am=a["actual_scores"][v]-b["actual_scores"][v]
                    tm=a["physical_endpoint"]["joint_xy"]-b["physical_endpoint"]["joint_xy"]
                    variants[v]=dict(predicted_margin=pm,actual_margin=am,task_margin=tm,
                                     model_rgb_reversal=pm*am<0,rgb_task_reversal=am*tm<0)
                pairs.append(dict(candidates=[a["candidate"],b["candidate"]],variants=variants))
        digest=base.write(output,dict(rows=rows,pairs=pairs,sources=sources,host=host,
             model_before=before,model_after=before,predict_calls=24,action_encode_calls=24,
             encoder_calls=0,diagnostic_only=True,physical_outcomes_used_for_deployment=False,
             P1_admission=False,independent_confirmation=False))
    print(json.dumps(dict(sha256=digest,rows=8,pairs=len(pairs))),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--authorize-host-cuda",action="store_true",required=True)
    args=p.parse_args()
    run(args.output)
