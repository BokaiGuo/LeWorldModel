"""Post-hoc domain-censor view over sealed P2 executions, no new execution."""
import argparse
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import types

import numpy as np
from PIL import Image
from experiments.spectral_reachability import codex_p2_feedback_candidate_physical_20261030_v1 as m

NAME="codex_p2_candidate_domain_censor_20261030_v1"
CONTRACT=m.ROOT/"innerdocs/codex_p2_candidate_domain_censor_20261030_v1_contract.md"
CONTRACT_SHA="185dbffcb3ac2778cedebcc8562d9e80da0b49b4425e47fe616fb43791a7da8c"
PINS=[
 ("5a1a34f3eaa461c07be44cb312600bb59369ecc6210a87d75a69bfdcac210330","2fbc4e6490b0d123e53e3e5fef7726a6fe4973c302e79b97e8de4950a4f2f0d7"),
 ("a4dff7ee004c7f45d12a54091f9b5aeb4611e97c8ade88fba74babd5c0946241","ba68f347ceb1330fe8189ff4d760593248a3f94d2246afee3ccd0bb2722ddcb2"),
 ("3ed0ab16982805f3a6467696acc8f3d20c5a3e437be61a6895b3e194d6978ee5","9d2d9c8e4d49b10455b5c604a1654209169b3270c8588c998d6433faf11f8416"),
 ("386b311bb14fd3c352e9f6cba5d28bd9f2f4d459abc8d6d6b169a359086aa31e","359bda80f38be3cf6f4222ea1f357b208ec10d77c285d410e5756f9d5fea8dab")]


def incident(report,events,files):
    need=m.require
    need(report["status"]=="partial_or_failed" and len(report["errors"])==1, "exact failed job")
    row=report["rows"][1]; step=row["own_returned"]
    need(row["policy"]=="fixed_index_1" and row["status"]=="engineering_failed" and
         row["first_error"]=={"type":"Rejected","message":"fresh RGB/canvas guard","stage":"branch"} and
         report["errors"]==[row["first_error"]] and row["own_attempted"]==step, "original failure identity")
    scoped=[e for e in events if e.get("branch",e.get("receipt",{}).get("branch"))==1]
    transitions=[e for e in scoped if e["kind"]=="returned_transition" and e["phase"]=="plan"]
    frames=[e for e in scoped if e["kind"]=="fresh_observation" and e["receipt"]["phase"]=="plan"]
    need(transitions and frames and transitions[-1]["step"]==step, "last returned step")
    t=transitions[-1];f=frames[-1];r=f["receipt"]
    need(r["step"]==step and r["overlay_index"]==0 and r["condition"]=="off" and
         r==row["last_observation"] and r["state"]==t["state"] and r["snapshot"]==t["snapshot"] and
         f["pre"]==f["post"] and f["index"]>t["index"], "saved same-step incident off frame")
    need(t["terminated"] is False and t["truncated"] is False and
         t["requested_bits"]==t["actual_bits"]==m.raw30(row["plan"]["commands"])[step-1].tobytes().hex(),
         "returned action bits and nonterminal")
    need(not any(e["kind"]=="operation_dispatch_intent" and e["receipt"]["resource"]=="public_step" and
                 e["index"]>t["index"] for e in scoped), "no later physical actions")
    state=np.asarray(t["state"],dtype=np.float64)
    need(state.shape==(7,) and np.isfinite(state).all() and np.any((state[:4]<0)|(state[:4]>512)),
         "strict physical canvas boundary violation")
    content=files[r["png"]]
    need(hashlib.sha256(content).hexdigest()==r["file_sha256"], "incident PNG bytes")
    with Image.open(io.BytesIO(content)) as image:
        need(image.format=="PNG" and image.mode=="RGB" and image.size==(224,224), "incident RGB")
        pixels=np.asarray(image).copy()
    count=int(np.count_nonzero(np.all(pixels==[65,105,225],axis=-1)))
    need(count>0 and m.moving.image_sha(pixels)==r["pixel_sha256"], "visible agent and pixel identity")
    try:
        m.moving.guard(pixels,state)
    except Exception as exc:
        need(type(exc).__name__=="Rejected" and str(exc)=="fresh RGB/canvas guard", "exact original guard")
    else:
        raise ValueError("original guard did not reject")
    return dict(step=step,state=t["state"],action_bits=t["actual_bits"],image=r["png"],
                file_sha256=r["file_sha256"],pixel_sha256=r["pixel_sha256"],agent_color_pixels=count,
                original_status=row["status"],classification="domain_censored",
                incident_checkpoint_valid=False,post_censor_metrics=None)


class GuardProxy:
    def __init__(self,base,incidents):
        self.base=base;self.incidents=incidents;self.seen=0
    def __getattr__(self,name):
        return getattr(self.base,name)
    def guard(self,pixels,state):
        for item in self.incidents:
            if np.array_equal(np.asarray(state),np.asarray(item["state"])) and self.base.image_sha(pixels)==item["pixel_sha256"]:
                self.seen+=1
                return item["agent_color_pixels"]
        return self.base.guard(pixels,state)


def consume_censored(directory,report_sha,completion_sha):
    files=m.saved_artifacts(directory)
    m.require(hashlib.sha256(files["report.json"]).hexdigest()==report_sha, "report pin")
    report=m.native.decode(files["report.json"])
    events=[m.native.decode(line) for line in files["journal.jsonl"].splitlines()]
    item=incident(report,events,files)
    # Copy the original validator into an isolated globals namespace. Only the
    # failed final off frame is omitted from the aggregate-trajectory equality;
    # all original checks and the actual journal remain intact.
    original="require(row['trajectory']==selected and row['source_unpaired_on_steps']==unknown,'all selected off frames, no fabricated old on references')"
    changed=("require(row['trajectory']==(selected[:-1] if row['status']=='engineering_failed' and "
             "row['index']==1 else selected) and row['source_unpaired_on_steps']==unknown,"
             "'trajectory plus separately verified incident')")
    text=inspect.getsource(m.validate_saved)
    m.require(text.count(original)==1, "exact sealed validator patch site")
    namespace=dict(m.validate_saved.__globals__)
    exec(compile(text.replace(original,changed),"<isolated_censor_validator>","exec"),namespace)
    proxy=GuardProxy(m.moving,[item])
    namespace["moving"]=proxy
    consume_globals=dict(m.consume.__globals__)
    consume_globals["validate_saved"]=namespace["validate_saved"]
    consume_globals["moving"]=proxy
    consume=types.FunctionType(m.consume.__code__,consume_globals,m.consume.__name__,m.consume.__defaults__,m.consume.__closure__)
    receipt=consume(directory,report_sha,completion_sha)
    m.require(proxy.seen==1 and receipt["status"]=="partial_or_failed", "one incident only, original failed")
    return item,receipt


def build():
    m.require(m.sha(CONTRACT)==CONTRACT_SHA, "fixed contract SHA")
    rows=[];counts={}
    for job,(report_sha,completion_sha) in enumerate(PINS):
        directory=m.ROOT/f"results/codex_p2_feedback_candidate_physical_20261030_v1_RUN_JOB{job}"
        if job<2:
            c,receipt=consume_censored(directory,report_sha,completion_sha)
        else:
            receipt=m.consume(directory,report_sha,completion_sha);c=None
        m.require(receipt["job_index"]==job and len(receipt["rows"])==2, "full denominator")
        for k,v in receipt["counts"].items():
            if k not in counts: counts[k]={"attempted":0,"completed":0}
            for field in ("attempted","completed"):counts[k][field]+=v[field]
        rows.append(dict(job=job,report_sha256=report_sha,completion_sha256=completion_sha,
                         original_status=receipt["status"],censor=c,
                         arms=[dict(policy=r["policy"],original_status=r["status"],
                                    classification=("domain_censored" if c and r["index"]==1 else r["status"]),
                                    returned=r["own_returned"],native_success=(r["native_success"] if r["status"]=="horizon_completed" else None),
                                    final_metrics=(r["final_metrics"] if r["status"]=="horizon_completed" else None),
                                    missing_checkpoints=r["missing_checkpoints"]) for r in receipt["rows"]]))
    return dict(schema=NAME,status="posthoc_censored_saved_review_not_admitted",contract_sha256=CONTRACT_SHA,
                jobs=rows,counts=counts,known_roots=2,base_count=4,slot_count=8,preregistered=False,
                original_failures_preserved=True,old_strict_failure_not_reversed=True,
                adapted_saved_validation=True,independent_neural_execution=False,
                new_neural_calls=0,new_physical_calls=0,equal_total_compute=False,P2_admission=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True);p.add_argument("--verify-sha")
    a=p.parse_args();path=Path(a.output)
    m.require(path.is_absolute() and path.parent.resolve()==path.parent and path.name.startswith(NAME+"_"),"own path")
    value=build();data=m.raw(value);sha=hashlib.sha256(data).hexdigest()
    if a.verify_sha:
        m.require(sha==a.verify_sha and path.read_bytes()==data,"whole saved rebuild")
    else:
        m.require(not os.path.lexists(path),"exclusive output")
        with path.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(status=value["status"],sha256=sha)))


if __name__=="__main__":main()
