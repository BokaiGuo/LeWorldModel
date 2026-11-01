"""Same-time descriptive pairs after the post-hoc saved domain-censor review."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from experiments.spectral_reachability import codex_p2_candidate_domain_censor_20261030_v1 as review

NAME="codex_p2_candidate_censored_pairs_20261030_v1"
ANALYSIS=review.m.ROOT/"results/codex_p2_candidate_domain_censor_20261030_v1_ANALYSIS01.json"
SHA="99e6da71ab93adeb8604f6b5cfdcce8b78bfabdccb70ba8ed57b50b57ec1f605"
CP=(0,5,10,15,20,25,30)


def pair(rows):
    need=review.m.require
    need(len(rows)==2 and [r["policy"] for r in rows]==["selected","fixed_index_1"],"fixed pair")
    result=[]
    for cp in CP:
        vals=[]
        for r in rows:
            found=[x for x in r["checkpoints"] if x["step"]==cp]
            need(len(found)<=1, "unique checkpoint")
            if not found:
                vals.append(None);continue
            metric=review.m.native.metrics(r["goal"],found[0]["state"])["native"]
            vals.append(dict(object_xy=metric["object_position_error"],
                             wrapped_angle=metric["wrapped_object_angle_error"],
                             native_success=metric["native_success"]))
        delta=None if None in vals else {k:vals[1][k]-vals[0][k] for k in ("object_xy","wrapped_angle")}
        result.append(dict(step=cp,selected=vals[0],fixed_index_1=vals[1],
                           fixed_minus_selected=delta))
    return result


def build():
    m=review.m
    m.require(m.sha(ANALYSIS)==SHA and m.native.decode(ANALYSIS.read_bytes())==review.build(),"review rebuilt")
    jobs=[]
    for job,(h,c) in enumerate(review.PINS):
        path=m.ROOT/f"results/codex_p2_feedback_candidate_physical_20261030_v1_RUN_JOB{job}"
        receipt=review.consume_censored(path,h,c)[1] if job<2 else m.consume(path,h,c)
        jobs.append(dict(job=job,root=receipt["rows"][0]["root_slot"],arm=receipt["rows"][0]["arm"],
                         checkpoints=pair(receipt["rows"])))
    return dict(schema=NAME,status="descriptive_known_root_not_admitted",review_sha256=SHA,jobs=jobs,
                base_count=4,known_roots=2,preregistered=False,independent_confirmation=False,
                equal_total_compute=False,P2_admission=False,new_neural_calls=0,new_physical_calls=0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True)
    a=p.parse_args();path=Path(a.output)
    review.m.require(path.is_absolute() and path.parent.resolve()==path.parent and path.name.startswith(NAME+"_") and
                     not os.path.lexists(path),"exclusive output")
    data=review.m.raw(build())
    with path.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(status="descriptive_known_root_not_admitted",sha256=hashlib.sha256(data).hexdigest())))


if __name__=="__main__":main()
