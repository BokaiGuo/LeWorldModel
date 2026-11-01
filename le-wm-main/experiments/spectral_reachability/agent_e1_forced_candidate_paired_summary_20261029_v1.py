"""Independent saved scalar panel for six fixed E1 forced-action branches."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

NAME='agent_e1_forced_candidate_paired_summary_20261029_v1'
PHYSICAL='agent_e1_forced_candidate_physical_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'experiments/spectral_reachability'/(PHYSICAL+'.py')
ARMS=('bank_winner','zero_extra','witness_extra')
SLOTS=tuple((r,a) for r in (0,1) for a in ARMS)
PAIRS=((ARMS[0],ARMS[1]),(ARMS[0],ARMS[2]),(ARMS[1],ARMS[2]))
STEPS=(0,5,10)
CAP=32*1024*1024


def require(value,message):
    if not value:raise ValueError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def read_bound(path,expected):
    p=Path(path)
    require(p.is_absolute() and p.resolve()==p and p.is_file() and not p.is_symlink(),'absolute regular input')
    require(isinstance(expected,str) and len(expected)==64 and all(c in '0123456789abcdef' for c in expected),'external sha256')
    require(p.stat().st_size<=CAP,'input cap')
    raw=p.read_bytes();require(len(raw)<=CAP and sha(raw)==expected,'input SHA')
    def unique(pairs):
        result={}
        for k,v in pairs:
            require(k not in result,'duplicate JSON key');result[k]=v
        return result
    value=json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:require(False,'nonfinite JSON'))
    canonical(value)
    return value


def vector(value):
    require(type(value) is list and len(value)==7 and all(type(x) in (int,float) and math.isfinite(x) for x in value),'finite state7')
    return [float(x) for x in value]


def metrics(state,goal):
    state,goal=vector(state),vector(goal)
    delta=[a-b for a,b in zip(state,goal,strict=True)];squares=[x*x for x in delta]
    xy=math.sqrt(math.fsum(squares[:4]));angle=abs(delta[4]);native_angle=min(angle,2*math.pi-angle)
    return dict(state=state,goal=goal,delta7=delta,axis_squared=squares,
                native7=math.sqrt(math.fsum(squares)),joint_xy=xy,
                agent_xy=math.sqrt(math.fsum(squares[:2])),object_xy=math.sqrt(math.fsum(squares[2:4])),
                agent_velocity=math.sqrt(math.fsum(squares[5:])),raw_angle=delta[4],
                native_angle=native_angle,wrapped_angle=abs(math.atan2(math.sin(delta[4]),math.cos(delta[4]))),
                native_success=xy<20 and native_angle<math.pi/9)


def summarize(prepared,receipt,report):
    require(receipt.get('schema')==PHYSICAL+'.consume' and report.get('schema')==PHYSICAL+'.report'
            and receipt.get('status')==report.get('status') and receipt['status'] in ('six_slot_native_recorded','partial_or_failed')
            and receipt.get('fixed_branch_denominator')==report.get('fixed_branch_denominator')==6
            and receipt.get('scientific_admission') is receipt.get('P1_admission') is False
            and report.get('scientific_admission') is report.get('P1_admission') is False
            and receipt['rows']==report['rows'] and receipt['counts']==report['counts'], 'strict parent/report join')
    rows=report['rows'];plans=prepared['plans']
    require(len(rows)==len(plans)==6 and [(p['root_slot'],p['arm']) for p in plans]==list(SLOTS)
            and [(r['root_slot'],r['arm']) for r in rows]==list(SLOTS),'fixed six slots')
    flattened=[]
    for i,row in enumerate(rows):
        require(row['branch_index']==i and row['selected_plan']==plans[i] and row['selection_digest']==plans[i]['selection_digest'],'plan row binding')
        flattened.extend(dict(branch=i,root_slot=row['root_slot'],arm=row['arm'],checkpoint=cp) for cp in row['checkpoints'])
    require(receipt['checkpoints']==flattened,'receipt checkpoint projection')
    for root in (0,1):
        group=plans[3*root:3*root+3];base=group[0]
        for p in group[1:]:
            for k in ('root_sha256','task_id','task_sha256','task_contract_sha256','blind_sha256','target'):
                require(canonical(p[k])==canonical(base[k]),'same-root same task and target')
    cells=[];index={}
    for i,row in enumerate(rows):
        cp=row['checkpoints'];steps=[x['step'] for x in cp]
        require(steps==sorted(set(steps)) and all(s in STEPS for s in steps),'ordered checkpoints')
        n=row['own_returned'];require(type(n) is int and 0<=n<=10 and all(s<=n for s in steps),'actual time bound')
        if row['status'] in ('horizon_completed','native_success','native_stopped'):
            require(row['initial_exact'] is True and steps==[s for s in STEPS if s<=n],'no early-stop padding')
        available={x['step']:x for x in cp}
        for step in STEPS:
            x=available.get(step);value=None if x is None else metrics(x['state'],row['goal'])
            cell=dict(branch=i,root=row['root_slot'],arm=row['arm'],step=step,status='observed' if x else 'missing',
                      missing_reason=None if x else 'checkpoint_not_returned',branch_status=row['status'],
                      own_returned=n,terminated=row['terminated'],truncated=row['truncated'],
                      checkpoint=None if x is None else dict(png=x['png'],file_sha256=x['file_sha256'],pixel_sha256=x['pixel_sha256']),
                      value=value)
            cells.append(cell);index[(row['root_slot'],row['arm'],step)]=cell
    pairs=[]
    scalar=('native7','joint_xy','agent_xy','object_xy','agent_velocity','raw_angle','native_angle','wrapped_angle')
    for root in (0,1):
        for a,b in PAIRS:
            left,right=plans[3*root+ARMS.index(a)],plans[3*root+ARMS.index(b)]
            same5=canonical(left['commands'][:5])==canonical(right['commands'][:5])
            same10=canonical(left['commands'])==canonical(right['commands'])
            for step in STEPS:
                x=index[(root,a,step)]['value'];y=index[(root,b,step)]['value']
                if x is not None and y is not None:require(canonical(x['goal'])==canonical(y['goal']),'native goal mismatch')
                known=x is not None and y is not None
                pairs.append(dict(root=root,arms=[a,b],step=step,status='observed' if known else 'missing',
                                  missing_arms=[arm for arm,v in ((a,x),(b,y)) if v is None],
                                  same_raw5=same5,same_raw10=same10,
                                  delta_b_minus_a=None if not known else {k:y[k]-x[k] for k in scalar},
                                  native_success_a=None if x is None else x['native_success'],
                                  native_success_b=None if y is None else y['native_success']))
    return dict(cells=cells,pairs=pairs,slot_denominator=6,checkpoint_denominator=18,pair_denominator=18,
                physical_resources=report['counts'],parent_errors=report['errors'],scientific_admission=False,
                same_target=True,equal_total_compute=False,flops=None,known_roots=2,fresh_roots=0,
                boundary='same-time native goal, known roots; no independent fullstrict or causal claim')


def analyze(receipt_path,receipt_sha,report_path,report_sha,physical_source_sha):
    require(sha(SOURCE.read_bytes())==physical_source_sha,'physical source SHA')
    receipt=read_bound(receipt_path,receipt_sha);report=read_bound(report_path,report_sha)
    require(receipt['report_sha256']==report_sha,'receipt binds report')
    prepared=read_bound(report['prepared'],report['prepare_sha256'])
    require(prepared['bindings'].get(str(SOURCE))==physical_source_sha,'prepare binds physical source')
    files=(Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'))
    return dict(schema=NAME+'.analysis',receipt_path=str(receipt_path),receipt_sha256=receipt_sha,
                report_path=str(report_path),report_sha256=report_sha,prepare_sha256=report['prepare_sha256'],
                physical_source_sha256=physical_source_sha,source_hashes={str(p):sha(p.read_bytes()) for p in files},
                authority_boundary='externally pinned full physical receipt; no rerun fullstrict',result=summarize(prepared,receipt,report))


def publish(path,value):
    p=Path(path);require(p.is_absolute() and p.resolve()==p and not p.exists(),'exclusive absolute output')
    raw=canonical(value);require(len(raw)<=CAP,'output cap')
    with p.open('xb') as f:require(f.write(raw)==len(raw),'short write');f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return sha(raw)


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    a=sub.add_parser('analyze')
    for name in ('receipt','receipt-sha','report','report-sha','physical-source-sha','output'):a.add_argument('--'+name,required=True)
    v=sub.add_parser('verify');v.add_argument('--analysis',required=True);v.add_argument('--analysis-sha',required=True)
    x=ap.parse_args()
    if x.command=='analyze':
        value=analyze(x.receipt,x.receipt_sha,x.report,x.report_sha,x.physical_source_sha)
        print(json.dumps(dict(path=x.output,sha256=publish(x.output,value)),sort_keys=True))
    else:
        value=read_bound(x.analysis,x.analysis_sha)
        require(value==analyze(value['receipt_path'],value['receipt_sha256'],value['report_path'],value['report_sha256'],value['physical_source_sha256']),'whole analysis reconstruction')
        print(json.dumps(dict(status='verified',sha256=x.analysis_sha),sort_keys=True))


if __name__=='__main__':main()
