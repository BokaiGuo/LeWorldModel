"""Exact saved-only native7 pair decomposition; no upstream execution."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path

NAME='agent_p1_native7_pair_decomposition_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
MARGIN=ROOT/'results/agent_p1_pair_margin_decomposition_20261028_v1_MAIN01.json'
EVAL=ROOT/'results/agent_p1_model_library_selector_evaluation_20261028_v1_MAIN01.json'
PINS={str(MARGIN):'f723c5eb73ecebd0615a2ded0aa4b741c6f132ca5f6063c673f12d5e98539143',
      str(EVAL):'d8b3bedd38e9f9a56d0156f2452d5a20f363acbdad42ea5e3ee4d91f9626ff67'}
WORDS=('zero','uZ','-uZ','uS','-uS')

def need(x,msg):
    if not x: raise ValueError(msg)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def canonical(x):
    return (json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()

def exact(x):
    need(type(x) in (float,int) and math.isfinite(x),'finite numeric state')
    return Fraction.from_float(float(x))

def pack(x): return {'numerator':str(x.numerator),'denominator':str(x.denominator),'float64':float(x)}
def sign(x): return (x>0)-(x<0)

def slot(row, source):
    need(row['index']==source['index'] and row['actual_return_event_sha256']==source['actual_return_event_sha256'], 'native event identity')
    for k in ('root_slot','condition','role','task_id','anchor_id','job_digest','plan_digest'):
        need(row[k]==source[k], 'slot join '+k)
    s=row['returned_state'];g=row['physical_goal']
    need(type(s) is list and type(g) is list and len(s)==len(g)==7,'state/goal shape7')
    need(g==row['goal_binding']['state'] and g==row['boundary_trace']['20']['goal'] and s==row['boundary_trace']['20']['state'], 'bound original goal and state')
    d=[exact(a)-exact(b) for a,b in zip(s,g)]; q=[x*x for x in d]
    value=math.sqrt(float(sum(q))); m=row['physical_metrics']
    need(math.isclose(value,m['native_distance'],rel_tol=1e-12,abs_tol=1e-12) and math.isclose(-row['returned_reward'],m['native_distance'],rel_tol=1e-12,abs_tol=1e-12), 'native distance/reward join')
    need(type(row['native_success']) is bool and row['native_success'] is m['task_success'], 'original success join')
    return dict(index=row['index'],root_slot=row['root_slot'],condition=row['condition'],role=row['role'],
        word=source['word'],state7=s,goal7=g,native_success=row['native_success'],native_distance=m['native_distance'],
        residual7=[pack(x) for x in d],squared7=[pack(x) for x in q],
        native_squared=pack(sum(q)),agent_xy_squared=pack(sum(q[:2])),object_xy_squared=pack(sum(q[2:4])),
        raw_angle_squared=pack(q[4]),agent_velocity_squared=pack(sum(q[5:])),event_sha256=row['actual_return_event_sha256'])

def unpack(x): return Fraction(int(x['numerator']),int(x['denominator']))

def analyze(margin,evaluation):
    need(margin['fixed_pairs']==80 and margin['fixed_slots']==40 and evaluation['fixed_slots']==40,'fixed denominator')
    need(margin['P1_admission'] is False and evaluation['P1_admission'] is False,'not admitted')
    slots=margin['slots']; rows=evaluation['rows']; pairs=margin['pairs']
    need(len(slots)==len(rows)==40 and len(pairs)==80,'full slot/pair lengths')
    computed=[]
    for i,(r,s) in enumerate(zip(rows,slots)):
        need(r['index']==s['index']==i and s['anchor_index']==i//5 and s['word']==WORDS[i%5],'fixed word roster')
        computed.append(slot(r,s))
    out=[]
    schedule=[(5*a+i,5*a+j) for a in range(8) for i in range(5) for j in range(i+1,5)]
    for p,(i,j) in zip(pairs,schedule):
        need((p['i'],p['j'],p['anchor_index'])==(i,j,i//5),'fixed pair order')
        a,b=computed[i],computed[j]; need(a['goal7']==b['goal7'] and a['root_slot']==b['root_slot'] and a['condition']==b['condition'] and a['role']==b['role'],'same goal/context stratum')
        axes=[unpack(x)-unpack(y) for x,y in zip(a['squared7'],b['squared7'])]
        xy=sum(axes[:4]); angle=axes[4]; velocity=sum(axes[5:]); total=sum(axes)
        need(total==xy+angle+velocity,'exact decomposition')
        metric_delta=p['physical_lower_is_better_deltas']['native_distance']
        need(sign(total)==sign(metric_delta), 'exact squared versus native distance sign')
        c=p['observed_to_physical']['native_distance']; observed=c['sign_a'] if c['available'] else None
        reversal=None if observed is None else bool(observed!=0 and sign(total)!=0 and observed!=sign(total))
        need(reversal is c['reversal'], 'inherited strict ranking classification')
        # Algebraic removal only: not an executable change to native goal or trajectory.
        without=xy+angle
        necessary=None if reversal is not True else not (observed!=0 and sign(without)!=0 and sign(without)!=observed)
        velocity_alone=None if observed is None or observed==0 else bool(sign(velocity)!=0 and sign(velocity)!=observed)
        out.append(dict(i=i,j=j,anchor_index=i//5,root_slot=a['root_slot'],condition=a['condition'],role=a['role'],
            words=[a['word'],b['word']],squared_axis_delta=[pack(x) for x in axes],xy=pack(xy),raw_angle=pack(angle),
            agent_velocity=pack(velocity),native_squared=pack(total),native_distance_delta=metric_delta,
            observed_latent_sign=observed,original_reversal=reversal,velocity_alone_opposes_latent=velocity_alone,
            velocity_necessary_for_existing_reversal=necessary,without_velocity_sign=sign(without),
            original_success=[a['native_success'],b['native_success']]))
    return dict(schema=NAME+'.analysis',inputs=PINS,fixed_slots=40,fixed_pairs=80,known_roots=2,fresh_roots=0,
        slots=computed,pairs=out,summary=dict(reversals=sum(p['original_reversal'] is True for p in out),
        unknown=sum(p['original_reversal'] is None for p in out),
        velocity_necessary=sum(p['velocity_necessary_for_existing_reversal'] is True for p in out),
        velocity_alone_opposes=sum(p['velocity_alone_opposes_latent'] is True for p in out)),
        interpretation='post-hoc exact arithmetic, not causal intervention or new goal',P1_admission=False,scientific_admission=False)

def build():
    values=[]
    for p,expected in PINS.items():
        need(sha(p)==expected,'pinned input SHA'); need(Path(p).stat().st_size<128*1024**2,'metadata cap')
        with open(p,'rb') as f: v=json.load(f)
        need(sha(p)==expected,'stable input SHA'); values.append(v)
    x=analyze(*values)
    paths=[Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    x['source_closure']={str(p):sha(p) for p in paths}
    return x

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('analyze','verify'));p.add_argument('--output',required=True);a=p.parse_args()
    dest=Path(a.output);need(dest.is_absolute() and str(dest.resolve())==str(dest),'canonical path')
    data=canonical(build())
    if a.command=='analyze':
        need(dest.parent==ROOT/'results' and dest.name.startswith(NAME+'_'),'new owned result')
        with dest.open('xb') as f: f.write(data);f.flush();os.fsync(f.fileno())
    else: need(dest.is_file() and not dest.is_symlink() and dest.read_bytes()==data,'whole analysis verification')
    print(hashlib.sha256(data).hexdigest())

if __name__=='__main__':main()
