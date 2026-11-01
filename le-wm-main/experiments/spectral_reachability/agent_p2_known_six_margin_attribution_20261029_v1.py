"""Saved scalar-level algebra, not causal attribution or a new selector."""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path

NAME='agent_p2_known_six_margin_attribution_20261029_v1'
SOURCE='agent_p2_six_target_physical_scalar_summary_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
CP=(0,5,10,15)
ARMS=('prior_FIRST','program_first_postcondition','arbitrary')
AXES=('joint_xy_squared','raw_angle_squared','agent_velocity_squared')
FLAGS=('P1_admission','P2_admission','P2_mechanism','scientific_admission','freshness',
       'equal_FLOPs','equal_total_compute','independent_confirmation','physical_release')
CAP=16*1024**2


def need(ok,label):
    if not ok:raise ValueError(label)


def encode(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sign(x):return 'positive' if x>0 else 'negative' if x<0 else 'tie'
def exact(x):
    need(type(x) in (int,float) and math.isfinite(x),'finite scalar, no bool')
    return Fraction(x)
def rational(x):return dict(numerator=str(x.numerator),denominator=str(x.denominator))


def absolute(path):
    p=Path(path);need(p.is_absolute() and str(p.resolve())==str(p),'canonical absolute path');return p


def read(path,expected):
    p=absolute(path)
    need(type(expected) is str and len(expected)==64 and all(c in '0123456789abcdef' for c in expected),'SHA256')
    need(p.is_file() and not p.is_symlink() and p.stat().st_size<=CAP,'bounded regular saved summary')
    b=p.read_bytes();need(len(b)<=CAP and hashlib.sha256(b).hexdigest()==expected,'input exact SHA')
    def pairs(items):
        out={}
        for k,v in items:need(k not in out,'duplicate JSON key');out[k]=v
        return out
    def bad(_):raise ValueError('nonfinite JSON')
    x=json.loads(b,object_pairs_hook=pairs,parse_constant=bad);encode(x);return x


def checked_metrics(m):
    if m is None:return None
    for k in ('native7_squared','native7_distance','joint_xy_error',*AXES):
        need(exact(m[k])>=0,'nonnegative squared/distance scalar')
    total=exact(m['native7_squared']);axes=sum((exact(m[k]) for k in AXES),Fraction())
    # Scalar-level residual of two rounded sums, not a new physical tolerance.
    eps=Fraction(1,2**52);bound=32*eps/(1-32*eps)*(abs(total)+sum(abs(exact(m[k])) for k in AXES))
    need(abs(total-axes)<=bound,'saved scalar squared identity rounding only')
    return dict(axis_squares={k:m[k] for k in AXES},native7_squared=m['native7_squared'],
        exact_scalar_sum_residual=rational(total-axes),native_success=m['native_success'])


def attribute(source):
    need(source['schema']==SOURCE+'.summary' and all(source[k]==0 for k in ('new_neural_calls','new_physics_calls','new_fullstrict_calls')),
         'original saved scalar summary schema')
    s=source['summary'];need(s['denominator']==dict(slots=6,checkpoints=24,pairs=24) and len(s['rows'])==len(s['pairs'])==24
        and s['known_roots']==2 and s['fresh_roots']==0 and all(s[k] is False for k in FLAGS),'fixed known-six complete denominator')
    rows=[];lookup={}
    for i,row in enumerate(s['rows']):
        slot,step=i//4,CP[i%4]
        need((row['slot'],row['root_slot'],row['arm'],row['step'])==(slot,slot//3,ARMS[slot%3],step),'ordered 24 checkpoints')
        m=row['metrics'];checked=checked_metrics(m)
        need((row['missing_reason'] is not None)==(m is None),'null status preserved')
        lookup[slot,step]=m
        rows.append(dict(slot=slot,root_slot=slot//3,arm=ARMS[slot%3],step=step,metrics=checked,
                         missing_reason=row['missing_reason']))
    pairs=[];reversals=[];census={}
    order=[(root,step,3*root+l,3*root+r) for root in (0,1) for l,r in ((0,1),(0,2),(1,2)) for step in CP]
    for original,(root,step,left,right) in zip(s['pairs'],order):
        need((original['root_slot'],original['step'],original['left_slot'],original['right_slot'])==(root,step,left,right),'ordered 24 same-time pairs')
        a,b=lookup[left,step],lookup[right,step];known=a is not None and b is not None
        missing=[i for i,m in ((left,a),(right,b)) if m is None]
        need(original['comparable'] is known and original['missing_slots']==missing,'all pair missing preserved')
        entry=dict(root_slot=root,step=step,left_slot=left,right_slot=right,left_arm=ARMS[left%3],right_arm=ARMS[right%3],
                   comparable=known,missing_slots=missing,attribution=None)
        if known:
            keys=('native7_squared','native7_distance','joint_xy_error',*AXES)
            delta={k:b[k]-a[k] for k in keys}
            need(original['delta_right_minus_left'] is not None and all(delta[k]==original['delta_right_minus_left'][k] for k in keys),
                 'independent pair subtraction equals saved full table')
            ex={k:exact(b[k])-exact(a[k]) for k in keys};total=ex['native7_squared'];axis_sum=sum((ex[k] for k in AXES),Fraction())
            residual=total-axis_sum;eps=Fraction(1,2**52)
            bound=32*eps/(1-32*eps)*sum((abs(exact(x[k])) for x in (a,b) for k in ('native7_squared',*AXES)),Fraction())
            need(abs(residual)<=bound,'paired scalar identity rounding only')
            pos=ex[AXES[0]];other=ex[AXES[1]]+ex[AXES[2]]
            opposite=pos*total<0
            observed_opposite=ex['joint_xy_error']*ex['native7_distance']<0
            relation={k:('tie' if ex[k]==0 else 'total_tie' if total==0 else 'aligned' if ex[k]*total>0 else 'opposed') for k in AXES}
            cancellation=any(x>0 for x in (ex[k] for k in AXES)) and any(x<0 for x in (ex[k] for k in AXES))
            entry['attribution']=dict(float64_margin=delta,exact_saved_scalar_margins={k:rational(v) for k,v in ex.items()},
                signs={k:sign(v) for k,v in ex.items()},axis_relation_to_total=relation,axis_cancellation=cancellation,
                exact_axis_sum=rational(axis_sum),exact_total_minus_axis_sum=rational(residual),rounding_bound=float(bound),
                position_vs_total_squared_opposite=opposite,position_vs_total_distance_opposite=observed_opposite,
                other_axes_sum=rational(other),
                # Includes residual explicitly: no attribution of rounding to physics.
                other_axes_plus_rounding=rational(other+residual),
                explanation='T=P+A+V+rounding; when P*T<0, A+V+rounding has T sign and magnitude greater than abs(P)',
                native_success_left=a['native_success'],native_success_right=b['native_success'])
            if opposite or observed_opposite:reversals.append({k:entry[k] for k in ('root_slot','step','left_slot','right_slot','left_arm','right_arm')})
            category=sign(total)
        else:
            need(original['delta_right_minus_left'] is None,'missing pair has no fabricated margin');category='null'
        census[category]=census.get(category,0)+1;pairs.append(entry)
    need(s['comparable_pairs']==sum(x['comparable'] for x in pairs),'complete comparable count')
    return dict(rows=rows,pairs=pairs,position_total_opposition_index=reversals,
        total_margin_census={k:census.get(k,0) for k in ('negative','tie','positive','null')},
        axis_margin_census={axis:{label:sum((label=='null' if x['attribution'] is None else
            x['attribution']['signs'][axis]==label) for x in pairs) for label in ('negative','tie','positive','null')} for axis in AXES},
        denominator=dict(slots=6,checkpoints=24,pairs=24),known_roots=2,fresh_roots=0,
        convention='right minus left; exact means exact differences of saved float64 scalars, not exact native dynamics',
        interpretation='descriptive arithmetic only; no causal control, no unique root cause, no selector or deployment repair',
        **{k:False for k in FLAGS})


def analyze(summary,summary_sha):
    p=absolute(summary);x=read(str(p),summary_sha)
    result=dict(schema=NAME+'.analysis',arguments=dict(summary=str(p),summary_sha=summary_sha),analysis=attribute(x),
        imported_source_bindings=x['bindings'],new_neural_calls=0,new_physics_calls=0,new_fullstrict_calls=0,
        bindings={str(p):summary_sha})
    for f in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py')):result['bindings'][str(f)]=sha(f)
    need(sha(p)==summary_sha,'source unchanged');return result


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('analyze','verify'))
    for k in ('summary','summary-sha','output','analysis','analysis-sha'):ap.add_argument('--'+k)
    a=ap.parse_args(argv)
    if a.command=='verify':
        saved=read(a.analysis,a.analysis_sha);need(saved['schema']==NAME+'.analysis','own schema')
        need(encode(saved)==encode(analyze(**saved['arguments'])),'whole analysis recomputation');out=dict(verified=True,sha256=a.analysis_sha)
    else:
        x=analyze(a.summary,a.summary_sha);p=absolute(a.output);b=encode(x)
        need(p.name.startswith(NAME+'_') and not p.exists() and not p.is_symlink() and len(b)<=CAP,'exclusive bounded new output')
        with p.open('xb') as f:need(f.write(b)==len(b),'complete write');f.flush();os.fsync(f.fileno())
        fd=os.open(p.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        out=dict(output=str(p),sha256=sha(p))
    print(encode(out).decode(),end='')


if __name__=='__main__':main()
