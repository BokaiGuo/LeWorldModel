"""Full30 fixed saved scalar margins: exact arithmetic, not causal inference."""
from __future__ import annotations
import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path

NAME='agent_p2_full30_margin_attribution_20261029_v1'
SOURCE='main_p2_full30_same_time_summary_20261029_v1'
SOURCE_SHA='d9b97b5c351bcee90f1d2bde5598d2d2c6469dd7bcceb09b9a66117cad00fa16'
ROOT=Path(__file__).resolve().parents[2]
CP=(0,5,10,15,20,25,30)
ARMS=('two_program_transfer','joint_two_round')
AXES=('joint_xy_squared','raw_angle_squared','velocity_squared')
TOTAL='native7_squared'
PARTS=('agent_xy_squared','object_xy_squared')
FLAGS=('P1_admission','P2_admission','P2_mechanism','scientific_admission','freshness',
       'equal_FLOPs','equal_total_compute','physical_release')
CAP=16*1024**2


def need(ok,label):
    if not ok:raise ValueError(label)


def encode(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sign(x):return 'positive' if x>0 else 'negative' if x<0 else 'tie'
def F(x):
    need(type(x) in (int,float) and math.isfinite(x),'finite scalar, no bool');return Fraction(x)
def rat(x):return dict(numerator=str(x.numerator),denominator=str(x.denominator))


def absolute(path):
    p=Path(path);need(p.is_absolute() and str(p.resolve())==str(p) and not p.is_symlink(),'canonical absolute non-symlink path');return p


def read(path,h):
    p=absolute(path)
    need(type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h),'actual SHA256')
    need(p.is_file() and p.stat().st_size<=CAP,'bounded regular JSON');b=p.read_bytes()
    need(len(b)<=CAP and hashlib.sha256(b).hexdigest()==h,'actual external SHA256')
    def pairs(items):
        out={}
        for k,v in items:need(k not in out,'duplicate JSON key');out[k]=v
        return out
    def bad(_):raise ValueError('nonfinite JSON')
    x=json.loads(b,object_pairs_hook=pairs,parse_constant=bad);encode(x);return x


def reduction(total,parts):
    residual=F(total)-sum((F(v) for v in parts),Fraction())
    eps=Fraction(1,2**52);bound=32*eps/(1-32*eps)*(abs(F(total))+sum(abs(F(v)) for v in parts))
    need(abs(residual)<=bound,'saved scalar reduction rounding bound, not scientific tolerance')
    return dict(exact_residual=rat(residual),rounding_bound=float(bound))


def metric(m):
    if m is None:return None
    for k in (TOTAL,*AXES,*PARTS,'native7_distance','joint_xy_error','object_xy_error'):
        need(F(m[k])>=0,'nonnegative scalar')
    need(type(m['native_success']) is bool,'literal native success')
    return dict(values=deepcopy(m),native7_reduction=reduction(m[TOTAL],[m[k] for k in AXES]),
                position_reduction=reduction(m[AXES[0]],[m[k] for k in PARTS]))


def pair_math(a,b,saved):
    keys=tuple(k for k in a if k!='native_success')
    need(set(keys)==set(saved['delta']) and set(a)==set(b),'all scalar keys retained')
    delta={k:b[k]-a[k] for k in keys};ex={k:F(b[k])-F(a[k]) for k in keys}
    need(encode(delta)==encode(saved['delta']) and saved['native_success_delta']==int(b['native_success'])-int(a['native_success']),
         'independent subtraction and native success delta')
    need(encode(saved['squared_components'])==encode({k:delta[k] for k in AXES})
        and saved['decomposition_roundoff']==delta[TOTAL]-math.fsum(delta[k] for k in AXES),'original rounded components and residual')
    subtraction={k:F(delta[k])-ex[k] for k in keys}
    reduction_error=ex[TOTAL]-sum((ex[k] for k in AXES),Fraction())
    displayed_residual=F(delta[TOTAL])-sum((F(delta[k]) for k in AXES),Fraction())
    reconstructed=reduction_error+subtraction[TOTAL]-sum((subtraction[k] for k in AXES),Fraction())
    need(displayed_residual==reconstructed,'exact separation of reduction and subtraction roundoff')
    position_reduction=ex[AXES[0]]-sum((ex[k] for k in PARTS),Fraction())
    total=ex[TOTAL];pos=ex[AXES[0]];other=ex[AXES[1]]+ex[AXES[2]]
    return dict(float64_margin=delta,exact_saved_scalar_margins={k:rat(v) for k,v in ex.items()},
        signs={k:sign(v) for k,v in ex.items()},axis_relation_to_total={k:('tie' if ex[k]==0 else 'total_tie' if total==0 else
            'aligned' if ex[k]*total>0 else 'opposed') for k in AXES},
        axis_cancellation=any(ex[k]>0 for k in AXES) and any(ex[k]<0 for k in AXES),
        exact_margin_reduction_residual=rat(reduction_error),exact_position_split_residual=rat(position_reduction),
        exact_subtraction_roundoff={k:rat(v) for k,v in subtraction.items()},
        exact_float_margin_minus_axis_sum=rat(displayed_residual),exact_rounding_reconstruction=rat(reconstructed),
        position_vs_total_squared_opposite=pos*total<0,
        position_vs_total_distance_opposite=ex['joint_xy_error']*ex['native7_distance']<0,
        other_axes_sum=rat(other),other_axes_plus_reduction_rounding=rat(other+reduction_error),
        explanation='T=P+A+V+reduction_rounding; displayed float margins add subtraction roundoff separately',
        native_success_left=a['native_success'],native_success_right=b['native_success'])


def attribute(x):
    need(x['schema']==SOURCE+'.summary' and all(type(x[k]) is int and x[k]==0 for k in
        ('new_neural_calls','new_physics_calls','new_fullstrict_calls')),'original saved full30 scalar schema')
    s=x['summary'];den=dict(branches=4,known_roots=2,fresh_roots=0,checkpoint_positions=28,pairs=14)
    need(s['denominator']==den and len(s['panel'])==28 and len(s['pairs'])==14
        and all(s[k] is False for k in FLAGS) and type(s['all_four_full30']) is bool,'fixed denominator and unchanged admission')
    rows=[];lookup={}
    for i,row in enumerate(s['panel']):
        slot,step=i//7,CP[i%7]
        need((row['slot'],row['root'],row['arm'],row['control_step'],row['absolute_step'])==
            (slot,slot//2,ARMS[slot%2],step,10+step),'fixed same-time row order')
        m=row['metrics'];v=metric(m)
        need((row['missing_reason'] is not None)==(m is None) and (row['checkpoint'] is None)==(m is None),'null checkpoint kept')
        if s['all_four_full30']:need(m is not None,'full30 has no missing cp')
        lookup[slot,step]=m
        rows.append(dict(slot=slot,root=slot//2,arm=ARMS[slot%2],control_step=step,absolute_step=10+step,
            metrics=v,missing_reason=row['missing_reason']))
    pairs=[];index=[]
    for i,p in enumerate(s['pairs']):
        root,step=i//7,CP[i%7];a,b=lookup[2*root,step],lookup[2*root+1,step]
        need((p['root'],p['control_step'],p['direction'])==(root,step,'joint_two_round minus two_program_transfer'), 'fixed same-time pair identity/direction')
        known=a is not None and b is not None;need(p['comparable'] is known,'comparable is complete not selected')
        entry=dict(root=root,control_step=step,left_slot=2*root,right_slot=2*root+1,direction=p['direction'],comparable=known,
            missing_slots=[slot for slot,m in ((2*root,a),(2*root+1,b)) if m is None],attribution=None)
        if known:
            entry['attribution']=pair_math(a,b,p)
            if entry['attribution']['position_vs_total_squared_opposite'] or entry['attribution']['position_vs_total_distance_opposite']:
                index.append(dict(root=root,control_step=step,left_slot=2*root,right_slot=2*root+1))
        else:need(all(p[k] is None for k in ('delta','squared_components','decomposition_roundoff','native_success_delta')),'null has no fabricated margin')
        pairs.append(entry)
    def census(key):
        return {label:sum((label=='null' if p['attribution'] is None else p['attribution']['signs'][key]==label) for p in pairs)
                for label in ('negative','tie','positive','null')}
    return dict(panel=rows,pairs=pairs,denominator=den,position_total_opposition_index=index,
        margin_census={k:census(k) for k in (TOTAL,*AXES)},original_counts=deepcopy(s['counts']),
        all_four_full30=s['all_four_full30'],
        convention='joint_two_round minus two_program_transfer; exact refers only to saved float64 scalars, not dynamics',
        interpretation='arithmetic attribution only, not causal control/unique cause/selector/deployment/equal resource proof',
        **{k:False for k in FLAGS})


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def analyze(summary,summary_sha):
    need(summary_sha==SOURCE_SHA,'only specified frozen full30 summary')
    p=absolute(summary);x=read(str(p),summary_sha)
    result=dict(schema=NAME+'.analysis',arguments=dict(summary=str(p),summary_sha=summary_sha),analysis=attribute(x),
        imported_source_arguments=deepcopy(x['arguments']),imported_source_code_sha256=deepcopy(x['own_sha256']),
        new_neural_calls=0,new_physics_calls=0,new_fullstrict_calls=0,bindings={str(p):summary_sha})
    for f in own_files():result['bindings'][str(f)]=sha(f)
    need(sha(p)==summary_sha,'source unchanged');return result


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('analyze','verify'))
    for k in ('summary','summary-sha','output','analysis','analysis-sha'):ap.add_argument('--'+k)
    a=ap.parse_args(argv)
    if a.command=='verify':
        saved=read(a.analysis,a.analysis_sha);need(saved['schema']==NAME+'.analysis','own schema')
        need(encode(saved)==encode(analyze(**saved['arguments'])),'whole deterministic reproduction');out=dict(verified=True,sha256=a.analysis_sha)
    else:
        x=analyze(a.summary,a.summary_sha);p=absolute(a.output);b=encode(x)
        need(p.name.startswith(NAME+'_') and not p.exists() and len(b)<=CAP,'exclusive bounded output')
        with p.open('xb') as f:need(f.write(b)==len(b),'complete write');f.flush();os.fsync(f.fileno())
        fd=os.open(p.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        out=dict(output=str(p),sha256=sha(p))
    print(encode(out).decode(),end='')


if __name__=='__main__':main()
