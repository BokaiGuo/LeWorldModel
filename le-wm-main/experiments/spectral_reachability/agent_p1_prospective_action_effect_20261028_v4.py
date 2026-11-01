"""Saved-only complete native25 phase-goal audit; no prepare/run/model/physics."""
from __future__ import annotations
import argparse
import ast
import inspect
import io
import os
from collections import Counter
from copy import deepcopy
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p1_prospective_action_effect_20261028_v3 as v3

v2=v3.v2
ROOT=v2.ROOT
require=v2.require
NAME='agent_p1_prospective_action_effect_20261028_v4'
ACTUAL_SHA='2e7794d32602ca2b045202099fedb6b99ac5988d6255ee68870032219530084c'
FROZEN_V3={
    'experiments/spectral_reachability/agent_p1_prospective_action_effect_20261028_v3.py':'bd6b909233f0d84ee7ecf2ecdc3e33bcce10a71a1764068de0deb606c839b79c',
    'tests/test_agent_p1_prospective_action_effect_20261028_v3.py':'3ffc23d9355a6532673b6fc5c6f78fbcf12b76db5abc10b1b3da2521194a785f',
    'innerdocs/agent_p1_prospective_action_effect_20261028_v3.md':'a1359686d007b1d7e1a20f0abb08dc719c086776437da5d68ac74f5b9c258508',
}


def authority():
    bindings=v3.authority()
    for name,h in FROZEN_V3.items():
        p=ROOT/name
        require(p.is_file() and not p.is_symlink() and v2.sha(p)==h,'frozen v3 authority')
        bindings[str(p)]=h
    require(Path(v3.__file__).resolve()==ROOT/next(iter(FROZEN_V3)),'imported frozen v3')
    return bindings


def exact(a,b):
    return v2.canonical(a)==v2.canonical(b)


def vector(a):
    require(isinstance(a,list) and len(a)==7 and all(type(x) is float and np.isfinite(x) for x in a),'finite literal full7')
    return np.asarray(a,dtype='<f8')


def native_info(output,e):
    """Decode lossless native info with its independent content hash/inventory."""
    receipt=e['info'];name=receipt['file']
    require(type(name) is str and Path(name).name==name,'native info local file')
    path=Path(output)/name
    require(path.is_file() and not path.is_symlink() and v2.sha(path)==receipt['sha256'],'native info SHA')
    with np.load(io.BytesIO(path.read_bytes()),allow_pickle=False) as z:
        used=set()
        def visit(x):
            if isinstance(x,dict) and set(x)=={'array','dtype','shape'}:
                a=z[x['array']];used.add(x['array'])
                require(a.dtype.str==x['dtype'] and list(a.shape)==x['shape'] and not a.dtype.hasobject,'native info dtype/shape')
                return a.copy()
            if isinstance(x,dict):return {k:visit(v) for k,v in x.items()}
            if isinstance(x,list):return [visit(v) for v in x]
            return x
        result=visit(receipt['tree'])
        require(used==set(z.files),'native info complete inventory')
    return result


def labels(goal,state):
    """Independent literal PushT full7 norm and native success, no new metric."""
    g=vector(goal);s=vector(state)
    position=np.linalg.norm(g[:4]-s[:4]);angle=np.abs(g[4]-s[4])
    angle=np.minimum(angle,2*np.pi-angle)  # Preserve native expression, NOT a new wrapped metric.
    return dict(native_success=bool(position<20 and angle<np.pi/9),native_distance=float(np.linalg.norm(g-s)))


def validate_branch(output,row,scoped):
    require(scoped and all(e.get('job_digest')==row['job_digest'] and e.get('anchor_index')==row['anchor_index'] and
        e.get('word')==row['word'] for e in scoped),'exact branch event scope')
    indices=[e['index'] for e in scoped]
    require(all(type(i) is int for i in indices) and indices==sorted(set(indices)),'strict actual event order/index')
    for e in scoped:
        require(e['event_sha256']==v2.digest({k:v for k,v in e.items() if k!='event_sha256'}),'authenticated event body')
    def one(kind,**fields):
        es=[e for e in scoped if e['kind']==kind and all(e.get(k)==v for k,v in fields.items())]
        require(len(es)==1,'unique '+kind+' '+str(fields));return es[0]
    def operation(resource,**fields):
        a=[e for e in scoped if e['kind']=='operation_attempted' and e.get('resource')==resource and e['fields']==fields]
        require(len(a)==1,'unique operation '+resource);a=a[0]
        b=[e for e in scoped if e['kind']=='operation_returned' and e.get('resource')==resource and e['operation_index']==a['operation_index']]
        require(len(b)==1 and b[0]['fields']==fields and a['index']<b[0]['index'],'paired operation '+resource)
        return a,b[0]
    raw=[e for e in scoped if e['kind']=='raw_return']
    expected=[('warmup',i) for i in range(1,11)]+[('prefix',i) for i in range(11,16)]+[('suffix',i) for i in range(1,11)]
    require([(e['phase'],e['step']) for e in raw]==expected,'complete25 literal phase chronology')
    recipe=one('placement_recipe');attached=one('goal_attached')
    native=recipe['native_goal'];target=row['goal'];vector(native);vector(target)
    seed_start,seed_end=operation('reset',role='seed',seed=recipe['slot']['reset_seed'])
    place_start,place_end=operation('reset',role='placement',seed=0)
    bind_start,bind_end=operation('goal_bind')
    require(seed_start['index']<seed_end['index']<recipe['index']<place_start['index']<place_end['index']<raw[0]['index'] and
        raw[9]['index']<bind_start['index']<bind_end['index']<attached['index']<raw[10]['index'],
        'placement -> warmup10 -> actual bind -> attached -> prefix chronology')
    # No hidden goal mutation/reset is permitted in the intervening or task phases.
    resets=[e for e in scoped if e['kind']=='operation_attempted' and e.get('resource')=='reset']
    binds=[e for e in scoped if e['kind']=='operation_attempted' and e.get('resource')=='goal_bind']
    setters=[e for e in scoped if e['kind']=='operation_attempted' and e.get('resource')=='goal_setter']
    require(len(resets)==2 and len(binds)==1 and len(setters)==3,'exact reset/bind/setter census')
    for setter,(start,end) in zip(setters,[(seed_start,seed_end),(place_start,place_end),(bind_start,bind_end)],strict=True):
        returns=[e for e in scoped if e['kind']=='operation_returned' and e.get('resource')=='goal_setter' and e['operation_index']==setter['operation_index']]
        require(len(returns)==1 and start['index']<setter['index']<returns[0]['index']<end['index'],'goal setter exact enclosing operation')
    require(exact(attached['goal'],target) and attached['goal_rgb_sha256']==row['goal_pixel_sha256'] and
        exact(attached['snapshot'],raw[9]['snapshot']),'canonical bound task goal/RGB and unchanged warmup state')
    roots=[e for e in scoped if e['kind']=='fresh_observation' and e['receipt']['phase']=='root']
    require(len(roots)==3 and all(attached['index']<e['index']<raw[10]['index'] and
        exact(e['receipt']['snapshot'],attached['snapshot']) for e in roots),'bound root triplet before prefix')
    intents=[e for e in scoped if e['kind']=='operation_attempted' and e.get('resource')=='public_step']
    returned=[e for e in scoped if e['kind']=='operation_returned' and e.get('resource')=='public_step']
    require(len(intents)==len(returned)==25,'all25 public dispatch pairs')
    records=[]
    for i,(e,a,b) in enumerate(zip(raw,intents,returned,strict=True)):
        goal=native if i<10 else target
        require(a['operation_index']==b['operation_index'] and exact(a['fields'],b['fields']) and
            a['fields']==dict(phase=e['phase'],step=e['step'],raw_bits=e['raw_bits']) and
            a['index']<b['index']<e['index'] and (i==0 or raw[i-1]['index']<a['index']),
            'actual caller/action/phase dispatch chronology')
        if i==0:require(place_end['index']<a['index'],'placement before warmup dispatch')
        if i==10:require(attached['index']<a['index'],'binding before first task dispatch')
        s=vector(e['state']);g=vector(goal);expected_labels=labels(goal,e['state'])
        require(exact(e['snapshot']['state'],e['state']) and e['snapshot']['state_float64_bits']==s.tobytes().hex(),
                'native full7 state/snapshot/signed bits')
        require(all(type(e[k]) is bool for k in ('native_success','terminated','truncated')) and
            e['native_success']==e['terminated']==expected_labels['native_success'] and e['truncated'] is False and
            type(e['native_distance']) is float and type(e['reward']) is float and
            e['native_distance']==expected_labels['native_distance'] and e['reward']==-expected_labels['native_distance'],
            'independent full7 native distance/reward/success/termination')
        require(i==24 or not e['terminated'],'no raw actions after native stop')
        info=native_info(output,e)
        block=e['snapshot']['bodies']['block']
        block_pose=np.asarray(block['position']+[block['unwrapped_angle']],dtype='<f8')
        require(info['env_name']=='PushT' and v2.exact_array(info['goal_state'],g) and
            v2.exact_array(info['pos_agent'],s[:2]) and v2.exact_array(info['vel_agent'],s[5:]) and
            v2.exact_array(info['block_pose'],block_pose) and
            v2.exact_array(info['goal_proprio'],g[[0,1,5,6]]) and
            v2.exact_array(block_pose[:2],s[2:4]) and block_pose[2]%(2*np.pi)==s[4],
            'native saved info phase goal and observed full7 state')
        records.append(dict(event=deepcopy(e),goal=deepcopy(goal),row_goal=deepcopy(target),row_index=row['index']))
    return records


def validate_phases(output,report,events):
    require(report['status']=='complete_unadmitted' and len(report['rows'])==40,'v4 currently complete40 only; partial not certified')
    records=[];jobs=set()
    for row in report['rows']:
        require(row['job_digest'] not in jobs,'unique branch job');jobs.add(row['job_digest'])
        scoped=[e for e in events if e.get('job_digest')==row['job_digest']]
        records.extend(validate_branch(output,row,scoped))
    allraw=[e for e in events if e['kind']=='raw_return']
    require(len(allraw)==1000 and exact([x['event'] for x in records],allraw),'all1000 raw returns exactly once, no orphan or reordered event')
    counts=Counter(x['event']['phase'] for x in records)
    return records,dict(rows=40,raw_returns=len(records),**counts)


def event_adapter(records):
    """Authenticate actual caller event AND complete consumption order; never state lookup."""
    cursor=0
    def phase_labels(row,event):
        nonlocal cursor
        require(cursor<len(records),'no extra native caller')
        r=records[cursor]
        require(row['index']==r['row_index'] and exact(row['goal'],r['row_goal']) and
            exact(event,r['event']),'exact authenticated caller event/job/index/phase/order')
        cursor+=1
        return v2.native_labels(r['goal'],event['state'])
    def finish():
        require(cursor==len(records),'every validated raw event consumed exactly once');return cursor
    return phase_labels,finish


def adapted_tree():
    """Exactly one AST call substitution; all require/native/payload checks retained."""
    tree=ast.parse(inspect.getsource(v2.consume));needle=ast.parse("native_labels(row['goal'],e['state'])",mode='eval').body
    count=0
    class Adapt(ast.NodeTransformer):
        def visit_Call(self,node):
            nonlocal count
            if ast.dump(node,include_attributes=False)==ast.dump(needle,include_attributes=False):
                count+=1
                return ast.copy_location(ast.parse('phase_labels(row,e)',mode='eval').body,node)
            return self.generic_visit(node)
    tree=Adapt().visit(tree);require(count==1,'exactly one frozen native_labels caller')
    return ast.fix_missing_locations(tree),count


def consume_local(output,report_sha,mappings,records):
    usage=Counter();phase_labels,finish=event_adapter(records)
    def canonical(value):
        raw=v2.canonical(value)
        if raw not in mappings:return raw
        projected,index=mappings[raw];usage[index]+=1;return projected
    tree,count=adapted_tree()
    namespace=dict(v2.consume.__globals__,canonical=canonical,phase_labels=phase_labels)
    exec(compile(tree,str(Path(__file__).resolve())+'::frozen_consume_phase_adapter','exec'),namespace)
    result=namespace['consume'](output,report_sha)
    require(dict(usage)=={index:1 for _,index in mappings.values()},'exact terminal projection usage')
    return result,dict(terminal_rows=len(usage),native_calls=finish(),AST_call_substitutions=count)


def consume(output,report_sha):
    bindings=authority();output=Path(output).absolute()
    report=v2.teacher.read(output/'report.json',report_sha,v2.phys.REPORT_CAP)
    events=v2.read_events(output/'journal.jsonl')
    require(v2.sha(output/'journal.jsonl')==report['artifacts']['journal.jsonl'],'report-bound journal')
    counts=v3.validate_eventset(events)
    records,phases=validate_phases(output,report,events)
    mappings=v3.terminal_projection(report,events)
    preprocessing=v3.validate_preprocessing(output,report['images'])
    base,usage=consume_local(output,report_sha,mappings,records)
    require(v2.sha(output/'report.json')==report_sha,'producer report unchanged')
    for name,h in report['artifacts'].items():
        p=output/name
        require(Path(name).name==name and p.is_file() and not p.is_symlink() and v2.sha(p)==h,'producer artifact unchanged')
    own=[Path(__file__),ROOT/'tests'/f'test_{NAME}.py',ROOT/'innerdocs'/f'{NAME}.md']
    require(base['status']=='saved_adapter_audited_not_P1_admission','complete original checks with two exact local adapters')
    return dict(schema=NAME+'.consume',status='strict_phase_saved_adapter_audited_not_P1_admission',report_sha256=report_sha,
        v2_adapted_consume=base,phase_validation=phases,adapter_usage=usage,full_event_counts=counts,preprocessing=preprocessing,
        frozen_authority=bindings,consumer_files={str(p.resolve()):v2.sha(p) for p in own},
        original_failures=['v2 line630: journal envelope vs raw observation body','v3/v2 line635: final task goal applied to placement warmup'],
        new_neural_calls=0,new_physics_calls=0,P1_admission=False,
        interpretation='saved consumer protocol repair only; no metric/gate change, new selector, deployed correction, or explanation of original P1 reversals')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('consume',))
    parser.add_argument('--output',required=True);parser.add_argument('--report-sha',required=True);parser.add_argument('--receipt')
    args=parser.parse_args();result=consume(args.output,args.report_sha)
    if args.receipt:
        p=Path(args.receipt).absolute()
        require(p.name.startswith(NAME+'_') and p.suffix=='.json' and p.resolve()==p and
            output_outside(p,Path(args.output).resolve()),'own receipt outside frozen producer output')
        raw=v2.canonical(result);require(len(raw)<=v2.phys.PREP_CAP,'bounded receipt')
        with p.open('xb') as f:
            require(f.write(raw)==len(raw),'complete receipt');f.flush();os.fsync(f.fileno())
        v2.phys.adapter.paired.fsync_dir(p.parent)
        result=dict(receipt=str(p),sha256=v2.sha(p),status=result['status'])
    print(v2.canonical(result).decode())


def output_outside(path,output):
    return not path.is_relative_to(output)


if __name__=='__main__':main()
