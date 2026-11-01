"""Saved-only independent four-slot physical arithmetic; no producer imports."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np

NAME='agent_p2_wallcapped_physical_pair_analysis_20261029_v1'
PHYSICAL='agent_p2_wallcapped_physical_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
ORDER=((0,'joint'),(0,'composition'),(1,'composition'),(1,'joint'))
CP=(0,5,10,15)
CAP=128*1024**2
KINDS={'run_started','run_finished','operation_dispatch_intent','operation_returned','operation_failed',
    'placement_recipe','goal_attached','invalid_observation','fresh_observation','returned_transition','own_response','branch_closed'}
AXES=('constructor','close','reset','public_step','physics_step','set_state','goal_setter',
    'goal_render','goal_bind','explicit_render')
ERRORS=('native7_distance','joint_xy','agent_xy','object_xy','angle_native_term',
    'angle_wrapped','velocity_error','native7_reward')


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical(x):return json.dumps(x,sort_keys=True,allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def hash_value(value):
    require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'actual SHA256')


def unique(items):
    out={}
    for k,v in items:
        require(k not in out,'duplicate JSON key');out[k]=v
    return out


def file_path(value):
    p=Path(value)
    require(p.is_absolute() and p.resolve()==p and p.is_file() and not p.is_symlink(),'absolute regular sealed file')
    return p


def read(value,expected=None):
    p=file_path(value);require(p.stat().st_size<=CAP,'bounded metadata')
    if expected is not None:hash_value(expected);require(sha(p)==expected,'external SHA256')
    v=json.loads(p.read_bytes(),object_pairs_hook=unique);canonical(v);return v


def vector(value,n):
    require(type(value) is list and len(value)==n and all(type(x) in (int,float) for x in value),'literal numeric vector, no bool')
    x=np.asarray(value,np.float64);require(np.isfinite(x).all(),'finite vector');return x


def bits(value,n):
    require(type(value) is list and len(value)==n,'fixed raw action count')
    a=np.asarray([vector(v,2) for v in value],dtype=np.float64).reshape(n,2)
    b=a.astype(np.float32)
    require(np.isfinite(b).all() and np.all(abs(b)<=1) and np.array_equal(a,b.astype(np.float64)),
        'lossless raw float32 box, no repair')
    return b


def metrics(goal,state):
    """Literal native success formula, not a replacement wrapped-angle gate."""
    g=vector(goal,7);s=vector(state,7);d=s-g
    xy=float(np.linalg.norm(d[:4]));angle=float(abs(d[4]));native=min(angle,2*np.pi-angle)
    distance=float(np.linalg.norm(d))
    return dict(native7_delta=d.tolist(),native7_distance=distance,native7_reward=-distance,joint_xy=xy,
        agent_xy=float(np.linalg.norm(d[:2])),object_xy=float(np.linalg.norm(d[2:4])),
        angle_native_term=float(native),angle_wrapped=float(abs(np.arctan2(np.sin(d[4]),np.cos(d[4])))),
        velocity_error=float(np.linalg.norm(d[5:])),native_success=bool(xy<20 and native<np.pi/9),
        native_angle_negative=bool(native<0),object_only_success=None)


def journal(payload,report):
    require(len(payload)<=64*1024**2 and payload.endswith(b'\n'),'bounded closed journal')
    require(hashlib.sha256(payload).hexdigest()==report['journal_sha256'],'original journal bytes hash')
    result=[];tail=None
    for i,line in enumerate(payload.splitlines()):
        require(line and len(line)<=16*1024**2,'single LF, no empty record filtering')
        e=json.loads(line,object_pairs_hook=unique)
        require(line==canonical(e) and e['kind'] in KINDS,'exact spaced native JSON ABI / known kind')
        require(type(e['index']) is int and e['index']==i and e['previous_sha256']==tail
            and e['event_sha256']==digest({k:v for k,v in e.items() if k!='event_sha256'}),'whole physical hashchain')
        tail=e['event_sha256'];result.append(e)
    require(tail==report['journal_chain_tail'] and result[0]['kind']=='run_started'
        and result[-1]['kind']=='run_finished','physical journal lifecycle')
    require(result[0]['prepare_sha256']==report['execution_binding']['prepare_sha256']
        and result[0]['contract']==report['contract']
        and all(result[-1][k]==report[k] for k in ('status','counts','errors')),'bound lifecycle fields')
    return result


def counters(events,report):
    counts={k:dict(attempted=0,completed=0) for k in AXES}
    # Each branch owns separate counters (not four aliases).
    by=[deepcopy(counts) for _ in range(4)];started={};ended={};stack=[];closed=0
    for e in events:
        if e['kind']=='branch_closed':
            require(closed<4 and e['receipt']==report['rows'][closed],'ordered physical branch closure');closed+=1
        if 'branch' in e:
            require(type(e['branch']) is int and e['branch']==closed and closed<4,'active branch chronology')
        if e['kind'] not in ('operation_dispatch_intent','operation_returned','operation_failed'):continue
        o=e['receipt'];i=o['index'];b=o['branch'];k=o['resource']
        require(type(i) is type(b) is int and b==closed and 0<=b<4 and k in counts,'physical units and slot')
        require(all(o[f]==report['rows'][b][f] for f in ('root_slot','condition','arm','selection_digest')),
            'operation identity matches selected slot')
        if e['kind']=='operation_dispatch_intent':
            require(i==len(started) and o['status']=='pending','durable unique intent')
            started[i]=o;stack.append(i);counts[k]['attempted']+=1;by[b][k]['attempted']+=1
        else:
            require(stack and stack.pop()==i and i not in ended and i in started
                and o['status']==('returned' if e['kind']=='operation_returned' else 'failed')
                and dict({f:v for f,v in o.items() if f!='error'},status='pending')==started[i],
                'exact nested return/failure, no refund')
            ended[i]=o
            if o['status']=='returned':counts[k]['completed']+=1;by[b][k]['completed']+=1
    require(not stack and len(started)==len(ended) and counts==report['counts']
        and [ended[i] for i in range(len(ended))]==report['actual_operation_receipts'], 'independent actual paid census')
    require([e['receipt'] for e in events if e['kind']=='branch_closed']==
        [x for x in report['rows'] if x['status']!='not_started'],'closed and unstarted denominator')
    return dict(total=counts,per_slot=by,closed_slots=closed)


def summarize_slot(i,plan,row,events,p,paid):
    root,arm=ORDER[i]
    local=[e for e in events if e.get('branch')==i or
        (e['kind'] in ('operation_dispatch_intent','operation_returned','operation_failed')
            and e['receipt']['branch']==i)]
    require(type(row['index']) is int and row['index']==i and row['root_slot']==root and row['arm']==arm
        and row['condition']=='off' and row['plan']==plan,'fixed original four slots')
    result=dict(index=i,root_slot=root,arm=arm,status=row['status'],plan_missing=plan is None,
        native_goal=None,model_terminal_Q=None,actual_latent_Q=None,steps=[None]*16,warmup=[None]*11,
        actions=[dict(step=j,attempted=False,returned=False,planned_raw=None,dispatch_bits_match=None,
            actual_bits_match=None) for j in range(1,16)],checkpoints=[dict(step=j,value=None) for j in CP],
        final_observed=None,full15=False,counts=deepcopy(paid),first_error=row['first_error'])
    if plan is None:
        require(row['status']=='export_missing' and not local and not row['trajectory'] and not row['responses']
            and all(v['attempted']==v['completed']==0 for v in paid.values())
            and row['own_attempted']==row['own_returned']==row['warmup_attempted']==row['warmup_returned']==0
            and row['final_state'] is row['final_metrics'] is row['native_success'] is None,'missing export has no execution')
        return result
    commands=bits(plan['commands'],15);warm=bits(plan['planner_plan']['warm_raw10'],10)
    require(plan['selection_digest']==row['selection_digest'] and plan['root_slot']==root and plan['arm']==arm
        and plan['condition']=='off' and plan['task_id']==row['task_id']
        and plan['selection_digest']==digest({k:v for k,v in plan.items() if k!='selection_digest'})
        and hashlib.sha256(commands.tobytes()).hexdigest()==plan['raw_float32_sha256']
        and commands.tobytes().hex()==plan['planner_plan']['raw_float32_bits'],'selected full15 literal witness')
    result['model_terminal_Q']=plan['planner_plan']['model_Q']
    for j in range(15):result['actions'][j]['planned_raw']=commands[j].tolist()
    goal=p['source_roots'][root]['goal'];warmgoal=p['warm_traces'][root][0]['pre']['goal_state']
    bindings=[e for e in local if e['kind']=='goal_attached']
    require(len(bindings)==int(row['initial_exact']),'actual task bind')
    if bindings:require(bindings[0]['goal']==goal,'original task goal after warmup')
    observations=[e['receipt'] for e in local if e['kind']=='fresh_observation' and e['receipt']['overlay_index']==0]
    require(observations==row['trajectory'],'every selected RGB state, not endpoint alone')
    lookup={};trans={};ops={}
    for e in local:
        if e['kind']=='operation_dispatch_intent' and e['receipt']['resource']=='public_step':
            o=e['receipt'];key=(o['phase'],o['step']);require(key not in ops,'one raw dispatch per phase/step');ops[key]=(e,o)
        if e['kind']=='returned_transition':
            key=(e['phase'],e['step']);require(key not in trans,'unique returned state');trans[key]=e
    for ob in observations:
        key=(ob['phase'],ob['step']);require(key not in lookup,'one selected frame per step');lookup[key]=ob
    for phase,label,raws,n,expected_goal in (('warmup','warmup',warm,10,warmgoal),('plan','own',commands,15,goal)):
        selected=[e for e in observations if e['phase']==phase]
        require([x['step'] for x in selected]==list(range(len(selected))),'contiguous observations; no omitted interior frame')
        attempts=[(key,value) for key,value in ops.items() if key[0]==phase]
        returned=[e for key,e in trans.items() if key[0]==phase]
        require([k[1] for k,_ in attempts]==list(range(1,len(attempts)+1)) and len(attempts)<=n
            and len(attempts)==row[label+'_attempted'] and [e['step'] for e in returned]==list(range(1,len(returned)+1))
            and len(returned)<=row[label+'_returned']<=len(attempts),'complete attempted/returned denominators')
        for (key,(intent,o)) in attempts:
            require(o['raw_bits']==raws[key[1]-1].tobytes().hex(),'literal expected requested action')
            if phase=='plan':
                require(bindings and bindings[0]['index']<intent['index'],'task bound before control')
                result['actions'][key[1]-1].update(attempted=True,dispatch_bits_match=True)
        for t in returned:
            key=(phase,t['step']);require(key in ops and ops[key][0]['index']<t['index'],'intent precedes actual return')
            require(t['actual_dtype']==np.dtype(np.float32).str and t['actual_bits']==t['requested_bits']==raws[t['step']-1].tobytes().hex(),
                'actual native action not substituted')
            require(t['goal_state']==expected_goal and t['state']==t['snapshot']['state'],'phase-specific native goal/state')
            m=metrics(expected_goal,t['state'])
            require(type(t['reward']) is float and t['reward']==m['native7_reward']
                and type(t['terminated']) is bool and t['terminated'] is m['native_success']
                and type(t['truncated']) is bool,'independent literal native labels')
            if phase=='plan':result['actions'][t['step']-1].update(returned=True,actual_bits_match=True)
        for ob in selected:
            require(ob['state']==ob['snapshot']['state'],'observed native state snapshot')
            step=ob['step'];m=metrics(expected_goal,ob['state'])
            if step:
                require((phase,step) in trans and trans[phase,step]['state']==ob['state'],'observed state joins actual return')
            elif phase=='plan':
                require(row['initial_exact'] and row['root_state']==ob['state']==p['source_roots'][root]['root_state']
                    and row['goal']==goal,'cp0 actual original root')
            x=dict(step=step,state=ob['state'],goal=expected_goal,metrics=m,
                png_sha256=ob['file_sha256'],pixel_sha256=ob['pixel_sha256'])
            result['warmup' if phase=='warmup' else 'steps'][step]=x
    if any(x is not None for x in result['steps']):result['native_goal']=goal
    responses=row['responses'];own=[e['response'] for e in local if e['kind']=='own_response']
    require(own==responses and [x['step'] for x in responses]==list(range(1,len(responses)+1)),'all responses retained')
    for x in responses:
        t=trans['plan',x['step']];require(all(t[k]==x[k] for k in ('state','reward','terminated','truncated')),'response joins actual native return')
    cps=[dict(step=j,value=result['steps'][j]) for j in CP];result['checkpoints']=cps
    require(row['missing_checkpoints']==[j for j in CP if result['steps'][j] is None]
        and [x['step'] for x in row['checkpoints']]==[j for j in CP if result['steps'][j] is not None],'cp denominator explicit')
    for c in row['checkpoints']:
        ob=lookup['plan',c['step']];require(all(c[k]==ob[k] for k in ('state','snapshot','png','file_sha256','pixel_sha256')),'checkpoint joins exact observation')
    values=[x for x in result['steps'][1:] if x is not None]
    if values:
        result['final_observed']=values[-1]
        require(row['final_state']==values[-1]['state'] and row['native_success'] is values[-1]['metrics']['native_success'],
            'original endpoint agrees with every-step recomputation')
    else:require(row['final_state'] is None,'unobserved endpoint remains unknown')
    result['full15']=result['steps'][15] is not None
    if row['status'] in ('horizon_completed','native_stopped'):
        require(len(responses)==row['own_returned']==sum(x['returned'] for x in result['actions']), 'clean native records cannot omit return')
        if row['status']=='horizon_completed':require(result['full15'],'horizon complete really returned15')
    return result


def difference(joint,composition):
    if joint is None or composition is None:return None
    require(joint['goal']==composition['goal'],'paired original task goal')
    return dict(composition_minus_joint={k:composition['metrics'][k]-joint['metrics'][k] for k in ERRORS},
        native_success=[joint['metrics']['native_success'],composition['metrics']['native_success']])


def paired(rows,plans):
    out=[]
    for root in (0,1):
        ji=next(i for i,k in enumerate(ORDER) if k==(root,'joint'))
        ci=next(i for i,k in enumerate(ORDER) if k==(root,'composition'));j,c=rows[ji],rows[ci]
        jp,cp=plans[ji],plans[ci];steps=[]
        for n in range(16):
            js,cs=j['steps'][n],c['steps'][n]
            same_raw=None if n==0 or jp is None or cp is None else bool(
                bits(jp['commands'],15)[n-1].tobytes()==bits(cp['commands'],15)[n-1].tobytes())
            steps.append(dict(step=n,metrics=difference(js,cs),planned_action_equal=same_raw,
                executed_action_equal=None if n==0 or not (j['actions'][n-1]['returned'] and c['actions'][n-1]['returned']) else same_raw,
                exact_state_equal=None if js is None or cs is None else vector(js['state'],7).tobytes()==vector(cs['state'],7).tobytes()))
        out.append(dict(root_slot=root,joint_index=ji,composition_index=ci,steps=steps,
            checkpoints=[steps[n] for n in CP],both_full15=j['full15'] and c['full15'],
            terminal15=steps[15]['metrics'],pair_eligible=steps[0]['metrics'] is not None,
            same_root_state=None if j['steps'][0] is None or c['steps'][0] is None else steps[0]['exact_state_equal']))
    return out


def analyze(output,report_sha,physical_receipt,receipt_sha):
    directory=Path(output);require(directory.is_absolute() and directory.resolve()==directory,'absolute output directory')
    rp=directory/'report.json';r=read(rp,report_sha);authority=read(physical_receipt,receipt_sha)
    require(r['schema']==PHYSICAL+'.report' and authority['schema']==PHYSICAL+'.consume'
        and authority['physical_report_sha256']==report_sha and authority['rows']==r['rows']
        and authority['counts']==r['counts'] and authority['errors']==r['errors'] and authority['status']==r['status']
        and r['fixed_denominator']==authority['fixed_denominator']==4 and r['known_roots']==authority['known_roots']==2
        and r['fresh_roots']==authority['fresh_roots']==0
        and r['status'] in ('wallcapped_export_records_native_recorded_not_admitted','partial_or_failed'),
        'actual strict physical authority / fixed four')
    for k in ('P1_admission','P2_admission','P2_mechanism','scientific_admission','physical_release'):
        require(r[k] is authority[k] is False,'no scientific authority')
    bindings=deepcopy(authority['bindings']);require(bindings[str(rp)]==report_sha,'receipt binds actual report')
    for f,h in bindings.items():hash_value(h);require(sha(file_path(f))==h,'all inherited strict bindings unchanged')
    b=r['execution_binding'];p=read(b['prepared'],b['prepare_sha256'])
    require(bindings[b['prepared']]==b['prepare_sha256'] and p['schema']==PHYSICAL+'.prepare'
        and p['plans']==authority['plans'] and len(p['plans'])==len(r['rows'])==4
        and p['planner_authority']==r['execution_binding']['planner_authority']==authority['planner_authority'],
        'same actual prepare / plans / model authority')
    payload=file_path(directory/'journal.jsonl').read_bytes();events=journal(payload,r);paid=counters(events,r)
    rows=[summarize_slot(i,p['plans'][i],r['rows'][i],events,p,paid['per_slot'][i]) for i in range(4)]
    for f,h in bindings.items():require(sha(file_path(f))==h,'unchanged after saved numeric recomputation')
    bindings[str(file_path(physical_receipt))]=receipt_sha
    for f in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):
        bindings[str(f)]=sha(f)
    return dict(schema=NAME+'.analysis',status='saved_numerics_recomputed_NOT_admitted',rows=rows,pairs=paired(rows,p['plans']),
        physical_report_sha256=report_sha,physical_receipt_sha256=receipt_sha,fixed_denominator=4,known_roots=2,
        pair_denominator=2,fresh_roots=0,paid=paid,errors=r['errors'],bindings=bindings,
        upstream_full_strict_rebuilt_here=False,authority='external SHA-pinned physical strict receipt plus unchanged complete bindings',
        original_success='jointXY<20 and min(abs(angle),2pi-abs(angle))<pi/9; not object-only coverage',
        outcome_selection=False,equal_FLOPs=False,equal_total_compute=False,P1_admission=False,P2_admission=False,
        P2_mechanism=False,physical_release=False,new_neural_calls=0,new_physics_calls=0)


def write_new(path,value):
    path=Path(path);body=canonical(value)+b'\n'
    require(path.is_absolute() and path.parent.resolve()==path.parent and path.name.startswith(NAME+'_')
        and not path.exists() and not path.is_symlink() and len(body)<=CAP
        and shutil.disk_usage(path.parent).free>=200*1024**2+len(body),'exclusive bounded own receipt')
    with path.open('xb') as f:require(f.write(body)==len(body),'complete receipt');f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(path=str(path),sha256=sha(path),bytes=len(body))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('output','report-sha','physical-receipt','receipt-sha','receipt'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();v=analyze(a.output,a.report_sha,a.physical_receipt,a.receipt_sha)
    print(json.dumps(write_new(a.receipt,v),sort_keys=True))


if __name__=='__main__':main()
