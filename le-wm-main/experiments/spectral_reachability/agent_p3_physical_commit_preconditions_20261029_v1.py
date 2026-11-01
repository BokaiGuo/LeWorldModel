"""Typed saved preconditions; never physical authorization or a fullstrict runner."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
from experiments.spectral_reachability import agent_p3_equal_cap_comparison_20261029_v1 as io
from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core

NAME='agent_p3_physical_commit_preconditions_20261029_v1'
ROOT=io.ROOT
require,sha,digest,canonical=io.require,io.sha,io.digest,io.canonical
ORDER=[(i,i//2,('fixed','adaptive')[i%2]) for i in range(4)]
SECTIONS={'task','anchor','restore_reference','actual_history','model_input','release','resources','model'}
FLAGS=dict(physical_release=False,execution_authorized=False,ready=False,P1_admission=False,
    P3_admission=False,C3=False,freshness=False,new_independent_roots=False)


def keys(obj,expected):require(type(obj) is dict and set(obj)==set(expected),'exact typed fields: '+str(sorted(expected)))


def pointer(obj,path):
    require(type(path) is str and (path=='' or path.startswith('/')),'JSON pointer')
    for part in ([] if path=='' else path[1:].split('/')):
        require('~' not in part.replace('~0','').replace('~1',''),'pointer escapes')
        part=part.replace('~1','/').replace('~0','~')
        if type(obj) is list:
            require(part.isdigit() and str(int(part))==part and int(part)<len(obj),'exact array pointer')
            obj=obj[int(part)]
        else:
            require(type(obj) is dict and part in obj,'existing object pointer');obj=obj[part]
    return deepcopy(obj)


def project(spec,documents,trace):
    if type(spec) is dict and '$ref' in spec:
        keys(spec,('$ref','pointer'));alias=spec['$ref']
        require(alias in documents,'known source document')
        trace.append(deepcopy(spec));return pointer(documents[alias],spec['pointer'])
    require(type(spec) is dict and bool(spec),'projection is structural dict or evidence ref; no literal data')
    return {k:project(v,documents,trace) for k,v in spec.items()}


def f32(value,shape,*,box=False):
    a=np.asarray(value)
    require(a.dtype.kind in 'fi' and a.shape==shape,'numeric shape '+str(shape))
    x=a.astype(np.float32)
    require(np.isfinite(x).all() and np.array_equal(a.astype(np.float64),x.astype(np.float64)),'lossless finite float32')
    if box:require(np.all(abs(x)<=1),'raw box [-1,1]; no clipping')
    return x


def same(a,b):return canonical(a)==canonical(b)
def bits(a,b):return a.shape==b.shape and a.tobytes()==b.tobytes()


def tokens(raw,mean,std):
    # Float32 subtraction then division, each exactly once, time-major rows.
    return np.divide(np.subtract(raw,mean,dtype=np.float32),std,dtype=np.float32)


def file_binding(ref):
    keys(ref,('path','sha256'));p=Path(ref['path'])
    require(p.is_absolute() and p.resolve()==p and not any(q.is_symlink() for q in (p,*p.parents))
        and p.is_file(),'exact regular source file')
    core.hash_value(ref['sha256']);require(sha(p)==ref['sha256'],'actual source bytes');return p


def png(frame):
    keys(frame,('absolute_step','plan_step','png','file_sha256','pixel_sha256','state','snapshot'))
    p=file_binding(dict(path=frame['png'],sha256=frame['file_sha256']))
    require(p.stat().st_size<=8*1024**2,'bounded PNG')
    with Image.open(p) as im:
        require(im.format=='PNG' and im.mode=='RGB' and im.size==(224,224),'original RGB224 PNG')
        pixels=np.asarray(im)
        require(io.hashlib.sha256(pixels.tobytes()).hexdigest()==frame['pixel_sha256'],'actual decoded pixels')
    require(len(frame['state'])==7 and np.isfinite(np.asarray(frame['state'],np.float64)).all()
        and same(frame['state'],frame['snapshot']['state']),'native7 snapshot state signed bits')


def authenticate(contract):
    docs={};missing=[];used=[]
    for alias,ref in contract['documents'].items():
        if ref is None:missing.append('document:'+alias);continue
        docs[alias]=io.load(ref)
    if missing:return docs,missing,used
    require(set(contract['authorities'])=={'planner','physical','history'},'three explicit actual strict authorities')
    for role,a in contract['authorities'].items():
        if a is None:missing.append('authority:'+role);continue
        keys(a,('report','receipt','report_schema','receipt_schema','receipt_status','report_sha_pointer','source_bindings'))
        require(a['report'] in docs and a['receipt'] in docs and a['report']!=a['receipt'],'distinct report/strict documents')
        r,s=docs[a['report']],docs[a['receipt']]
        require(r['schema']==a['report_schema'] and s['schema']==a['receipt_schema'] and
            s['status']==a['receipt_status'],'exact externally fixed authority ABI, no impersonation')
        require(pointer(s,a['report_sha_pointer'])==contract['documents'][a['report']]['sha256'],'strict actual report SHA join')
        require(type(a['source_bindings']) is list and a['source_bindings'],'actual authority source closure required')
        for binding in a['source_bindings']:
            keys(binding,('file','sha_ref'));ref=binding['file'];where=binding['sha_ref']
            keys(where,('$ref','pointer'))
            require(where['$ref'] in (a['report'],a['receipt']) and
                pointer(docs[where['$ref']],where['pointer'])==ref['sha256'],'source SHA explicitly joined to authority')
            file_binding(ref);used.append(ref)
    return docs,missing,used


def validate_slot(data,identity):
    keys(data,SECTIONS)
    task=data['task'];core.validate_task(task)
    require(task['root_context']['latent_dim']==192,'frozen latent192 task')
    anchor,reference=data['anchor'],data['restore_reference']
    for a in (anchor,reference):
        keys(a,('root_sha256','task_id','condition','goal','goal_pose','renderer_sha256','checkpoint','prefix51','warm_raw10','prefix_raw5'))
        require(a['condition']=='off' and a['root_sha256']==task['binding']['root_sha256'] and
            a['task_id']==task['binding']['task_id'],'same known root/task/off')
        require(a['checkpoint']['absolute_step']==15 and a['checkpoint']['plan_step']==5,'actual cp5 after warm10')
        png(a['checkpoint']);f32(a['warm_raw10'],(10,2),box=True);f32(a['prefix_raw5'],(5,2),box=True)
        frames=a['prefix51']
        require(type(frames) is list and len(frames)==51,'all51 original prefix frames, not terminal only')
        require(all(type(f) is dict and set(f)=={'pre','post','receipt'} for f in frames),'native prefix trace objects')
        sequence=[('warmup',step,k) for step in range(11) for k in range(3)]+[('plan',step,k) for step in range(6) for k in range(3)]
        require([(f['receipt']['phase'],f['receipt']['step'],f['receipt']['overlay_index']) for f in frames]==sequence,'complete chronological warmup0..10/plan0..5 off/on/off')
        require(all(f['receipt']['condition']==('on' if f['receipt']['overlay_index']==1 else 'off') and
            same(f['pre'],f['post']) and same(f['receipt']['state'],f['receipt']['snapshot']['state']) for f in frames),'renderer same-state and correct condition')
        terminal=frames[-3]['receipt']
        require(all(same(terminal[k],a['checkpoint'][k]) for k in ('state','snapshot','file_sha256','pixel_sha256')),'cp5 checkpoint linked to real prefix51 off frame')
        core.hash_value(a['renderer_sha256']);core.finite(a['goal'],(7,));core.finite(a['goal_pose'],(3,))
    # Paths may differ for independently replayed images; compare all remaining
    # native trace content and exact bytes through their SHA, never float tolerance.
    a,b=deepcopy(anchor),deepcopy(reference)
    for x in (a,b):x['checkpoint'].pop('png')
    require(same(a,b),'saved restore reference full51/raw/state/snapshot/renderer/goal exact')
    history=data['actual_history'];keys(history,('frames','raw10','latent3','history_tokens','goal192','normalizer_sha256','context_sha256'))
    require(len(history['frames'])==3,'three actual anchor frames')
    for frame,step in zip(history['frames'],(5,10,15),strict=True):
        png(frame);require(frame['absolute_step']==step and frame['plan_step']==step-10,'actual three-frame time identity')
        phase,local=('warmup',step) if step<=10 else ('plan',5)
        matched=[f['receipt'] for f in anchor['prefix51'] if f['receipt']['phase']==phase and f['receipt']['step']==local and f['receipt']['overlay_index']==0]
        require(len(matched)==1 and all(same(matched[0][k],frame[k]) for k in ('state','snapshot','file_sha256','pixel_sha256')),'three input frames tied to real source prefix trace')
    require(history['frames'][-1]==anchor['checkpoint'],'last actual frame exact cp5')
    raw=f32(history['raw10'],(10,2),box=True)
    expected=np.concatenate([f32(anchor['warm_raw10'],(10,2),box=True)[-5:],f32(anchor['prefix_raw5'],(5,2),box=True)])
    require(bits(raw,expected),'chronological actual history warm tail5+prefix5')
    mean=f32(task['action_mean'],(2,));std=f32(task['action_std'],(2,));require(np.all(std>0),'positive std')
    incoming=tokens(raw,mean,std).reshape(2,10)
    require(bits(f32(history['history_tokens'],(2,10)),incoming),'both actual history tokens regenerated once')
    latent=f32(history['latent3'],(3,192));goal=f32(history['goal192'],(192,))
    require(bits(goal,f32(task['goal'],(192,))) and bits(latent,f32(task['root_context']['latent_history'],(3,192)))
        and bits(incoming,f32(task['root_context']['action_history'],(2,10))),'task exact actual context/goal')
    require(history['context_sha256']==core.digest(task['root_context']) and
        history['normalizer_sha256']==core.digest([task['action_mean'],task['action_std']]),'producer compact hash identity')
    model=data['model'];keys(model,('expected_model_hash','before','after','loaded_bytes','sources_before','sources_after'))
    require(same(model['before'],model['after']) and model['before']['original_state_dict']==model['expected_model_hash']==task['binding']['model_sha256'],
        'frozen model before/after/task identity')
    require(type(model['loaded_bytes']) is int and model['loaded_bytes']>0 and same(model['sources_before'],model['sources_after'])
        and model['sources_before'],'actual loaded bytes and source closure')
    for path,h in model['sources_before'].items():file_binding(dict(path=path,sha256=h))
    release=data['release'];keys(release,('plan_available','status','released_plan','core_plan','whole_replay','segmented_replay',
        'guards_passed','errors','context_sha256','task_sha256'))
    require(release['context_sha256']==core.digest(task['root_context']) and release['task_sha256']==core.digest(task),'release same task/actual history')
    rr=data['resources'];core.audit_ledger(rr)
    require(release['plan_available'] is True and release['status']=='released' and release['guards_passed'] is True
        and release['errors']==[] and same(release['released_plan'],release['core_plan']),'complete plan release, not diagnostic best')
    plan=release['released_plan'];commands=f32(plan['actions'],(len(plan['actions']),2),box=True)
    require(len(commands)>=5 and len(commands)%5==0,'full plan packet length; commit first5 only')
    require(same(plan['path'],release['whole_replay']) and same(plan['path'],release['segmented_replay']),'both complete plan replays exact')
    core.checked_path(task,task['root_context'],plan['actions'],plan['path'])
    require(rr['attempted']==rr['completed'] and all(e['status']=='returned' for e in rr['events'])
        and rr['completed']['replay_calls']>=2,'no failed attempt or incomplete final replay release')
    mi=data['model_input'];keys(mi,('latent3','history_tokens','outgoing_tokens','raw_commit5','goal192','input_fields'))
    require(mi['input_fields']==['latent3','history_tokens','outgoing_tokens','goal192'],'no privileged native state model input')
    commit=commands[:5];outgoing=np.concatenate([incoming,tokens(commit,mean,std).reshape(1,10)])
    require(bits(f32(mi['raw_commit5'],(5,2),box=True),commit) and bits(f32(mi['outgoing_tokens'],(3,10)),outgoing)
        and bits(f32(mi['latent3'],(3,192)),latent) and bits(f32(mi['history_tokens'],(2,10)),incoming)
        and bits(f32(mi['goal192'],(192,)),goal),'exact raw first5 and once-normalized model IO; no /100')
    return dict(slot=identity[0],root=identity[1],mode=identity[2],root_sha256=anchor['root_sha256'],
        task_id=anchor['task_id'],plan_step=5,absolute_step=15,anchor_sha256=digest(anchor),
        context_sha256=core.digest(task['root_context']),commit_raw5=commit.tolist(),
        commit_float32_sha256=io.hashlib.sha256(commit.tobytes()).hexdigest(),plan_sha256=core.digest(plan),
        resource_attempted=rr['attempted'],resource_completed=rr['completed'],
        runtime_guard_required='before public_step6: exact reset/warm/prefix51/cp5 state snapshot PNG/raw/goal; then dispatch',
        runtime_restore_proven=False,**FLAGS)


def build(contract):
    keys(contract,('schema','documents','authorities','slots'))
    require(contract['schema']==NAME+'.input' and len(contract['slots'])==4,'typed input/fixed4')
    docs,missing,sources=authenticate(contract);rows=[];seen=[]
    for row,identity in zip(contract['slots'],ORDER,strict=True):
        keys(row,('slot','root','mode','projection'))
        require((row['slot'],row['root'],row['mode'])==identity,'fixed4 roster')
        if missing or row['projection'] is None:
            rows.append(dict(slot=identity[0],root=identity[1],mode=identity[2],status='BLOCKED',reason='missing_actual_authority_or_projection',**FLAGS));continue
        trace=[];data={}
        for section,spec in row['projection'].items():
            refs=[];data[section]=project(spec,docs,refs)
            role='physical' if section in ('anchor','restore_reference') else 'history' if section=='actual_history' else 'planner'
            require(all(ref['$ref']==contract['authorities'][role]['report'] for ref in refs),'section bound to authenticated '+role+' report')
            trace.extend(refs)
        if set(data)!=SECTIONS or any(v is None for v in data.values()):
            rows.append(dict(slot=identity[0],root=identity[1],mode=identity[2],status='BLOCKED',reason='missing_sections',missing=sorted(SECTIONS-set(data)),**FLAGS));continue
        if data['release'].get('plan_available') is not True:
            core.audit_ledger(data['resources'])
            rows.append(dict(slot=identity[0],root=identity[1],mode=identity[2],status='BLOCKED',reason='upstream_no_plan_release',
                resource_attempted=data['resources']['attempted'],resource_completed=data['resources']['completed'],**FLAGS));continue
        require(data['anchor']['renderer_sha256'] in {r['file']['sha256'] for r in contract['authorities']['physical']['source_bindings']},'renderer code bound by physical authority')
        out=validate_slot(data,identity);out.update(status='saved_inputs_validated_pending_runtime_guard',evidence_refs=trace)
        rows.append(out);seen.append((identity,data))
    for root in (0,1):
        pair=[d for i,d in seen if i[1]==root]
        if len(pair)==2:
            require(same(pair[0]['task'],pair[1]['task']) and same(pair[0]['anchor'],pair[1]['anchor'])
                and same(pair[0]['actual_history'],pair[1]['actual_history']),'paired same-root/time/goal/history control')
    identities={i[1]:d['anchor']['root_sha256'] for i,d in seen}
    if set(identities)=={0,1}:require(identities[0]!=identities[1],'two known source identities, not duplicated root')
    for ref in contract['documents'].values():
        if ref is not None:require(sha(ref['path'])==ref['sha256'],'all documents stable')
    for ref in sources:file_binding(ref)
    return dict(schema=NAME+'.manifest',status='BLOCKED' if any(r['status']=='BLOCKED' for r in rows) else
        'saved_inputs_validated_pending_runtime_guard',contract_sha256=digest(contract),source_sha256=sha(__file__),
        authority_contract=contract['authorities'],documents=contract['documents'],missing=missing,
        denominator=dict(slots=4,known_roots=2,fresh_roots=0),rows=rows,
        authentication_scope='external strict SHA bindings and typed saved preconditions only; no producer/fullstrict rerun',**FLAGS)


def template():
    return dict(schema=NAME+'.input',documents={role+'_'+kind:None for role in ('planner','physical','history')
        for kind in ('report','receipt')},authorities={role:None for role in ('planner','physical','history')},
        slots=[dict(slot=i,root=root,mode=mode,projection=None) for i,root,mode in ORDER])


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--template',action='store_true');ap.add_argument('--contract');ap.add_argument('--contract-sha');ap.add_argument('--output')
    a=ap.parse_args(argv)
    if a.template:
        require(not any((a.contract,a.contract_sha,a.output)),'template only');print(canonical(template()).decode());return
    require(all((a.contract,a.contract_sha,a.output)),'explicit contract SHA and output required')
    contract=io.load(dict(path=str(Path(a.contract).absolute()),sha256=a.contract_sha))
    result=build(contract);p=Path(a.output).absolute();body=canonical(result)
    require(p.parent==ROOT/'results' and p.name.startswith(NAME) and p.resolve()==p and not p.exists(),'exclusive own output')
    require(len(body)<=16*1024**2,'manifest cap')
    with p.open('xb') as f:require(f.write(body)==len(body),'whole write');f.flush();os.fsync(f.fileno())
    print(canonical(dict(path=str(p),sha256=sha(p),status=result['status'])).decode())


if __name__=='__main__':main()
