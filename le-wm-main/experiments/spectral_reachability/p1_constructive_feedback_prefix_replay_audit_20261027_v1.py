"""Saved-only exact old/new physical prefix replay audit; no producer imports.

Read physical reports/prepares/manifests/journals and PNG only. Never open a
neural report, task latent, checkpoint, sourcepack or certificate. Tail outcomes
cannot select or reorder branches. This is NOT a suffix efficacy/P1 admission.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import numpy as np
from PIL import Image

NAME='p1_constructive_feedback_prefix_replay_audit_20261027_v1'
PHYSICAL='p1_constructive_plan_physical_20261027_v1'
LIMIT=8*1024**2


def require(ok,message):
    if not ok:raise ValueError(message)
def sha_bytes(data):return hashlib.sha256(data).hexdigest()
def digest(value):return sha_bytes(json.dumps(value,sort_keys=True,allow_nan=False).encode())
def decode(data):
    def pairs(items):
        out={}
        for k,v in items:require(k not in out,'duplicate JSON key');out[k]=v
        return out
    v=json.loads(data,object_pairs_hook=pairs);json.dumps(v,allow_nan=False);return v
def file_bytes(path,limit=LIMIT,expected=None):
    p=Path(path);require(p.is_file() and not p.is_symlink() and p.stat().st_size<=limit,'bounded regular input: '+str(p))
    data=p.read_bytes();require(expected is None or sha_bytes(data)==expected,'external SHA mismatch: '+str(p));return data
def read(path,limit=LIMIT,expected=None):return decode(file_bytes(path,limit,expected))


def f64(value,shape):
    a=np.asarray(value,dtype=np.float64);require(a.shape==shape and np.isfinite(a).all(),'finite numeric shape '+str(shape));return a.tobytes()
def raw(value,shape):
    a=np.asarray(value);require(a.shape==shape and a.dtype.kind in 'fi' and np.isfinite(a).all() and (abs(a)<=1).all() and np.array_equal(a,a.astype(np.float32)),'literal raw float32 '+str(shape));return a.astype(np.float32).tobytes()
def snapshot_bytes(s):
    require(set(s)=={'state','block_velocity','block_angular_velocity','block_angle'},'known snapshot schema')
    return f64(s['state'],(7,))+f64(s['block_velocity'],(2,))+f64(s['block_angular_velocity'],())+f64(s['block_angle'],())
def chain(data):
    require(data.endswith(b'\n'),'whole durable journal lines');events=[decode(x) for x in data.splitlines()];previous=None
    for i,e in enumerate(events):
        require(type(e['index']) is int and e['index']==i and e['previous_sha256']==previous and
            e['event_sha256']==digest({k:v for k,v in e.items() if k!='event_sha256'}),'physical journal hashchain')
        previous=e['event_sha256']
    return events


def load_run(report_path,report_sha):
    path=Path(report_path).absolute();r=read(path,expected=report_sha)
    require(r['schema']==PHYSICAL+'.report' and r['status']=='physical_calibration_recorded' and r['errors']==[],'actual complete physical terminal; no partial prefix admission')
    b=read(path.parent/'execution_binding.json',256*1024)
    p=read(b['prepared'],256*1024,b['prepare_sha256']);require(p['schema']==PHYSICAL+'.prepare','physical prepare schema')
    m=read(p['manifest'],256*1024,p['manifest_sha256']);require(m['schema']==PHYSICAL+'.selection' and m['plans']==p['plans'],'exact preselected physical plans')
    ev=chain(file_bytes(path.parent/'journal.jsonl',expected=r['journal_sha256']))
    require(ev[0]['kind']=='run_started' and ev[0]['prepare']==p and ev[-1]['kind']=='run_finished' and
        ev[-1]['status']==r['status'] and ev[-1]['errors']==[] and ev[-1]['counts']==r['counts'],'run prepare/terminal closure')
    require(r['contract']==p['contract'] and r['sourcepack_sha256']==p['sourcepack_sha256']==m['sourcepack_sha256'],'own source/contract closure')
    require(p['source_slots']==m['roots'] and m['selection_fixed_before_physics'] is True and m['physical_outcome_selection'] is False,'native evaluator roots and preselection declaration')
    require(r['P1_admission'] is False and r['new_neural_calls']==0 and r['new_independent_samples']==0,'no admission/sample inflation')
    def png(receipt):
        name=receipt['png'];require(Path(name).name==name,'flat PNG path')
        return file_bytes(path.parent/name,expected=receipt['file_sha256'])
    return dict(report=r,prepared=p,manifest=m,events=ev,png=png,report_path=str(path),report_sha256=report_sha,
        prepare_sha256=b['prepare_sha256'])


def extract(run):
    r,p,m,events=(run[k] for k in ('report','prepared','manifest','events'))
    require(r['status']=='physical_calibration_recorded' and r['errors']==[],'complete actual physical report, never partial')
    arms=m['arms'];require(isinstance(arms,list) and len(arms)==2 and len(set(arms))==2,'fixed two distinct arms')
    expected=[(i,a) for i in (0,1) for a in arms]
    require(len(r['rows'])==len(p['plans'])==4 and [(x['root_slot'],x['arm']) for x in r['rows']]==expected and
        [(x['root_slot'],x['arm']) for x in p['plans']]==expected,'complete original two roots x two arms order')
    require(len(m['roots'])==2 and all(type(x['root_slot']) is int and x['root_slot']==i for i,x in enumerate(m['roots'])),'literal native root slots')
    result={}
    for index,(row,plan) in enumerate(zip(r['rows'],p['plans'])):
        slot=row['root_slot'];require(type(slot) is int and slot in (0,1) and type(plan['root_slot']) is int,'literal source/root slot')
        key=(slot,row['arm']);need_digest=plan['selection_digest']
        require(need_digest==digest({k:v for k,v in plan.items() if k!='selection_digest'})==row['selection_digest'],'own selection digest')
        require(row['status'] in ('horizon_completed','native_success','native_stopped') and row['initial_exact'] is True and
            row['first_error'] is None and row['secondary_errors']==[] and 5<=row['own_returned']<=10,'complete branch with observed prefix')
        raw(plan['commands'],(10,2))
        be=[e for e in events if e.get('branch')==index]
        require(all(type(e['root_slot']) is int and e['root_slot']==slot and e['arm']==row['arm'] and e['selection_digest']==need_digest for e in be),'all branch receipts own identity')
        started=[e for e in be if e['kind']=='branch_started']
        require(len(started)==1 and started[0]['plan']==plan,'branch_started own plan')
        closed=[e for e in events if e['kind']=='branch_closed' and e['receipt']['root_slot']==slot and e['receipt']['arm']==row['arm']]
        require(len(closed)==1 and closed[0]['receipt']==row,'complete saved branch receipt')
        placement=[e for e in be if e['kind']=='placement_request'];require(len(placement)==1,'one reset placement recipe')
        resets=[{k:e[k] for k in ('kind_of_reset','seed')} for e in be if e['kind']=='operation_attempted' and e['resource']=='reset']
        require(len(resets)==2 and resets[0]['kind_of_reset']=='original_seed' and resets[1]==dict(kind_of_reset='placement',seed=0),'original two-reset recipe')
        frames={};transitions={};dispatches={}
        for phase,steps in (('plan',range(6)),('warmup',(0,5,10))):
            for step in steps:
                obs=[e for e in be if e['kind']=='fresh_observation' and e['receipt']['phase']==phase and e['receipt']['step']==step]
                guards=[e for e in be if e['kind']=='guard_receipt' and e['phase']==phase and e['step']==step]
                require(len(obs)==len(guards)==1,'exact fresh frame and guard '+str((key,phase,step)))
                e,g=obs[0],guards[0];receipt=e['receipt']
                require(e['index']<g['index'] and g['accepted'] is True and g['render_preserved_dynamics'] is True and g['returned_state_equal'] is True,'persisted accepted fresh prefix guard')
                require(f64(receipt['state'],(7,))==f64(receipt['observation_state'],(7,))==f64(receipt['snapshot']['state'],(7,)),'actual/returned/snapshot state')
                data=run['png'](receipt);require(sha_bytes(data)==receipt['file_sha256']==g['file_sha256'] and receipt['png']==g['png'],'saved PNG SHA and guard filename')
                with Image.open(io.BytesIO(data)) as im:
                    require(im.format=='PNG' and im.mode=='RGB' and im.size==(224,224),'RGB224 PNG');rgb=np.asarray(im).copy()
                pixels=int(np.count_nonzero(np.all(rgb==[65,105,225],axis=-1)));state=np.asarray(receipt['state'])
                require(pixels>0 and pixels==g['canonical_agent_pixels'] and ((state[:4]>=0)&(state[:4]<=512)).all() and sha_bytes(rgb.tobytes())==receipt['pixel_sha256'],'independent frame guard and pixel hash')
                frames[phase,step]=dict(receipt=receipt,png_bytes=data)
                if phase=='plan' and step:
                    ts=[e for e in be if e['kind']=='returned_transition' and e['phase']=='plan' and e['step']==step]
                    ats=[e for e in be if e['kind']=='operation_attempted' and e['resource']=='public_step' and e.get('phase')=='plan' and e['step']==step]
                    done=[e for e in be if e['kind']=='operation_completed' and e['resource']=='public_step' and e.get('phase')=='plan' and e['step']==step]
                    require(len(ts)==len(ats)==len(done)==1 and ats[0]['index']<done[0]['index']<ts[0]['index']<obs[0]['index'],'completed own dispatch then return then frame')
                    t,at=ts[0],ats[0];c=raw(plan['commands'][step-1],(2,))
                    require(c==raw(t['command'],(2,))==raw(at['raw_action'],(2,)),'literal planned/attempted/returned prefix command')
                    require(f64(t['state'],(7,))==f64(receipt['state'],(7,)) and not t['terminated'] and not t['truncated'],'exact prefix return, no terminal before suffix')
                    require(f64(t['returned_goal_state'],(7,))==f64(row['goal'],(7,)),'prefix actual goal')
                    transitions[step]={k:t[k] for k in ('state','command','reward','terminated','truncated','metrics','actual_native_success','actual_native_distance','returned_goal_state','returned_goal_rgb_sha256')}
                    dispatches[step]=at['raw_action']
        goal_events=[e for e in be if e['kind']=='goal_attached'];require(len(goal_events)==1,'constructed goal attached receipt')
        goal=goal_events[0]
        require(f64(goal['goal'],(7,))==f64(row['goal'],(7,)) and f64(row['root_state'],(7,))==f64(frames['plan',0]['receipt']['state'],(7,)),'row root/goal exact')
        root_sha=sha_bytes(f64(row['root_state'],(7,))+f64(row['goal'],(7,)))
        require(root_sha==m['roots'][slot]['root_sha256'],'native root plus constructed goal identity')
        require(frames['plan',0]['png_bytes']==frames['warmup',10]['png_bytes'] and snapshot_bytes(frames['plan',0]['receipt']['snapshot'])==snapshot_bytes(frames['warmup',10]['receipt']['snapshot']),'goal attach unchanged target-free root')
        result[key]=dict(row=row,plan=plan,frames=frames,transitions=transitions,dispatches=dispatches,placement=placement[0],resets=resets,
            root_sha256=root_sha,goal_rgb_sha256=goal['goal_rgb_sha256'])
    return result


def compare(old,new):
    """Pure bundle comparison; any mismatch raises, never reports partial match as success."""
    a,b=extract(old),extract(new)
    require(list(a)==list(b),'same fixed source/root/arm mapping')
    for k in ('sourcepack_sha256','source_prepare_id','source_slots'):
        require(old['prepared'][k]==new['prepared'][k],'same source identity '+k)
    for k in ('sourcepack','sourcepack_sha256','source_prepared','source_prepare_sha256','roots','arms'):
        require(old['manifest'][k]==new['manifest'][k],'same evaluator source/recipe '+k)
    summaries=[]
    for key,x in a.items():
        y=b[key];label=str(key)
        require(x['root_sha256']==y['root_sha256'] and x['goal_rgb_sha256']==y['goal_rgb_sha256'],'same root and constructed goalRGB '+label)
        require(f64(x['row']['root_state'],(7,))==f64(y['row']['root_state'],(7,)) and f64(x['row']['goal'],(7,))==f64(y['row']['goal'],(7,)),'root/goal float64 bits '+label)
        placement_keys=('original_goal','requested','placement_seed')
        require(x['resets']==y['resets'] and all(x['placement'][k]==y['placement'][k] for k in placement_keys),'same reset recipe '+label)
        require(raw(x['plan']['commands'][:5],(5,2))==raw(y['plan']['commands'][:5],(5,2)),'prefix planned command bits '+label)
        for phase,step in x['frames']:
            l,r=x['frames'][phase,step],y['frames'][phase,step]
            require(l['png_bytes']==r['png_bytes'],'PNG FILE BYTES mismatch '+str((key,phase,step)))
            require(f64(l['receipt']['state'],(7,))==f64(r['receipt']['state'],(7,)) and snapshot_bytes(l['receipt']['snapshot'])==snapshot_bytes(r['receipt']['snapshot']),'state/snapshot float64 bits '+str((key,phase,step)))
        for step in range(1,6):
            l,r=x['transitions'][step],y['transitions'][step]
            require(raw(l['command'],(2,))==raw(r['command'],(2,))==raw(x['dispatches'][step],(2,))==raw(y['dispatches'][step],(2,)),'returned prefix command bits '+label)
            require(l==r,'prefix returned state/native/goal fields '+str((key,step)))
            require(l['returned_goal_rgb_sha256']==x['goal_rgb_sha256'],'prefix attached goalRGB '+label)
        summaries.append(dict(root_slot=key[0],arm=key[1],root_sha256=x['root_sha256'],prefix_state_steps=list(range(6)),
            own_command_steps=list(range(1,6)),prefix_PNG_byte_pairs=6,warmup_checkpoint_byte_pairs=3,all_exact=True))
    return dict(schema=NAME+'.audit',status='exact_saved_prefix_replay_verified',rows=summaries,old_report_sha256=old['report_sha256'],new_report_sha256=new['report_sha256'],
        distinct_report_inputs=old['report_path']!=new['report_path'] and old['report_sha256']!=new['report_sha256'],
        new_model_calls=0,new_physics=0,P1_admission=False,new_independent_samples=0,
        scope='saved prefix0..5 reproduction only; no suffix outcome comparison, neural provenance audit, physical replay or scientific admission')


def audit(old_report,old_sha,new_report,new_sha):
    old=load_run(old_report,old_sha);new=load_run(new_report,new_sha);result=compare(old,new)
    require(sha_bytes(file_bytes(old_report))==old_sha and sha_bytes(file_bytes(new_report))==new_sha,'final report seals unchanged')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('old-report','old-sha','new-report','new-sha'):p.add_argument('--'+k,required=True)
    p.add_argument('--output');a=p.parse_args();result=audit(a.old_report,a.old_sha,a.new_report,a.new_sha)
    data=(json.dumps(result,sort_keys=True,allow_nan=False)+'\n').encode()
    if a.output:
        path=Path(a.output);require(len(data)<=64*1024 and not path.exists() and not path.is_symlink() and shutil.disk_usage(path.parent).free>=200*1024**2+len(data),'NEW <=64KiB +200MiB reserve')
        require(path.parent.resolve() not in (Path(a.old_report).parent.resolve(),Path(a.new_report).parent.resolve()),'do not write in sealed outputs')
        with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    else:print(data.decode(),end='')


if __name__=='__main__':main()
