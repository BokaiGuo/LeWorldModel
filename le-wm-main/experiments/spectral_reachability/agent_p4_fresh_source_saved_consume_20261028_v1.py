"""Read-only four-slot native sourcepack checker. Never collects any root.

Does not import producer/environment/model or grant freshness. Missing actual
witness-action telemetry remains unknown, even when all structural checks pass.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image

NAME='agent_p4_fresh_source_saved_consume_20261028_v1'
PRODUCER='agent_p4_fresh_source_20261028_v1'
PINS={
    'experiments/spectral_reachability/'+PRODUCER+'.py':'9a94776b5005d349c74c0432d90f71dbf3ec811012bf40a6b5bd9f3217e01335',
    'tests/'+PRODUCER+'_tests.py':'4487f5ac0c9e362f7e1b83419986b23473b691ddfc938a26f797df7390d5c638',
    'innerdocs/'+PRODUCER+'.md':'d0bff239d5ecc28cbae3ad437ead474d1526029650f2618319bb2817ae0ec47f'}
ROOT=Path(__file__).resolve().parents[2]
PHASES=('construction','replay0','replay1')
CAPS=dict(constructor=1,reset=24,set_state=48,goal_render=24,goal_setter=32,goal_bind=8,
          public_step=240,physics_step=2448,explicit_render=252)


def need(ok,message):
    if not ok:raise ValueError(message)


def digest(payload):return hashlib.sha256(payload).hexdigest()
def semantic(value):return digest(json.dumps(value,sort_keys=True,allow_nan=False).encode())


def decode(payload):
    def pairs(items):
        out={}
        for k,v in items:need(k not in out,'duplicate JSON key');out[k]=v
        return out
    out=json.loads(payload,object_pairs_hook=pairs)
    semantic(out);return out


class Reader:
    def __init__(self,provider=None):self.provider=provider;self.inputs={};self.total=0
    def read(self,path,expected=None,cap=16*1024**2):
        p=Path(path)
        need(p.is_absolute() and '..' not in p.parts,'literal absolute input path')
        if self.provider is None:
            need(p.resolve()==p and p.is_file() and p.stat().st_size<=cap,'regular canonical bounded input')
            payload=p.read_bytes()
        else:payload=self.provider(str(p))
        need(type(payload) is bytes and len(payload)<=cap,'bounded bytes')
        h=digest(payload);need(expected is None or h==expected,'input SHA: '+str(p))
        need(str(p) not in self.inputs or self.inputs[str(p)]==h,'input changed while consuming')
        if str(p) not in self.inputs:self.total+=len(payload)
        need(self.total<=128*1024**2,'128MiB aggregate read cap')
        self.inputs[str(p)]=h;return payload
    def json(self,path,expected=None):return decode(self.read(path,expected))


def flat(folder,name):
    need(type(name) is str and Path(name).name==name and name not in ('','.','..'),'flat source artifact')
    return folder/name


def values(value,shape,dtype=np.float64):
    a=np.asarray(value,dtype=dtype)
    need(a.shape==shape and np.isfinite(a).all(),'finite exact array shape');return a


def exact(a,b,label):
    a,b=np.asarray(a),np.asarray(b)
    need(a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),label)


def snapshot(s,state):
    need(s['state']==state,'snapshot/observation state')
    v=values(state,(7,)).astype('<f8')
    need(s['state_float64_bits']==v.tobytes().hex(),'state7 raw float64 bits')
    for body in ('agent','block'):
        b=s['bodies'][body];x=np.frombuffer(bytes.fromhex(b['float64_bits']),dtype='<f8')
        expected=values([*b['position'],*b['velocity'],b['unwrapped_angle'],b['angular_velocity'],*b['force'],b['torque']],(9,)).astype('<f8')
        exact(x,expected,'full exported body bits')
        need(type(b['is_sleeping']) is bool,'exported sleeping flag')
    agent=s['bodies']['agent'];block=s['bodies']['block']
    exact(v[:4],values([*agent['position'],*block['position']],(4,)),'actual physical positions')
    exact(v[5:],values(agent['velocity'],(2,)),'actual agent velocity')
    need(v[4]==block['unwrapped_angle']%(2*np.pi),'native wrapped block observation')


def identity(obs,history,warm):
    state=values(obs['state'],(7,)).astype('<f8');pose=state[:5].copy()
    pose[4]=np.remainder(pose[4],2*np.pi);pose[pose==0]=0.
    return dict(root_state=state.tolist(),physical_pose=pose.tolist(),physical_pose_key=digest(pose.tobytes()),
                root_state_sha256=digest(state.tobytes()),root_snapshot_sha256=semantic(obs['snapshot']),
                history_pixels_sha256=[x['pixel_sha256'] for x in history],raw_history_sha256=digest(warm.tobytes()))


def duplicate(a,b):
    state=values(b['root_state'],(7,));delta=np.asarray(a['physical_pose'])[:4]-state[:4]
    angle=a['physical_pose'][4]-np.remainder(state[4],2*np.pi)
    return bool((np.abs(delta)<=1e-9).all() and abs(np.arctan2(np.sin(angle),np.cos(angle)))<=1e-12) or any(
        a[k]==b.get(k) for k in ('root_state_sha256','root_snapshot_sha256','history_pixels_sha256'))


def witness(state,sign):
    s=values(state,(7,));v=s[2:4]-s[:2];need(np.linalg.norm(v)>0,'nonzero radial recipe')
    radial=v/np.linalg.norm(v);tangent=np.asarray([-radial[1],radial[0]])
    logits=np.concatenate((np.tile(.5*radial,(5,1)),np.tile(.25*radial+sign*.5*tangent,(5,1)))).astype(np.float32)
    return logits,np.tanh(logits)


def npz(reader,path,expected,names):
    payload=reader.read(path,expected,2*1024**2)
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        need(len(z.infolist())==len(names) and {x.filename for x in z.infolist()}=={n+'.npy' for n in names}
             and sum(x.file_size for x in z.infolist())<=4*1024**2,'exact bounded NPZ members')
    with np.load(io.BytesIO(payload),allow_pickle=False) as z:out={k:z[k].copy() for k in z.files}
    need(set(out)==set(names) and all(a.dtype.kind in 'buif' and np.isfinite(a).all() for a in out.values()),'native numerical arrays only')
    return out


def journal(payload,pack,p):
    need(payload.endswith(b'\n'),'complete terminal journal line')
    events=[decode(x) for x in payload.splitlines()];previous=None;pending={};last=-1;stack=[]
    counts={k:dict(attempted=0,completed=0) for k in CAPS};scope={};operations={}
    for index,event in enumerate(events):
        body={k:v for k,v in event.items() if k!='event_sha256'}
        need(event['index']==index and event['previous_sha256']==previous and event['event_sha256']==semantic(body),'full event chain')
        previous=event['event_sha256']
        if event['kind'] not in ('operation_attempted','operation_completed'):continue
        e={k:event[k] for k in ('operation_id','episode_id','resource','units','region')}
        ident=e['operation_id'];resource=e['resource'];units=e['units']
        need(type(ident) is int and type(units) is int and units==1 and resource in CAPS,'native operation identity')
        scope.setdefault(e['episode_id'],{k:dict(attempted=0,completed=0) for k in CAPS})
        if event['kind']=='operation_attempted':
            need(ident>last and ident not in pending,'monotonic once-only attempt');last=ident
            pending[ident]=e;operations[ident]=dict(entry=e,index=index,parent=stack[-1] if stack else None)
            stack.append(ident)
            counts[resource]['attempted']+=units;scope[e['episode_id']][resource]['attempted']+=units
            need(counts[resource]['attempted']<=CAPS[resource],'predispatch bounded resource denominator')
        else:
            need(pending.pop(ident,None)==e,'actual matching operation return')
            need(stack and stack.pop()==ident,'properly nested actual native operation returns')
            operations[ident]['completed_index']=index
            counts[resource]['completed']+=units;scope[e['episode_id']][resource]['completed']+=units
    need(events[0]['kind']=='run_started' and events[0]['prepare_id']==semantic(p) and events[0]['mode']==p['mode'],'actual run prepare identity')
    need(events[-1]['kind']=='run_finished' and previous==pack['journal_chain_tail'] and
         events[-1]['rows']==pack['rows'] and events[-1]['status']==pack['status'] and
         events[-1]['counts']==pack['counts'] and counts==pack['counts'] and
         {str(k):v for k,v in pending.items()}=={str(k):v for k,v in pack['pending'].items()},'terminal rows/actual census/pending')
    return events,counts,scope,operations


def operation_tree(operations):
    for ident,operation in operations.items():
        need('completed_index' in operation,'no pending native operation')
        children=[o['entry']['resource'] for o in operations.values() if o['parent']==ident]
        resource=operation['entry']['resource']
        want={'constructor':[], 'reset':['set_state','goal_setter','goal_render','set_state'],
              'set_state':['physics_step'], 'goal_bind':['goal_setter'],
              'public_step':['physics_step']*10}.get(resource,[])
        need(children==want,'native actual operation tree, not just aggregate hashes: '+resource)


def one(events,kind,**fields):
    out=[e for e in events if e['kind']==kind and all(e.get(k)==v for k,v in fields.items())]
    need(len(out)==1,'unique '+kind+' '+str(fields));return out[0]


def consume(output,pack_sha,prepared,prepare_sha,*,reader=None):
    reader=Reader() if reader is None else reader;output=Path(output).absolute();prepared=Path(prepared).absolute()
    p=reader.json(prepared,prepare_sha);pack=reader.json(output/'sourcepack.json',pack_sha)
    need(p['schema']==PRODUCER+'.prepare' and pack['schema']==PRODUCER+'.sourcepack' and
         pack['prepare_id']==semantic(p),'sourcepack/prepare schemas')
    for rel,h in PINS.items():need(p['bindings'][str(ROOT/rel)]==h,'frozen producer three-file pins')
    for path,h in p['bindings'].items():reader.read(path,h)
    expected=[dict(index=i,root_seed=203027101+i,reset_seed=203027201+i,sign=1 if i%2==0 else -1) for i in range(4)]
    need(p['slots']==expected and p['caps']==CAPS and p['fixed_denominator']==pack['fixed_denominator']==4,
         'four fixed source denominator/caps, not four successes')
    need(pack['mode']==p['mode'] and pack['mode'] in ('calibration_only','fresh_candidate') and
         pack['confirmed'] is pack['P1_admission'] is pack['P4_admission'] is False and
         pack['neural_calls']==0 and pack['training'] is False,'no confirmatory/neural release')
    if p['mode']=='calibration_only':need(pack['freshness'] is False and all(r['freshness'] is False for r in pack['rows']),'calibration is never fresh')
    if p['mode']=='fresh_candidate':need(p['history_certificate'] is not None,'no certified-mode prepare without historical certificate')
    need(reader.json(output/'execution_binding.json')==dict(prepared=str(prepared),prepare_sha256=prepare_sha),'external preparation execution binding')
    need(len(pack['rows'])==4 and [r['slot'] for r in pack['rows']]==expected and
         all([x['phase'] for x in r['phases']]==list(PHASES) for r in pack['rows']),'all four slots/twelve phase rows retained')
    events,counts,scope,operations=journal(reader.read(output/'journal.jsonl',pack['journal_sha256']),pack,p)
    warm=np.asarray([[.1,0]]*5+[[0,.1]]*5,np.float32)
    exact(np.asarray(p['raw_warmup'],np.float32),warm,'original raw float32 warmup10')
    observations={};pixels={};guards={}
    for e in events:
        if e['kind'] in ('fresh_guard_passed','fresh_guard_rejected'):
            key=(e['slot'],e['phase'],e['step']);need(key not in guards,'unique native RGB guard');guards[key]=e
        if e['kind']!='fresh_observation':continue
        key=(e['slot'],e['phase'],e['step']);need(key not in observations,'unique actual frame')
        need(e['phase'] in PHASES and type(e['slot']) is int and e['slot'] in range(4) and
             type(e['step']) is int and 0<=e['step']<=20,'actual observation identity')
        need(e['png']==f's{e["slot"]}_{e["phase"]}_{e["step"]:02}.png','own real observation namespace')
        payload=reader.read(flat(output,e['png']),e['file_sha256'],1024**2)
        with Image.open(io.BytesIO(payload)) as im:
            need(im.mode=='RGB' and im.size==(224,224),'native RGB224');rgb=np.asarray(im).copy()
        need(digest(rgb.tobytes())==e['pixel_sha256'],'full real PNG pixel SHA');snapshot(e['snapshot'],e['state'])
        observations[key]=e;pixels[key]=rgb
    for key,g in guards.items():
        o=observations[key];rgb=pixels[key];state=values(o['state'],(7,))
        color=int(np.count_nonzero(np.all(rgb==[65,105,225],axis=-1)))
        valid=color>0 and bool(((state[:4]>=0)&(state[:4]<=512)).all())
        need(o['index']<g['index'] and (g['kind']=='fresh_guard_passed')==valid,'guard independently follows actual saved observation')
        if valid:need(g['canonical_agent_pixels']==color,'actual canonical agent pixel count')
    clean=pack['sourcepack_complete'] is True and pack['status']=='fixed_slots_processed' and not pack['errors'] and not pack['pending']
    # Preserve partial accounting and all raw observations; do not infer missing
    # native returns/state/phase from an attempted operation or a good subset.
    if not clean:
        return dict(schema=NAME+'.receipt',status='partial_preserved_NOT_complete',fixed_denominator=4,
                    phase_denominator=12,rows=pack['rows'],counts=counts,inputs=reader.inputs,
                    verified_observations=len(observations),freshness=False,confirmed=False,physical_release=False)
    need(all(c['attempted']==c['completed'] for c in counts.values()),'no pending completed batch')
    operation_tree(operations)
    need(set(scope)<=({'source_constructor'}|{f's{i}_{ph}' for i in range(4) for ph in PHASES}), 'no unassigned native operation scope')
    need(scope['source_constructor']=={k:dict(attempted=int(k=='constructor'),completed=int(k=='constructor')) for k in CAPS},
         'constructor cannot hide unallocated source operations')
    need(counts['constructor']['completed']==1 and counts['physics_step']['completed']==2*counts['reset']['completed']+10*counts['public_step']['completed'] and
         counts['set_state']['completed']==2*counts['reset']['completed'] and counts['goal_render']['completed']==counts['reset']['completed'] and
         counts['goal_setter']['completed']==counts['reset']['completed']+counts['goal_bind']['completed'] and
         counts['explicit_render']['completed']==len(observations),'full native census')
    batch=[];audited=[];native_returns=0
    for slot,row in enumerate(pack['rows']):
        need(row['confirmed'] is False,'no per-row confirmed qualification')
        need(one(events,'slot_finished',receipt=row)['receipt']==row,'durable slot finish')
        construction_root=None;construction_goal=None;reference=None
        for ph in row['phases']:
            phase=ph['phase'];eid=f's{slot}_{phase}';phaseobs=sorted((o for k,o in observations.items() if k[:2]==(slot,phase)),key=lambda x:x['step'])
            ledger=scope.get(eid,{k:dict(attempted=0,completed=0) for k in CAPS})
            if ph['status']=='not_started':
                need(not phaseobs and not any(c['attempted'] for c in ledger.values()),'unstarted phase is not silently executed');continue
            need(ph['status'] in ('returned','rejected'),'terminal phase status, no suppressed failure')
            attempt=one(events,'phase_attempt',slot=slot,phase=phase)
            need(ledger['reset']['completed']==2 and ledger['set_state']['completed']==4 and ledger['goal_render']['completed']==2,
                 'each actual source phase has original two full native resets')
            placement=one(events,'placement_recipe',phase=phase,slot=expected[slot])
            values(placement['requested'],(7,));original_goal=values(placement['original_goal'],(7,))
            need(attempt['index']<placement['index'],'placement belongs to started phase')
            resets=[o for o in operations.values() if o['entry']['episode_id']==eid and o['entry']['resource']=='reset']
            need(resets[0]['completed_index']<placement['index']<resets[1]['index'],'native seed reset then placement recipe then placement reset')
            need([o['step'] for o in phaseobs]==list(range(len(phaseobs))),'no dropped/reordered source frames')
            need(ledger['public_step']['completed']==max(0,len(phaseobs)-1),'all actual returned steps have saved state/RGB')
            wc=reader.json(output/f's{slot}_{phase}_warmup_commit.json')
            exact(np.asarray(wc['raw_actions'],np.float32),warm,'committed original raw10')
            warms=[e for e in events if e['kind']=='warmup_return' and e.get('slot')==slot and e['phase']==phase]
            need([e['step'] for e in warms]==list(range(1,len(warms)+1)) and len(warms)<=10,'all actual warmup returns')
            for j,e in enumerate(warms):
                exact(np.asarray(e['raw_action'],np.float32),warm[j],'actual saved warmup raw32')
                need(e['raw_float32_bits']==warm[j].tobytes().hex(),'actual warmup bit witness')
                state=values(observations[slot,phase,j+1]['state'],(7,));delta=original_goal-state;angle=abs(delta[4])
                success=bool(np.linalg.norm(delta[:4])<20 and min(angle,2*np.pi-angle)<np.pi/9)
                need(e['terminated'] is success and e['truncated'] is False,'actual native warmup termination flags')
                need(not any(t['terminated'] or t['truncated'] for t in warms[:j]),'no warmup after native stop')
            terminal=[e for e in events if e['kind']=='native_return' and e.get('slot')==slot and e['phase']==phase]
            need([e['step'] for e in terminal]==list(range(1,len(terminal)+1)) and len(terminal)<=10,'all chronological witness returns')
            if len(warms)==10 and (slot,phase,10) in guards and guards[slot,phase,10]['kind']=='fresh_guard_passed':
                history=[observations[slot,phase,b] for b in (0,5,10)];root=identity(history[-1],history,warm)
                event=one(events,'actual_root_pre_witness_check' if phase=='construction' else 'replay_root_pre_witness_check',slot=slot,phase=phase)
                need(event['identity']==root and guards[slot,phase,10]['index']<event['index'],'actual root checked before any witness')
                witness_attempts=[v['index'] for v in operations.values() if v['entry']['episode_id']==eid and v['entry']['resource']=='public_step' and v['entry']['region']=='suffix']
                need(not witness_attempts or event['index']<min(witness_attempts),'no witness dispatch before actual-root exclusion')
                if phase=='construction':
                    collision=any(duplicate(root,h) for h in p['historical_roots']+batch);batch.append(root)
                    need(event['duplicate'] is collision and row['actual_root_identity']==root,'independent historical/batch pose exclusion')
                    construction_root=root
                    if collision:
                        need(not terminal and not witness_attempts and ph['status']=='rejected' and row['reachable'] is False,'duplicate never dispatches witness/publishes blind');continue
                else:need(root==construction_root,'actual full root/history matches own construction')
                commit=reader.json(output/f's{slot}_{phase}_commit.json');logits,commands=witness(root['root_state'],expected[slot]['sign'])
                exact(np.asarray(commit['logits'],np.float32),logits,'original geometric witness logits')
                exact(np.asarray(commit['commands'],np.float32),commands,'original raw witness tanh, no altered amplitude')
                need(commit['root_snapshot']==history[-1]['snapshot'],'commit actual root snapshot')
                if phase!='construction' and construction_goal is not None:
                    for b in range(min(11,len(phaseobs))):
                        o=observations[slot,phase,b];ref=observations[slot,'construction',b]
                        need(o['snapshot']==ref['snapshot'] and o['pixel_sha256']==ref['pixel_sha256'],'all replay warmup actual state/RGB equal construction')
                goal=original_goal if phase=='construction' else construction_goal
                need(goal is not None,'replay cannot invent missing construction goal')
                for j,e in enumerate(terminal):
                    o=observations[slot,phase,11+j];state=values(o['state'],(7,));delta=values(goal,(7,))-state
                    angle=abs(delta[4]);success=bool(np.linalg.norm(delta[:4])<20 and min(angle,2*np.pi-angle)<np.pi/9)
                    need(e['state']==o['state'] and e['reward']==-float(np.linalg.norm(delta)) and
                         e['terminated'] is success and e['truncated'] is False,'actual native7 reward/success, not object-only')
                    need(not any(t['terminated'] or t['truncated'] for t in terminal[:j]),'no returns after native stop')
                    if phase!='construction':
                        ref=observations[slot,'construction',11+j]
                        need(o['snapshot']==ref['snapshot'] and o['pixel_sha256']==ref['pixel_sha256'],'all actual replay future state/RGB equal construction')
                    native_returns+=1
                if phase=='construction' and ph['status']=='returned':
                    need(len(terminal)==10,'derived goal is original full construction endpoint')
                    construction_goal=values(terminal[-1]['state'],(7,));reference=commands
                if ph['status']=='returned':
                    need(ph['public_steps']==10+len(terminal) and ph['counts_delta']==ledger and ledger['public_step']['completed']==10+len(terminal),'phase actual returns match ledger, not planned budget')
                    pub=10+len(terminal);replay=int(phase!='construction')
                    expected_counts=dict(constructor=0,reset=2,set_state=4,goal_render=2,goal_setter=2+replay,
                                         goal_bind=replay,public_step=pub,physics_step=4+10*pub,explicit_render=pub+1)
                    need(ledger=={k:dict(attempted=v,completed=v) for k,v in expected_counts.items()},'independent per-phase complete native budget')
                    one(events,'phase_return',slot=slot,receipt=ph)
                    if phase!='construction':need(terminal and terminal[-1]['terminated'] is True and ph['native_success'] is True,'actual native replay success')
        if row['reachable']:
            need(row['status']=='reachable_native_replayed' and all(ph['status']=='returned' for ph in row['phases']),'reachable has all three real phases')
            need(row['replay_native_success']==[True,True],'two actual successful native replays')
            cert_event=one(events,'construction_certificate_saved',file=f's{slot}_construction_certificate.npz')
            need(row['certificate']==cert_event['file'] and row['certificate_sha256']==cert_event['sha256'],'certificate identity')
            cert=npz(reader,flat(output,row['certificate']),row['certificate_sha256'],
                     'goal goal_rgb witness_commands witness_logits future_states observed_states root_history raw_action_history'.split())
            for key,value in (('goal',construction_goal),('witness_commands',reference),('witness_logits',witness(construction_root['root_state'],expected[slot]['sign'])[0]),
                              ('future_states',np.asarray([observations[slot,'construction',b]['state'] for b in range(11,21)])),
                              ('observed_states',np.asarray([observations[slot,'construction',b]['state'] for b in (0,5,10)])),
                              ('root_history',np.stack([pixels[slot,'construction',b] for b in (0,5,10)])),
                              ('raw_action_history',warm),('goal_rgb',pixels[slot,'construction',20])):
                exact(cert[key],value,'full construction certificate array '+key)
            blind=npz(reader,flat(output,row['blind_payload']),row['blind_payload_sha256'],'rgb_history raw_action_history goal_rgb'.split())
            exact(blind['rgb_history'],np.stack([pixels[slot,'replay0',b] for b in (0,5,10)]),'blind is actual replay0 RGB3')
            exact(blind['raw_action_history'],warm,'blind actual float32 raw10');exact(blind['goal_rgb'],cert['goal_rgb'],'blind actual derived goalRGB')
            need(row['root_sha256']==digest(np.concatenate((values(construction_root['root_state'],(7,)),construction_goal)).tobytes()),'original root-goal identity')
            need(row['root_sha256'] not in p['root_goal_exclusions'],'historical root-goal exclusion')
            need(row['executed_replay_steps']==[len([e for e in events if e['kind']=='native_return' and e.get('slot')==slot and e['phase']==ph]) for ph in ('replay0','replay1')],'actual early stop lengths preserved')
        else:need(row['status']=='rejected_fixed_candidate' and row['blind_payload'] is None,'rejected denominator cannot publish blind')
        audited.append(dict(slot=slot,status=row['status'],reachable=row['reachable'],phases=row['phases']))
    need(pack['reachable_count']==sum(r['reachable'] for r in pack['rows']),'full four-slot reachable denominator')
    active=sum(ph['status']!='not_started' for r in pack['rows'] for ph in r['phases'])
    need(counts['reset']['completed']==2*active and counts['public_step']['completed']==len(observations)-active,
         'all and only actual source-phase public steps/frames; no hidden reset/dispatch')
    # Frozen producer source asserts equality of witness latest_action, but only
    # warmup writes its raw bits. A commit/certificate is NOT that missing trace.
    return dict(schema=NAME+'.receipt',status='saved_structure_verified_STRICT_BLOCKED_witness_raw_telemetry',
                fixed_denominator=4,phase_denominator=12,reachable_count=pack['reachable_count'],rows=audited,
                verified_observations=len(observations),verified_native_returns=native_returns,counts=counts,
                actual_witness_raw_per_step=None,independent_goal_pose_trace=None,strict_complete=False,
                historical_certificate='unknown' if p['history_certificate'] is None else 'SHA_bound_MAIN_input_not_reapproved',
                inputs=reader.inputs,freshness=False,confirmed=False,P1_admission=False,P4_admission=False,physical_release=False,
                neural_calls=0,physics_calls=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('output','sourcepack-sha','prepared','prepare-sha'):parser.add_argument('--'+key,required=True)
    a=parser.parse_args();result=consume(a.output,a.sourcepack_sha,a.prepared,a.prepare_sha)
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    # This v1 producer cannot satisfy the independent witness-action requirement.
    # Emit the measured structural result, but do not return a false strict pass.
    if not result.get('strict_complete',False):raise SystemExit(2)


if __name__=='__main__':main()
