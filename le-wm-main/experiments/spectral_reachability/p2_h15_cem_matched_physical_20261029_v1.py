"""Two known roots, exact matched CEM HOST winners; physical-only run."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np

from experiments.spectral_reachability import agent_p2_intersection_physical_20261029_v2 as physical
from experiments.spectral_reachability import p2_h15_cem_matched_host_20261029_v1 as host

NAME='p2_h15_cem_matched_physical_20261029_v1'
AUTH='MAIN_P2_H15_MATCHED_CEM_TWO_ROOT_PHYSICS_ONLY'
ROOT=physical.ROOT
HOST_DIR=ROOT/'results/p2_h15_cem_matched_host_20261029_v1_cuda_MAIN01'
HOST_PREP=ROOT/'results/p2_h15_cem_matched_host_20261029_v1_prepare_CODEX01.json'
HOST_PREP_SHA='7fa4d83320ecad6391126dc2906f8156965af98f366aa4b0f0e61715af4ed1cb'
PHYS_PREP=physical.prep.ROOT/'results/agent_p2_intersection_physical_20261029_v2_prepare_MAIN01.json'
PHYS_SHA='619a3202a9e14c6d164375a46ee8e51e7e9fe217678e6fb6f3c2725e5219341c'
need=physical.require


def contract():
    c=deepcopy(physical.contract())
    c.update(schema=NAME+'.contract',fixed_denominator=2,order=[[0,'off','cem_matched'],[1,'off','cem_matched']],
             arms=['cem_matched'],arm='cem_matched_saved_winner',
             caps=dict(constructor=2,close=2,reset=4,public_step=50,physics_step=508,set_state=8,
                       goal_setter=6,goal_render=4,goal_bind=2,explicit_render=162),
             raw_authority='independently consumed matched CEM HOST winner raw15 float32, no physical selection',
             interpretation='two known off roots, separate retrospective physical run, not fresh or matched total compute')
    return c


def make_plan(root,task,native_plan,mapped_root,physical_root,slot,saved):
    need(task['binding']['task_id']==native_plan['task_id'] and task['binding']['task_sha256']==native_plan['task_sha256'] and
         task['root_context']==native_plan['root_context'] and task['goal']==native_plan['goal'] and
         mapped_root['physical_root_sha256']==physical_root['root_sha256'],'exact task/root/goal')
    need(slot['root']==root and slot['status']=='complete','HOST slot')
    winner=saved['result']['winner'];commands=deepcopy(winner['actions']);arr=physical.raw15(commands)
    need(type(winner['index']) is int and 0<=winner['index']<42,'fixed delivered winner')
    plan=dict(schema=NAME+'.plan',root_slot=root,condition='off',arm='cem_matched',task_id=native_plan['task_id'],
              task_sha256=native_plan['task_sha256'],root_sha256=physical_root['root_sha256'],
              host_slot_report_sha256=slot['report_sha256'],winner=winner['index'],commands=commands,
              raw_float32_bits=arr.tobytes().hex(),raw_steps=15)
    plan['selection_digest']=physical.digest(plan);return plan


def build(host_sha):
    need(type(host_sha) is str and len(host_sha)==64 and all(c in '0123456789abcdef' for c in host_sha),'external host report SHA')
    consumed=host.consume(HOST_DIR,host_sha,HOST_PREP,HOST_PREP_SHA)
    need(consumed['status']=='saved_native_checked_not_admitted' and len(consumed['resources'])==2,'strict HOST consume')
    physical_p=physical.read(PHYS_PREP,physical.META)
    need(physical.sha(PHYS_PREP)==PHYS_SHA and physical_p['contract']==physical.contract(),'six-slot source authority')
    for filename,digest in physical_p['bindings'].items():
        need(physical.sha(filename)==digest,'historical physical source binding')
    cases=physical.source_material()
    roots=physical.native.allmode.root_metadata(cases)
    need(roots==physical_p['source_roots'] and len(roots)==2,'original native roots')
    mapped=host.bridge.build_bridge();report=physical.read(HOST_DIR/'report.json',physical.META)
    plans=[]
    for root in (0,1):
        slot=report['slots'][root];task=mapped['roots'][root]['task']
        saved=physical.read(HOST_DIR/('root'+str(root))/'report.json',physical.META)
        need(slot['root']==root and slot['status']=='complete' and physical.sha(HOST_DIR/('root'+str(root))/'report.json')==slot['report_sha256'], 'HOST root report')
        native_plan=next(x for x in physical_p['plans'] if x['root_slot']==root)
        plans.append(make_plan(root,task,native_plan,mapped['roots'][root],roots[root],slot,saved))
    return dict(schema=NAME+'.prepare',contract=contract(),host_report_sha256=host_sha,host_prepare_sha256=HOST_PREP_SHA,
                historical_physical_prepare_sha256=PHYS_SHA,plans=plans,source_roots=roots,
                source_slots=deepcopy(physical_p['source_slots']),warm_traces=deepcopy(physical_p['warm_traces']),
                physical_release=False,scientific_admission=False,selection_uses_physics=False,
                bindings={str(Path(__file__).resolve()):physical.sha(__file__),str(Path(physical.__file__).resolve()):physical.sha(physical.__file__)})


def validate(value, rebuilt):
    need(value==rebuilt and value['contract']==contract() and len(value['plans'])==2,'full prepare rebuild')
    return True


def prepare(path,host_sha):
    p=Path(path).absolute();need(p.name.startswith(NAME+'_') and not p.is_symlink(),'own output')
    v=build(host_sha);b=physical.raw(v);need(len(b)<=physical.META and shutil.disk_usage(p.parent).free>=physical.FLOOR+len(b),'prepare size/floor')
    with p.open('xb') as f:need(f.write(b)==len(b),'complete write');f.flush();os.fsync(f.fileno())
    physical.paired.fsync_dir(p.parent);return physical.sha(p)


def verify(path,digest):
    p=physical.read(path,physical.META);need(physical.sha(path)==digest,'external prepare SHA');validate(p,build(p['host_report_sha256']));return p


class Store(physical.native.allmode.Store):
    def __init__(self,path):
        self.path=Path(physical.native.absolute(str(path)))
        need(self.path.name.startswith(NAME+'_') and not self.path.exists() and not self.path.is_symlink() and
             shutil.disk_usage(self.path.parent).free>=physical.FLOOR+physical.CAP,'exclusive output and disk')
        self.path.mkdir();physical.paired.fsync_dir(self.path.parent);self.previous=None;self.index=0
    def check(self,size=4096,final=False):
        files=list(self.path.iterdir());need(all(f.is_file() and not f.is_symlink() for f in files) and
            sum(f.stat().st_size for f in files)+size<=physical.CAP-(0 if final else physical.FINAL) and
            shutil.disk_usage(self.path).free>=physical.FLOOR+size,'output cap/reserve/floor')


class Meter(physical.Meter):
    def __init__(self,store,started):
        super().__init__(store,started)
        self.counts={k:dict(attempted=0,completed=0) for k in contract()['caps']};self.active={k:0 for k in self.counts}
    def call(self,kind,fn,*,cleanup=False,**fields):
        need(kind in self.counts and self.counts[kind]['attempted']<contract()['caps'][kind],'two-root resource cap')
        return super().call(kind,fn,cleanup=cleanup,**fields)


def rows_for(p):
    rows=[]
    for i,plan in enumerate(p['plans']):
        rows.append(dict(index=i,root_slot=i,condition='off',arm='cem_matched',task_id=plan['task_id'],
          selection_digest=plan['selection_digest'],plan=deepcopy(plan),status='not_started',first_error=None,secondary_errors=[],
          warmup_attempted=0,warmup_returned=0,own_attempted=0,own_returned=0,PNG=0,initial_exact=False,
          trajectory=[],checkpoints=[],missing_checkpoints=list(physical.CP),last_observation=None,responses=[],
          final_state=None,final_metrics=None,native_success=None,terminated=None,truncated=None,
          prefix51=dict(complete=False,frames=[]),source_unpaired_on_steps=[],evidence_origin='new_native_execution_known_root'))
    return rows


def run(prepared,expected,output,factory,*,authorization):
    need(authorization==AUTH,'explicit physical authorization before callbacks')
    started=time.monotonic();p=verify(prepared,expected);data=physical.source_material()
    need(physical.native.allmode.root_metadata(data)==p['source_roots'] and
         [x['slot'] for x in data['roots']]==p['source_slots'] and [x['warm_trace'] for x in data['roots']]==p['warm_traces'],'physical source binding')
    store=Store(output);meter=Meter(store,started);rows=rows_for(p);errors=[];seen=[]
    binding=dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected,host_report_sha256=p['host_report_sha256'])
    try:
        store.put('execution_binding.json',binding);store.emit('run_started',prepare_sha256=expected,contract=contract())
        for row,plan in zip(rows,p['plans'],strict=True):
            meter.where=dict(branch=row['index'],**{k:row[k] for k in ('root_slot','condition','arm','selection_digest')})
            env=None;row['status']='running'
            try:
                env=meter.call('constructor',factory)
                need(all(env is not x and env.unwrapped is not x.unwrapped for x in seen),'fresh native env');seen.append(env)
                physical.validate_env(env.unwrapped)
                need(env.unwrapped.render_action is False and env.unwrapped.render_size==224 and env.unwrapped.render_mode=='rgb_array','native renderer')
                with physical.native_hooks(env,meter,row['secondary_errors']):
                    physical.episode(env,data['roots'][plan['root_slot']],data['source_plan'],plan,row,meter,store)
                need(not row['secondary_errors'],'hooks restored')
            except physical.native.allmode.NativeStop as exc:row.update(status='native_stopped',stop_reason=str(exc))
            except BaseException as exc:
                row['status']='engineering_failed';row['first_error']=physical.native.base.error('branch',exc);errors.append(row['first_error'])
            finally:
                if env is not None:
                    try:meter.call('close',env.close,cleanup=True)
                    except BaseException as exc:
                        row['status']='engineering_failed';errors.append(physical.native.base.error('close',exc))
                for phase,label in (('warmup','warmup'),('plan','own')):
                    ops=[x for x in meter.receipts if x['resource']=='public_step' and x.get('branch')==row['index'] and x['phase']==phase]
                    row[label+'_attempted']=len(ops);row[label+'_returned']=sum(x['status']=='returned' for x in ops)
            store.emit('branch_closed',receipt=row)
            if errors:break
        verify(prepared,expected)
    except BaseException as exc:errors.append(physical.native.base.error('execution',exc))
    errors.extend(meter.secondary)
    good=not errors and all(x['status'] in ('horizon_completed','native_stopped') for x in rows) and time.monotonic()-started<physical.WALL
    status='two_root_native_recorded_not_admitted' if good else 'partial_or_failed'
    store.emit('run_finished',status=status,counts=meter.counts,errors=errors)
    report=dict(schema=NAME+'.report',contract=contract(),status=status,execution_binding=binding,rows=rows,
      errors=errors,counts=deepcopy(meter.counts),actual_operation_receipts=deepcopy(meter.receipts),fixed_denominator=2,
      journal_chain_tail=store.previous,journal_sha256=physical.sha(store.path/'journal.jsonl'),
      execution_binding_sha256=physical.sha(store.path/'execution_binding.json'),wall_seconds=time.monotonic()-started,
      physical_release=False,scientific_admission=False,training=False,new_neural_calls=0)
    store.put('report.json',report,final=True);return report


def consume(output,report_sha,prepared,prepare_sha):
    """Audit saved transcript and PNG bytes; not an independent physics rerun."""
    import hashlib
    import io
    from PIL import Image
    path=Path(output);need(path.is_dir() and not path.is_symlink(),'output directory')
    p=verify(prepared,prepare_sha);files=physical.saved_artifacts(path)
    need(hashlib.sha256(files['report.json']).hexdigest()==report_sha,'external report SHA')
    r=physical.native.decode(files['report.json'])
    need(r['schema']==NAME+'.report' and r['contract']==contract() and r['fixed_denominator']==2 and
         r['physical_release'] is False and r['scientific_admission'] is False and r['training'] is False and
         r['new_neural_calls']==0 and r['status']=='two_root_native_recorded_not_admitted' and not r['errors'], 'complete two-root report')
    b=r['execution_binding'];need(b==dict(prepared=str(Path(prepared).absolute()),prepare_sha256=prepare_sha,host_report_sha256=p['host_report_sha256']) and
        hashlib.sha256(files['execution_binding.json']).hexdigest()==r['execution_binding_sha256'] and
        physical.native.decode(files['execution_binding.json'])==b,'binding')
    data=files['journal.jsonl'];need(data.endswith(b'\n') and hashlib.sha256(data).hexdigest()==r['journal_sha256'],'journal')
    events=[physical.native.decode(line) for line in data.splitlines()]
    need(events and events[0]['kind']=='run_started' and events[0]['contract']==contract() and
         events[0]['prepare_sha256']==prepare_sha and events[-1]['kind']=='run_finished' and
         all(events[-1][k]==r[k] for k in ('status','counts','errors')),'lifecycle bookends')
    previous=None;stack=[];intents={};terminals={};closed=0;names={'journal.jsonl','report.json','execution_binding.json'}
    allowed={'run_started','run_finished','operation_dispatch_intent','operation_returned','operation_failed',
             'placement_recipe','goal_attached','invalid_observation','fresh_observation','returned_transition','own_response','branch_closed'}
    for i,e in enumerate(events):
        need(e['kind'] in allowed and e['index']==i and e['previous_sha256']==previous and
             e['event_sha256']==physical.source.digest({k:v for k,v in e.items() if k!='event_sha256'}),'event chain')
        previous=e['event_sha256']
        if 'branch' in e:
            need(closed<2 and e['branch']==closed and all(e[k]==r['rows'][closed][k] for k in ('root_slot','condition','arm','selection_digest')),'branch identity')
        if e['kind']=='branch_closed':
            need(closed<2 and e['receipt']==r['rows'][closed],'branch closure');closed+=1
        if e['kind'] in ('operation_dispatch_intent','operation_returned','operation_failed'):
            op=e['receipt'];n=op['index'];need(op['branch']==closed and op['resource'] in contract()['caps'],'operation identity')
            if e['kind']=='operation_dispatch_intent':
                need(n==len(intents) and op['status']=='pending','operation sequence');intents[n]=op;stack.append(n)
            else:
                need(stack and stack.pop()==n and n not in terminals and op['status']==('returned' if e['kind']=='operation_returned' else 'failed') and
                     dict({k:v for k,v in op.items() if k!='error'},status='pending')==intents[n],'operation disposition');terminals[n]=op
        if e['kind']=='invalid_observation':
            name=e['file'];need(Path(name).name==name and name not in names and hashlib.sha256(files[name]).hexdigest()==e['file_sha256'],'invalid image');names.add(name)
        if e['kind']=='fresh_observation':
            f=e['receipt'];name=f['png'];need(Path(name).name==name and name not in names and len(files[name])<=1024**2 and
                hashlib.sha256(files[name]).hexdigest()==f['file_sha256'],'PNG bytes')
            with Image.open(io.BytesIO(files[name])) as image:
                need(image.format=='PNG' and image.mode=='RGB' and image.size==(224,224),'PNG format');array=np.asarray(image).copy()
            need(physical.moving.image_sha(array)==f['pixel_sha256'],'PNG pixels');physical.moving.guard(array,physical.source.vector(f['state']));names.add(name)
    need(previous==r['journal_chain_tail'] and closed==2 and not stack and len(intents)==len(terminals) and
         r['actual_operation_receipts']==[terminals[i] for i in range(len(terminals))] and set(files)==names,'complete receipts and artifacts')
    counts={k:dict(attempted=0,completed=0) for k in contract()['caps']}
    for op in r['actual_operation_receipts']:
        counts[op['resource']]['attempted']+=1;counts[op['resource']]['completed']+=op['status']=='returned'
    need(counts==r['counts'] and all(v['attempted']==v['completed']<=contract()['caps'][k] for k,v in counts.items()),'resource counts')
    for i,row in enumerate(r['rows']):
        plan=p['plans'][i];root=p['source_roots'][i]
        need(row['index']==row['root_slot']==i and row['condition']=='off' and row['arm']=='cem_matched' and
             row['plan']==plan and row['selection_digest']==plan['selection_digest'] and row['status'] in ('native_stopped','horizon_completed'),'row plan')
        scoped=[e for e in events if e.get('branch')==i];ops=[o for o in r['actual_operation_receipts'] if o['branch']==i]
        physical.audit_labels(row,plan,scoped,p)
        for phase,label,actions in (('warmup','warmup',physical.moving.warmup_commands()),('plan','own',physical.raw15(plan['commands']))):
            curr=[o for o in ops if o['resource']=='public_step' and o['phase']==phase]
            returned=[e for e in scoped if e['kind']=='returned_transition' and e['phase']==phase]
            need(len(curr)<=len(actions) and len(returned)==len(curr)==row[label+'_attempted']==row[label+'_returned'] and
                 [o['step'] for o in curr]==list(range(1,len(curr)+1)) and
                 all(o['raw_bits']==actions[j].tobytes().hex() for j,o in enumerate(curr)) and
                 all(e['step']==j+1 and e['actual_bits']==e['requested_bits']==actions[j].tobytes().hex() for j,e in enumerate(returned)), 'exact raw action chronology')
        if row['initial_exact']:
            need(all(row[k]==root[k] for k in ('root_state','root_sha256','goal','goal_pose')) and
                 row['goal_pixel_sha256']==root['goal_pixels']['off'] and
                 physical.source.root_hash(physical.source.vector(row['root_state']),physical.source.vector(row['goal']))==row['root_sha256'],'root/goal exact')
        observations=[e for e in scoped if e['kind']=='fresh_observation']
        need(row['PNG']==len(observations),'observation count')
        for e in observations:
            f=e['receipt'];phase,step,overlay=f['phase'],f['step'],f['overlay_index']
            need(phase in ('warmup','plan') and type(step) is int and type(overlay) is int and
                 0<=step<=(10 if phase=='warmup' else 15) and overlay in (0,1,2) and
                 f['condition']==('on' if overlay==1 else 'off') and e['pre']==e['post'] and
                 f['snapshot']==e['post']['snapshot'] and f['state']==f['snapshot']['state'] and
                 e['pre']['renderer']==root['renderer'] and e['pre']['goal_pose']==root['goal_pose'],'observation binding')
            if phase=='warmup':
                trace=p['warm_traces'][i]
                need(e['pre']['goal_state']==trace[0]['pre']['goal_state'] and
                     e['pre']['goal_rgb_sha256']==trace[0]['pre']['goal_rgb_sha256'],'warm goal')
                ref=physical.native.allmode.reference_for(trace,step,overlay)
                if ref is not None:physical.native.allmode.check_warm_frame(e,ref)
            else:
                need(e['pre']['goal_state']==root['goal'] and e['pre']['goal_rgb_sha256']==root['goal_pixels']['off'],'plan goal')
                if step:
                    matches=[x for x in scoped if x['kind']=='returned_transition' and x['phase']=='plan' and x['step']==step]
                    need(len(matches)==1 and f['state']==matches[0]['state'] and f['snapshot']==matches[0]['snapshot'],'observation transition')
        keys=[('warmup',b,k) for b in range(row['warmup_returned']+1) for k in (0,1,2)]
        if row['initial_exact']:keys += [('plan',b,k) for b in range(row['own_returned']+1) for k in (0,1,2)]
        need([(e['receipt']['phase'],e['receipt']['step'],e['receipt']['overlay_index']) for e in observations]==keys,'exact unpadded render sequence')
        for j in range(0,len(observations),3):
            a,b,c=observations[j:j+3]
            need(a['receipt']['file_sha256']==c['receipt']['file_sha256'] and a['receipt']['pixel_sha256']==c['receipt']['pixel_sha256'] and
                 a['receipt']['state']==b['receipt']['state']==c['receipt']['state'],'off/on/off closure')
        selected=[e['receipt'] for e in observations if e['receipt']['overlay_index']==0]
        need(row['trajectory']==selected,'off trajectory')
        prefix=[{k:deepcopy(e[k]) for k in ('receipt','pre','post')} for e in observations
                if e['receipt']['phase']=='warmup' or e['receipt']['step']<=5]
        need(row['prefix51']==dict(complete=len(prefix)==51,frames=prefix),'actual prefix51')
        cp=[f for f in selected if f['phase']=='plan' and f['step'] in physical.CP]
        need(row['checkpoints']==[{k:f[k] for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')} for f in cp] and
             row['missing_checkpoints']==[b for b in physical.CP if b not in [f['step'] for f in cp]],'checkpoint missing semantics')
        responses=row['responses'];need([x['step'] for x in responses]==list(range(1,len(responses)+1)) and
            len(responses)==row['own_returned'] and [e['response'] for e in scoped if e['kind']=='own_response']==responses,'response sequence')
        for response in responses:
            calc=physical.native.metrics(row['goal'],response['state'])
            need(response['native_metrics']==calc['native'] and response['reward']==calc['native7_reward'] and
                 response['native_success'] is response['terminated'] is calc['native_success'],'native labels')
            transition=[e for e in scoped if e['kind']=='returned_transition' and e['phase']=='plan' and e['step']==response['step']]
            need(len(transition)==1 and all(transition[0][k]==response[k] for k in ('state','reward','terminated','truncated')),'response native return')
        if row['status']=='horizon_completed':need(row['initial_exact'] and row['warmup_returned']==10 and row['own_returned']==15,'full horizon')
        else:
            transitions=[e for e in scoped if e['kind']=='returned_transition']
            goal=root['goal'] if row['initial_exact'] else p['warm_traces'][i][0]['pre']['goal_state']
            final=observations[-1]['receipt'] if observations else None
            need(bool(transitions and (transitions[-1]['terminated'] or transitions[-1]['truncated'])) or
                 bool(final and physical.native.metrics(goal,final['state'])['native_success']), 'authenticated native stop')
        need(all(not (x['terminated'] or x['truncated']) for x in responses[:-1]),'no post-terminal padding')
        kinds={k:sum(o['resource']==k and o['status']=='returned' for o in ops) for k in contract()['caps']}
        need(kinds['constructor']==kinds['close']==1 and kinds['reset']==2 and kinds['set_state']==4 and
             kinds['goal_render']==2 and kinds['goal_bind']==int(row['initial_exact']) and
             kinds['goal_setter']==2+int(row['initial_exact']) and
             kinds['physics_step']==4+10*kinds['public_step'] and kinds['explicit_render']==row['PNG'],
             'native per-root lifecycle counts')
    return dict(schema=NAME+'.consume',status=r['status'],fixed_denominator=2,counts=counts,
                rows=r['rows'],physical_report_sha256=report_sha,host_report_sha256=p['host_report_sha256'],
                physical_release=False,scientific_admission=False,independent_physics_rerun=False,
                limitations='saved trace audit only; not full six-slot verifier or independent physics replay')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','verify','run','consume'))
    ap.add_argument('--prepared',required=True);ap.add_argument('--sha256');ap.add_argument('--output');ap.add_argument('--authorization');ap.add_argument('--report-sha256');ap.add_argument('--host-report-sha256')
    a=ap.parse_args()
    if a.command=='prepare':need(a.host_report_sha256,'external complete host report SHA');result=dict(sha256=prepare(a.prepared,a.host_report_sha256))
    elif a.command=='verify':verify(a.prepared,a.sha256);result=dict(verified=True)
    elif a.command=='run':
        need(a.authorization==AUTH and a.sha256 and a.output,'explicit physical run arguments')
        def factory():
            import gymnasium as gym
            import stable_worldmodel.envs
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False,render_action=False)
        r=run(a.prepared,a.sha256,a.output,factory,authorization=a.authorization)
        result=dict(status=r['status'],report_sha256=physical.sha(Path(a.output)/'report.json'),errors=r['errors'])
    else:
        need(a.sha256 and a.output and a.report_sha256,'consume arguments')
        r=consume(a.output,a.report_sha256,a.prepared,a.sha256)
        result={k:v for k,v in r.items() if k!='rows'}
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
