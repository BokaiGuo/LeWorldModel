"""Strict saved physical consumer: four controller branches, six E1 slots.
No neural, simulator, producer execution or latent-error fabrication.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_experiment1_common_controller_physical_20261028_v1 as physical

NAME='agent_experiment1_common_controller_physical_consume_20261028_v1'
ROOT,kernel,source=physical.ROOT,physical.kernel,physical.source
require,sha,read=physical.require,physical.sha,physical.read
SEALS=('61a16e772175627fed91d53c95e0b78566117b02676b54184d7d8e5e11e01f97',
    'ecc923f93a2ee9a03685e5ff8837c3cabd3506243b025c8ed8302af72da8dc86',
    '96fe737f5eff069e5e9d14cf655ff5d562313e314b846843d1dec02097cc8f85')


def vector(v):
    a=np.asarray(v,dtype=float);require(a.shape==(7,) and np.isfinite(a).all(),'actual finite native7, no truncation');return a


def metric(goal,state):
    state=vector(state);d=state-vector(goal);a=abs(float(d[4]));native=min(a,2*np.pi-a)
    wrapped=float(abs(np.arctan2(np.sin(d[4]),np.cos(d[4]))))
    return dict(native_success=bool(np.linalg.norm(d[:4])<20 and native<np.pi/9),native_distance=float(np.linalg.norm(d)),
        native_angle_term=float(native),negative_native_angle=bool(native<0),
        wrapped_allxy_success=bool(np.linalg.norm(d[:4])<20 and wrapped<np.pi/9),wrapped_object_angle_error=wrapped,
        object_position_error=float(np.linalg.norm(d[2:4])),object_pose=state[2:5].tolist(),object_success=None)


def operations(events,p,report):
    pending={};terminal={};stack=[];counts={k:dict(attempted=0,completed=0) for k in p['contract']['caps']}
    for e in events:
        kind=e['kind']
        if kind not in ('operation_dispatch_intent','operation_returned','operation_failed'):continue
        r=e['receipt'];i=r['operation_index'];resource=r['resource']
        require(type(i) is int and resource in counts,'operation identity')
        if kind=='operation_dispatch_intent':
            require(i==len(pending) and r['status']=='pending','unique sequential intent');pending[i]=r;stack.append(i)
        else:
            require(stack and stack.pop()==i and i in pending and i not in terminal and
                r['status']==('returned' if kind=='operation_returned' else 'failed'),'proper nested operation return/failure')
            original={k:v for k,v in r.items() if k not in ('error_type','message')}
            require(dict(original,status='pending')==pending[i],'same operation identity');terminal[i]=r
    require(not stack and len(terminal)==len(pending) and
        report['actual_operation_receipts']==[terminal[i] for i in range(len(terminal))],'complete durable actual receipts; incomplete journals reject')
    for r in terminal.values():
        i=r['branch'];require(type(i) is int and 0<=i<4,'fixed operation branch')
        plan=p['plans'][i]
        require(all(r[k]==plan[k] for k in ('root_slot','arm','selection_digest')),'operation selected-plan identity')
        counts[r['resource']]['attempted']+=1;counts[r['resource']]['completed']+=r['status']=='returned'
    require(counts==report['counts'] and all(c['attempted']<=p['contract']['caps'][k] for k,c in counts.items()),'actual resources/caps')
    return list(terminal.values())


def assemble(r,events,p):
    require(r['schema']==physical.NAME+'.report' and r['status'] in ('common_controller_physical_calibration_recorded','partial_or_failed') and
        r['contract']==p['contract']==physical.contract(p['contract']['wall_seconds']) and
        r['sourcepack_sha256']==p['sourcepack_sha256']==physical.SOURCE_SHA and r['source_prepare_id']==p['source_prepare_id'] and
        r['fixed_branch_denominator']==4 and r['scientific_slot_denominator']==6 and r['learned_unknown_slots']==2,
        'original physical protocol/full denominators')
    require(r['new_neural_calls']==r['new_independent_samples']==0 and r['training'] is False and
        r['calibration_only'] is True and r['P1_admission'] is r['scientific_admission'] is False,'calibration only, no latent evaluation')
    require(events and events[0]['kind']=='run_started' and events[0]['prepare']==p and
        events[0]['execution_binding']==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']) and
        events[-1]['kind']=='run_finished' and events[-1]['status']==r['status'] and events[-1]['counts']==r['counts'] and
        events[-1]['errors']==r['errors'] and events[-1]['event_sha256']==r['journal_chain_tail'],'durable run boundaries')
    order=[(s,a) for s in (0,1) for a in physical.ARMS]
    require([(x['root_slot'],x['arm']) for x in r['rows']]==[(x['root_slot'],x['arm']) for x in p['plans']]==order,'all four ordered branches')
    receipts=operations(events,p,r)
    require([e['receipt'] for e in events if e['kind']=='branch_closed']==[x for x in r['rows'] if x['status']!='not_started'],'durable closed rows')
    checkpoints=[];expected_science=[]
    for i,(row,plan) in enumerate(zip(r['rows'],p['plans'],strict=True)):
        require(row['branch_index']==i and row['selected_plan']==plan and row['selection_digest']==plan['selection_digest']==
            source.digest({k:v for k,v in plan.items() if k!='selection_digest'}),'full selected raw10/target/predictions seal')
        ev=[e for e in events if e.get('branch')==i];ownops=[x for x in receipts if x['branch']==i and x['resource']=='public_step']
        own=[x for x in ownops if x['phase']=='plan'];warm=[x for x in ownops if x['phase']=='warmup']
        require(row['own_attempted']==len(own) and row['own_returned']==sum(x['status']=='returned' for x in own) and
            row['warmup_returned']==sum(x['status']=='returned' for x in warm) and row['full_horizon']==(row['own_returned']==10),'row attempted/returned denominator')
        for ops,commands in ((own,plan['commands']),(warm,p['contract']['warmup_raw'])):
            require([x['step'] for x in ops]==list(range(1,len(ops)+1)) and len(ops)<=10 and
                all(x['raw_action']==commands[j] for j,x in enumerate(ops)),'contiguous literal raw10, no padding/substitution')
        observed=[e['receipt'] for e in ev if e['kind']=='fresh_observation'];tracking=[e['receipt'] for e in ev if e['kind']=='tracking_observation']
        require(row['PNG']==len(observed) and row['tracking']==tracking,'all actual frame/tracking census')
        for cp in observed:
            guards=[e for e in ev if e['kind']=='guard_receipt' and e['phase']==cp['phase'] and e['step']==cp['step']]
            require(len(guards)==1,'exact guard per valid saved RGB')
            guard=guards[0]
            expected=guard['canonical_agent_pixels']>0 and bool(((vector(cp['state'])[:4]>=0)&(vector(cp['state'])[:4]<=512)).all()) and np.array_equal(vector(cp['state']),vector(cp['observation_state'])) and guard['render_preserved_dynamics']
            require(type(guard['accepted']) is bool and guard['accepted']==bool(expected),'saved guard arithmetic')
            require(next(e['index'] for e in ev if e['kind']=='fresh_observation' and e['receipt']==cp)<guard['index'],'failed frame persisted before guard')
        planobs=[c for c in observed if c['phase']=='plan']
        require([x['step'] for x in tracking]==[x['step'] for x in planobs],'one tracking receipt per actual plan render')
        for cp,tr in zip(planobs,tracking,strict=True):
            require(vector(cp['state']).tolist()==vector(cp['observation_state']).tolist(),'native observation agreement')
            require(tr==dict(step=cp['step'],png=cp['png'],pixel_sha256=cp['pixel_sha256'],state=cp['state'],
                native=metric(row['tracking_goal'],cp['state']),target_sha256=physical.producer.core.digest(plan['target']),
                model_prefix_target_cost=plan['prefix_target_cost'],model_terminal_cost=plan['model_terminal_cost'],
                observed_latent192=None,latent_tracking_error=None,
                model_successor=plan['predicted_contexts'][cp['step']//5-1] if cp['step'] in (5,10) else None),'strict native7 tracking/latent null/prediction alignment')
        transitions=[e for e in ev if e['kind']=='returned_transition'];owntrans=[e for e in transitions if e['phase']=='plan'];warmtrans=[e for e in transitions if e['phase']=='warmup']
        require([e['step'] for e in owntrans]==list(range(1,row['own_returned']+1)) and
            [e['step'] for e in warmtrans]==list(range(1,row['warmup_returned']+1)),'every actual returned step saved')
        placement=[e for e in ev if e['kind']=='placement_request'];attached=[e for e in ev if e['kind']=='goal_attached']
        for transition in transitions:
            if transition['phase']=='plan':
                goal=row['goal'];require(transition['command']==plan['commands'][transition['step']-1] and len(attached)==1 and
                    transition['returned_goal_state']==goal and transition['returned_goal_rgb_sha256']==attached[0]['goal_rgb_sha256'],'returned own raw/goal exact')
            else:require(len(placement)==1,'original placement goal');goal=placement[0]['original_goal']
            native=metric(goal,transition['state'])
            require(type(transition['terminated']) is bool and type(transition['truncated']) is bool and
                transition['terminated']==native['native_success'] and transition['reward']==-native['native_distance'],'full7 reward/native success')
        if row['initial_exact']:
            require(row['goal']==row['tracking_goal'] and source.root_hash(row['root_state'],row['goal'])==plan['root_sha256'] and
                row['initial_metrics']==metric(row['goal'],row['root_state']) and len(attached)==1 and attached[0]['goal']==row['goal'],'exact native root+goal')
        if row['final_state'] is not None:
            require(owntrans and row['final_state']==owntrans[-1]['state'] and row['final_metrics']==metric(row['goal'],row['final_state']) and
                row['native_success']==row['final_metrics']['native_success'] and row['terminated']==owntrans[-1]['terminated'] and
                row['truncated']==owntrans[-1]['truncated'],'last actual full7 final state')
        require([c['step'] for c in row['checkpoints']]==sorted(set(c['step'] for c in row['checkpoints'])),'unique checkpoint order')
        for cp in row['checkpoints']:
            guards=[e for e in ev if e['kind']=='guard_receipt' and e['phase']=='plan' and e['step']==cp['step']]
            require(cp in planobs and cp['step'] in (0,5,10) and len(guards)==1 and guards[0]['accepted'] is True,'only real guarded0/5/10 checkpoints')
            require(cp['state']==(row['root_state'] if cp['step']==0 else owntrans[cp['step']-1]['state']),'checkpoint state/transition')
            checkpoints.append(dict(branch_index=i,root_slot=row['root_slot'],arm=row['arm'],selection_digest=row['selection_digest'],checkpoint=deepcopy(cp),target=plan['target']))
        if row['status']=='horizon_completed':require(row['full_horizon'] and row['terminated'] is False and row['truncated'] is False,'full10 status')
        if row['status']=='native_success':require(row['native_success'] is True and row['terminated'] is True and row['truncated'] is False,'native terminal status')
        if row['status'] in ('horizon_completed','native_success','native_stopped'):
            require([c['step'] for c in planobs]==list(range(row['own_returned']+1)),'all clean plan observations')
            require([c['step'] for c in row['checkpoints']]==[k for k in (0,5,10) if k<=row['own_returned']],'all available real checkpoint slots')
            require(all(e['accepted'] is True for e in ev if e['kind']=='guard_receipt'),'clean row has no failed guard')
    for s in (0,1):
        for a in physical.producer.ALL_ARMS:
            if a==physical.LEARNED:expected_science.append(dict(root_slot=s,arm=a,status='unknown',commands=None,native_success=None,resources=None,error='no certified learned prior; not dispatched, not a failure rate'))
            else:
                row=next(x for x in r['rows'] if x['root_slot']==s and x['arm']==a)
                expected_science.append(dict(root_slot=s,arm=a,status=row['status'],branch_index=row['branch_index'],native_success=row['native_success']))
    require(r['scientific_rows']==expected_science and len(expected_science)==6 and
        r['full_horizons']==sum(x['full_horizon'] for x in r['rows']) and r['native_successes']==sum(x['status']=='native_success' for x in r['rows']) and
        r['engineering_failures']==sum(x['status']=='failed' for x in r['rows']),'six-slot unknown/full/failure/native census')
    if r['status']=='common_controller_physical_calibration_recorded':
        c=r['counts'];steps=sum(x['warmup_returned']+x['own_returned'] for x in r['rows'])
        require(not r['errors'] and all(x['status'] in ('horizon_completed','native_success','native_stopped') for x in r['rows']) and
            all(v['attempted']==v['completed'] for v in c.values()) and
            c['environment_construct']['completed']==c['environment_close']['completed']==4 and c['reset']['completed']==8 and
            c['set_state']['completed']==16 and c['goal_render']['completed']==8 and c['goal_bind']['completed']==4 and
            c['public_step']['completed']==steps and c['physics_step']['completed']==16+10*steps and
            c['explicit_render']['completed']==sum(x['PNG'] for x in r['rows']),'complete/earlystop actual-count census')
    return checkpoints


def consume(output,expected):
    for f,h in zip((Path(physical.__file__),ROOT/'tests'/('test_'+physical.NAME+'.py'),ROOT/'innerdocs'/(physical.NAME+'.md')),SEALS,strict=True):
        require(sha(f)==h,'frozen physical source/tests/doc')
    output=Path(physical.absolute(str(output)));physical.primitives.hash_literal(expected);files=list(output.iterdir())
    require(all(f.is_file() and not f.is_symlink() for f in files) and sum(f.stat().st_size for f in files)<=physical.CAP,'bounded flat actual output')
    require(sha(output/'report.json')==expected,'external actual physical report SHA')
    r=read(output/'report.json',physical.CAP);p=physical.verify(r['prepared'],r['prepare_sha256'])
    require(read(output/'execution_binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'execution binding')
    jp=output/'journal.jsonl';require(sha(jp)==r['journal_sha256'],'journal SHA')
    data=jp.read_bytes();require(data.endswith(b'\n'),'complete journal')
    events=[];previous=None
    for i,line in enumerate(data.splitlines()):
        e=physical.producer.decode(line)
        require(e['index']==i and e['previous_sha256']==previous and e['event_sha256']==source.digest({k:v for k,v in e.items() if k!='event_sha256'}),'journal hashchain')
        previous=e['event_sha256'];events.append(e)
    require(previous==r['journal_chain_tail'],'final chain tail')
    checkpoints=assemble(r,events,p);names={'report.json','execution_binding.json','journal.jsonl'}
    manifest=read(p['manifest']);data=kernel.load_source(manifest['sourcepack'],manifest['sourcepack_sha256'],
        manifest['source_prepared'],manifest['source_prepare_sha256'])
    source_frames={}
    for i,row in enumerate(r['rows']):
        ref=data['references'][row['root_slot']]
        if row['initial_exact']:
            require(kernel.moving.exact(vector(row['root_state']),ref['states'][-1]) and
                kernel.moving.exact(vector(row['goal']),ref['goal']),'actual original source root/goal7')
            for phase,step,j in [('warmup',0,0),('warmup',5,1),('warmup',10,2),('plan',0,2)]:
                obs=[e['receipt'] for e in events if e['kind']=='fresh_observation' and e.get('branch')==i and e['receipt']['phase']==phase and e['receipt']['step']==step]
                require(len(obs)==1 and kernel.moving.exact(vector(obs[0]['state']),ref['states'][j]) and
                    obs[0]['snapshot']==ref['dynamics'][j],'original complete0/5/10 source state/snapshot')
                source_frames[obs[0]['png']]=ref['history'][j]
            attached=[e for e in events if e['kind']=='goal_attached' and e.get('branch')==i]
            require(len(attached)==1 and attached[0]['goal_rgb_sha256']==kernel.moving.image_sha(ref['goal_rgb']),'original goalRGB binding')
    from PIL import Image
    for e in events:
        if e['kind']=='fresh_observation':
            cp=e['receipt'];name=cp['png'];require(Path(name).name==name and sha(output/name)==cp['file_sha256'],'actual PNG file SHA')
            with Image.open(output/name) as image:
                require(image.size==(224,224) and image.mode=='RGB','RGB224');pixels=np.asarray(image)
            require(pixels.dtype==np.uint8 and kernel.moving.image_sha(pixels)==cp['pixel_sha256'],'actual pixel SHA');names.add(name)
            if name in source_frames:require(kernel.moving.exact(pixels,source_frames[name]),'original source RGB bytes, not merely matching report')
            guards=[g for g in events if g['kind']=='guard_receipt' and g.get('branch')==e.get('branch') and g['phase']==cp['phase'] and g['step']==cp['step']]
            require(len(guards)==1 and guards[0]['canonical_agent_pixels']==int(np.count_nonzero(np.all(pixels==[65,105,225],axis=-1))),'actual guard pixel census')
        elif e['kind']=='invalid_render' and e['receipt'].get('raw_render'):
            cp=e['receipt'];name=cp['raw_render'];require(Path(name).name==name and sha(output/name)==cp['file_sha256'],'saved invalid-render bytes');names.add(name)
    require({f.name for f in files}==names,'complete exact artifact census')
    bindings=dict(p['bindings']);physical.merge(bindings,{r['prepared']:r['prepare_sha256']})
    physical.merge(bindings,{str(f):sha(f) for f in files})
    for f in (Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):physical.merge(bindings,{str(f.resolve()):sha(f)})
    require(bindings[str(output/'report.json')]==expected,'report final binding')
    for f,h in bindings.items():require(sha(f)==h,'final full saved closure')
    for c in checkpoints:c['png_path']=str(output/c['checkpoint']['png'])
    return dict(schema=NAME+'.consume',status=r['status'],rows=deepcopy(r['rows']),scientific_rows=deepcopy(r['scientific_rows']),
        checkpoints=checkpoints,counts=r['counts'],report=r,prepared=p,bindings_sha256=bindings,physical_report_sha256=expected,
        fixed_branch_denominator=4,scientific_slot_denominator=6,learned_unknown_slots=2,
        checkpoint_complete=all([c['step'] for c in x['checkpoints']]==[0,5,10] for x in r['rows']),
        observed_latent_errors_available=False,new_neural_calls=0,calibration_only=True,P1_admission=False,scientific_admission=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('output','report-sha'):p.add_argument('--'+k,required=True)
    a=p.parse_args();r=consume(a.output,a.report_sha)
    print(source.canonical({k:v for k,v in r.items() if k not in ('rows','scientific_rows','checkpoints','report','prepared','bindings_sha256')}),flush=True)


if __name__=='__main__':main()
