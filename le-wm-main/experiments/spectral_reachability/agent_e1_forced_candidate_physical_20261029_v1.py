"""Six-slot E1 native physical adapter; execution requires explicit MAIN authorization."""
from __future__ import annotations

from copy import deepcopy
import argparse
import os
import shutil
from pathlib import Path
import numpy as np

from experiments.spectral_reachability import agent_e1_six_slot_common_cuda_20261029_v2 as producer
from experiments.spectral_reachability import agent_e1_forced_candidate_replay_cuda_20261029_v1 as replay
from experiments.spectral_reachability import agent_experiment1_common_controller_physical_20261028_v1 as old

NAME = 'agent_e1_forced_candidate_physical_20261029_v1'
ROOT = old.ROOT
OUTPUT = ROOT/'results/agent_e1_forced_candidate_replay_cuda_20261029_v1_cuda_MAIN01'
RECEIPT = ROOT/'results/agent_e1_forced_candidate_replay_cuda_20261029_v1_consume_MAIN01.json'
RECEIPT_SHA = '0605d651ffcb7b4edafed61c12604a89453594e7df5b573b3a944a048c548dc5'
REPORT_SHA = 'ba2df80d87c825452cdbf3050813e6720f074cd69ba70aea4f716b9007d5717e'
COMPLETION_SHA = '8e1af678def94bb603ca2e50452452892b1188aa19a35e85281815c2e7e68783'
TIMING_SHA = 'dd3c6dfb30fb906f06f22b3b188cf490010d473377d217acc8633b192423d8db'
SOURCEPACK = ROOT/'results/p1_constructive_moving_history_source_20261027_v1_main01/sourcepack.json'
SOURCEPACK_SHA = 'c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6'
SOURCE_PREPARED = ROOT/'results/p1_constructive_moving_history_source_20261027_v1_prepare01.json'
SOURCE_PREPARED_SHA = 'c1292f0ebfb1827e5da92b32ab13d2e4a12c025c4741277ab4c4f478d6ee8cb9'
ROOT_SHA = ('a3bd09470de75652506544b8bea0ff3c5fbae0c0d3b28775da47ee3e1adbb6f7',
            'df038a122a660da57675fa281ec7dd358c9f9f67cb387aca9595f31711ca60ad')
ARMS = replay.contract.METHODS
SLOTS = replay.SLOTS
require = old.require
AUTH='MAIN_E1_FORCED_CANDIDATE_PHYSICS_ONLY'
FRAME_LIMIT=160*1024
FRAME_COUNT=132
JOURNAL_LIMIT=6*1024**2
BINDING_LIMIT=256*1024
REPORT_LIMIT=4*1024**2
EMERGENCY_LIMIT=32*1024


def storage_census():
    """Allocation limits, not an assertion that arbitrary native returns fit."""
    parts=dict(frames=FRAME_COUNT*FRAME_LIMIT,journal=JOURNAL_LIMIT,
               binding=BINDING_LIMIT,report=REPORT_LIMIT,emergency=EMERGENCY_LIMIT)
    total=sum(parts.values())
    require(total<=old.CAP,'all allocated bytes fit unchanged cap')
    return dict(parts=parts,total=total,cap=old.CAP,slack=old.CAP-total,
                frame_count=FRAME_COUNT,frame_bytes=FRAME_LIMIT,
                scope='enforced output allocation; arbitrary malformed native output is not bounded')


class Store(old.Store):
    """Within original cap reserve terminal report and a separate failure marker."""
    def __init__(self,output,wall=600):
        super().__init__(output,wall);self.frame_count=0;self.frame_bytes=0

    def check(self,size=4096):
        self.check_time()
        require(self.used()+size<=old.CAP-REPORT_LIMIT-EMERGENCY_LIMIT and
                shutil.disk_usage(self.path).free>=old.FLOOR+size,'ordinary allocation and terminal reserve')

    def write(self,name,value,append=False,final=False):
        require(type(value) is bytes and Path(name).name==name and name not in ('','.','..') and
                not (self.path/name).is_symlink(),'flat own bytes')
        if not final:self.check_time()
        existing=(self.path/name).stat().st_size if (self.path/name).exists() else 0
        if name=='journal.jsonl':require(existing+len(value)<=JOURNAL_LIMIT,'journal allocation exceeded')
        elif name=='execution_binding.json':require(len(value)<=BINDING_LIMIT,'binding allocation exceeded')
        elif name=='report.json':require(final and len(value)<=REPORT_LIMIT,'report allocation exceeded')
        elif name=='emergency.json':require(final and len(value)<=EMERGENCY_LIMIT,'emergency allocation exceeded')
        else:
            require(not final and not append and self.frame_count<FRAME_COUNT and
                    self.frame_bytes+len(value)<=FRAME_COUNT*FRAME_LIMIT,
                    'frame allocation exceeded')
        if not final:
            require(self.used()+len(value)<=old.CAP-REPORT_LIMIT-EMERGENCY_LIMIT,'terminal reserve before write')
        else:
            require(self.used()+len(value)<=old.CAP-(EMERGENCY_LIMIT if name=='report.json' else 0),'final allocation')
        require(shutil.disk_usage(self.path).free>=old.FLOOR+len(value),'disk floor')
        path=self.path/name
        with path.open('ab' if append else 'xb') as f:
            require(f.write(value)==len(value),'short write');f.flush();os.fsync(f.fileno())
        fd=os.open(self.path,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        if name not in ('journal.jsonl','execution_binding.json','report.json','emergency.json'):
            self.frame_count+=1;self.frame_bytes+=len(value)
        return old.sha(path)

    def emergency(self,stage):
        journal=self.path/'journal.jsonl'
        record=dict(schema=NAME+'.emergency',status='partial_unverified',stage=stage,
                    journal_sha256=old.sha(journal) if journal.exists() else None,
                    journal_chain_tail=self.previous,events=self.index,
                    report_present=(self.path/'report.json').exists(),
                    fixed_branch_denominator=6,scientific_admission=False)
        return self.put('emergency.json',record,final=True)


def contract():
    c = old.kernel.low.contract(6, 600)
    c['caps'].update(environment_construct=6,environment_close=6)
    require(c['caps']['public_step']==120 and c['caps']['physics_step']==1224
            and c['caps']['explicit_render']==132, 'six branch native operation census')
    c.update(schema=NAME+'.contract', roots=list(ROOT_SHA),arms=list(ARMS),
                denominator=6, sourcepack_sha256=SOURCEPACK_SHA, retries=0,padding=0,
                no_neural=True,training=False,physical_outcome_selection=False,
                scientific_admission=False, P1_admission=False,
                warmup_raw=old.kernel.moving.warmup_commands().tolist(),
                storage_census=storage_census(),
                inherited_controller_cost='parent receipt separately',inherited_extra_replay_per_axis=24,
                qualification='known roots, no total compute equivalence')
    return c


def source_data():
    data=old.kernel.load_source(str(SOURCEPACK),SOURCEPACK_SHA,str(SOURCE_PREPARED),SOURCE_PREPARED_SHA)
    require(len(data['references'])==2 and tuple(x['root_sha256'] for x in data['references'])==ROOT_SHA,
            'both original root identities')
    return data


def checked_plans(receipt, payload, data, fixed, authority, previous):
    require(receipt['schema']==replay.NAME+'.consume' and receipt['status']=='six_forced_saved_replays_verified'
            and receipt['report_sha256']==REPORT_SHA and receipt['completion_sha256']==COMPLETION_SHA
            and receipt['timing_sha256']==TIMING_SHA and receipt['result']['slots']==6
            and receipt['result']['predict_per_axis']==24, 'actual full six receipt')
    tasks,targets,_=replay.host.tasks_and_targets(authority,previous)
    plans=[]
    for i, (root,arm) in enumerate(SLOTS):
        slot=fixed['slots'][i];output=payload[f'slot_{i}.replay.z'];task=tasks[root];target=targets[root]
        require((slot['root'],slot['method'])==(root,arm) and output['raw']==slot['raw10']
                and old.source.digest(output['raw'])==old.source.digest(slot['raw10'])
                and len(output['whole'])==len(output['segmented'])==2,'fixed literal replay actions and context')
        old.kernel.commands(slot['raw10']);producer.core.old.vec(target)
        original=previous['slots'][root]
        bridge=dict(source_slot=root,task_export=original['task_export'],task_sha256=original['task_sha256'],
                    task_id=original['task_id'],blind_sha256=original['blind_sha256'])
        old.primitives.helper.task_bridge(bridge,data['references'][root],{})
        prefix=float(np.mean((producer.core.old.vec(output['whole'][0]['latent_history'][-1])-producer.core.old.vec(target))**2))
        terminal=float(np.mean((producer.core.old.vec(output['whole'][1]['latent_history'][-1])-producer.core.old.vec(task['goal']))**2))
        projected=dict(root_slot=root,root_sha256=ROOT_SHA[root],arm=arm,
            commands=slot['raw10'],task_id=original['task_id'],blind_sha256=original['blind_sha256'],
            task_export=original['task_export'],task_sha256=original['task_sha256'],task_contract_sha256=original['task_contract_sha256'],
            target=target,predicted_contexts=output['whole'],model_terminal_cost=terminal,
            prefix_target_cost=prefix,tracking_cost=.5*(prefix+terminal),
            compiler_authority=replay.NAME,compiler_report=str(OUTPUT/'report.json'),compiler_report_sha256=REPORT_SHA,
            compiler_selection_sha256=slot['raw_sha256'],own_plan_sha256=old.source.digest(slot))
        projected['selection_digest']=old.source.digest(projected)
        plans.append(projected)
    require([(p['root_slot'],p['arm']) for p in plans]==list(SLOTS),'all fixed six slots')
    return plans


def build():
    """Authenticate existing full saved consume and original task bridge; no physics."""
    require(old.sha(RECEIPT)==RECEIPT_SHA, 'external receipt SHA')
    persisted=old.read(RECEIPT,old.CAP)
    receipt=replay.consume(OUTPUT,REPORT_SHA,COMPLETION_SHA,TIMING_SHA)
    require(receipt==persisted, 'entire saved receipt recomputation')
    report=producer.parse_canonical(producer.read_bound(OUTPUT/'report.json',REPORT_SHA,producer.RESERVE))
    replay_prepare=Path(report['prepared'])
    replay_binding=producer.parse_canonical(producer.read_bound(replay_prepare,report['prepare_sha256'],16*1024*1024))
    payload,_=producer.read_store(OUTPUT,report)
    data=source_data();fixed=replay.forced();authority,previous=replay.host.authorities()
    plans=checked_plans(receipt,payload,data,fixed,authority,previous)
    files=[Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    bindings={str(f.resolve()):old.sha(f) for f in files}
    for f in (RECEIPT, OUTPUT/'report.json', OUTPUT/'completion.json', OUTPUT/'timing.json', Path(producer.__file__),Path(replay.__file__),replay.CONTRACT,replay_prepare):
        old.merge(bindings,{str(f.resolve()):old.sha(f)})
    old.merge(bindings,replay_binding['closure'])
    for name, digest in report['artifacts'].items():
        require(Path(name).name==name and old.sha(OUTPUT/name)==digest,'producer saved artifact binding')
        old.merge(bindings,{str((OUTPUT/name).resolve()):digest})
    for f,h in old.primitives.helper.SEALED_HELPERS.items():old.merge(bindings,{str(f):h})
    for f in old.primitives.physical_files():old.merge(bindings,{str(f.resolve()):old.sha(f)})
    old.merge(bindings,data['bindings'])
    native,h=old.primitives.helper.native_constructor_contract();old.merge(bindings,{native:h})
    return dict(schema=NAME+'.prepare',plans=plans,contract=contract(),bindings=bindings,
                receipt_sha256=RECEIPT_SHA,report_sha256=REPORT_SHA,
                completion_sha256=COMPLETION_SHA,timing_sha256=TIMING_SHA,
                source_prepare_id=old.source.digest(data['plan']), dispatch_allowed=False,
                status='saved_inputs_verified_not_physical_execution')


def prepare(output):
    p=build(); output=Path(old.absolute(str(output))); raw=old.raw(p)
    require(len(raw)<=old.PREPARE_CAP and not output.exists() and shutil.disk_usage(output.parent).free>=old.FLOOR+len(raw), 'prepare bound')
    with output.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    fd=os.open(output.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(prepared=str(output),prepare_sha256=old.sha(output))


def verify(path,expected):
    path=Path(old.absolute(str(path)))
    require(old.sha(path)==expected,'physical prepare SHA')
    p=old.read(path)
    require(p==build() and old.sha(path)==expected,'full source and saved receipt rebuild')
    return p


def execute(p,data,store,factory,*,authorization,execution_binding=None):
    require(authorization==AUTH and p['contract']==contract() and
            [(x['root_slot'],x['arm']) for x in p['plans']]==list(SLOTS),'authorized six fixed plans')
    for plan in p['plans']:
        require(plan['root_sha256']==data['references'][plan['root_slot']]['root_sha256'] and
                plan['selection_digest']==old.source.digest({k:v for k,v in plan.items() if k!='selection_digest'}), 'root/plan seal')
        old.kernel.commands(plan['commands']);producer.core.old.vec(plan['target'])
    rows=old.rows_for(p);ops=old.Ops(p,store)
    r=dict(schema=NAME+'.report',status='running',rows=rows,errors=[],contract=p['contract'],
           sourcepack_sha256=SOURCEPACK_SHA,source_prepare_id=p['source_prepare_id'],fixed_branch_denominator=6,
           prepared=None if execution_binding is None else execution_binding['prepared'],
           prepare_sha256=None if execution_binding is None else execution_binding['prepare_sha256'],
           new_neural_calls=0,training=False,scientific_admission=False,P1_admission=False)
    try:
        store.emit('run_started',prepare=p,execution_binding=execution_binding)
        for i,(plan,row) in enumerate(zip(p['plans'],rows,strict=True)):
            env=None;holder=[];ops.where=dict(branch=i,root_slot=plan['root_slot'],arm=plan['arm'],selection_digest=plan['selection_digest'])
            def construct():
                row['constructor_internal_work']='unknown';value=factory();holder.append(value);return value
            try:
                env=ops.call('environment_construct',construct);old.kernel.moving.validate_env(env.unwrapped);body=env.unwrapped
                require(all(getattr(body,k,None) is None for k in ('space','render_buffer','latest_action')) and
                        body.render_action is False and body.render_size==224 and body.render_mode=='rgb_array','pristine target-free constructor')
                row.update(status='running',constructor_internal_work='sealed_native_pristine_no_physics',
                           tracking_goal=data['references'][plan['root_slot']]['goal'].tolist())
                with old.kernel.native_hooks(env,ops,row):
                    old.kernel.branch(env,data['references'][plan['root_slot']],data['plan'],plan,row,ops,old.BranchStore(store,row,plan),i)
            except BaseException as exc:old.kernel.retain(row,exc,'setup_or_branch')
            finally:
                if env is None and holder:env=holder[0]
                if env is not None:old.primitives.close_env(env,row,ops,store)
                steps=[x for x in ops.receipts if x.get('branch')==i and x['resource']=='public_step']
                row['own_attempted']=sum(x.get('phase')=='plan' for x in steps)
                row['own_returned']=sum(x.get('phase')=='plan' and x['status']=='returned' for x in steps)
                row['warmup_returned']=sum(x.get('phase')=='warmup' and x['status']=='returned' for x in steps)
                row['full_horizon']=row['own_returned']==10
                if row['status']=='failed':row['endpoint_scope']='failed_partial'
                store.emit('branch_closed',receipt=row)
    except BaseException as exc:r['errors'].append(dict(stage='execution',type=type(exc).__name__,message=str(exc)))
    try:
        for f,h in p['bindings'].items():require(old.sha(f)==h,'final source drift')
    except BaseException as exc:r['errors'].append(dict(stage='source',type=type(exc).__name__,message=str(exc)))
    clean=not r['errors'] and all(x['status'] in ('horizon_completed','native_success','native_stopped') for x in rows)
    if clean:
        try:
            c=ops.counts;steps=sum(x['warmup_returned']+x['own_returned'] for x in rows)
            require(all(x['attempted']==x['completed'] for x in c.values()) and
                    c['environment_construct']['completed']==c['environment_close']['completed']==6 and
                    c['reset']['completed']==12 and c['set_state']['completed']==24 and c['goal_render']['completed']==12 and
                    c['goal_bind']['completed']==6 and c['public_step']['completed']==steps and
                    c['physics_step']['completed']==24+10*steps and c['explicit_render']['completed']==sum(x['PNG'] for x in rows), 'native resource census')
        except BaseException as exc:r['errors'].append(dict(stage='census',type=type(exc).__name__,message=str(exc)));clean=False
    r.update(status='six_slot_native_recorded' if clean else 'partial_or_failed',counts=ops.counts,
             actual_operation_receipts=ops.receipts,full_horizons=sum(x['full_horizon'] for x in rows),
             native_successes=sum(x['status']=='native_success' for x in rows),engineering_failures=sum(x['status']=='failed' for x in rows))
    journal_failed=False
    try:
        store.emit('run_finished',status=r['status'],counts=ops.counts,errors=r['errors'])
        r.update(journal_sha256=old.sha(store.path/'journal.jsonl'),journal_chain_tail=store.previous)
    except BaseException as exc:
        r['errors'].append(dict(stage='journal',type=type(exc).__name__,message=str(exc)));r['status']='partial_or_failed';journal_failed=True
    try:store.put('report.json',r,final=True)
    except BaseException as exc:
        r['errors'].append(dict(stage='report',type=type(exc).__name__,message=str(exc)));r['status']='partial_or_failed'
        if hasattr(store,'emergency'):store.emergency('report_publication_failed')
    else:
        if journal_failed and hasattr(store,'emergency'):store.emergency('journal_publication_failed')
    return r


def validate_operation_ledger(events,p,report):
    """Independently authenticate nested operation intents, returns and caps."""
    pending={};terminal={};stack=[]
    counts={k:dict(attempted=0,completed=0) for k in p['contract']['caps']}
    for event in events:
        kind=event['kind']
        if kind not in ('operation_dispatch_intent','operation_returned','operation_failed'):continue
        r=event['receipt'];i=r['operation_index'];resource=r['resource']
        require(type(i) is int and resource in counts,'operation identity')
        if kind=='operation_dispatch_intent':
            require(i==len(pending) and r['status']=='pending','sequential intent')
            pending[i]=r;stack.append(i)
        else:
            require(stack and stack.pop()==i and i in pending and i not in terminal and
                    r['status']==('returned' if kind=='operation_returned' else 'failed'),'nested return')
            original={k:v for k,v in r.items() if k not in ('error_type','message')}
            require(dict(original,status='pending')==pending[i],'same original intent')
            terminal[i]=r
    require(not stack and len(terminal)==len(pending) and
            report['actual_operation_receipts']==[terminal[i] for i in range(len(terminal))], 'complete operation journal')
    for r in terminal.values():
        i=r['branch'];require(type(i) is int and 0<=i<6,'six branch identity')
        plan=p['plans'][i]
        require(all(r[k]==plan[k] for k in ('root_slot','arm','selection_digest')),'bound operation plan')
        counts[r['resource']]['attempted']+=1
        counts[r['resource']]['completed']+=r['status']=='returned'
    require(counts==report['counts'] and all(c['attempted']<=p['contract']['caps'][k] for k,c in counts.items()),'count/cap')
    return list(terminal.values())


def verify_journal(path,expected,tail):
    require(old.sha(path)==expected,'journal file SHA')
    data=Path(path).read_bytes();require(data.endswith(b'\n'),'complete journal')
    events=[];previous=None
    for i,line in enumerate(data.splitlines()):
        event=producer.core.old.decode(line)
        require(event['index']==i and event['previous_sha256']==previous and
                event['event_sha256']==old.source.digest({k:v for k,v in event.items() if k!='event_sha256'}), 'journal chain')
        previous=event['event_sha256'];events.append(event)
    require(previous==tail,'journal terminal')
    return events


def validate_saved(p, data, report, events, output):
    """Validate all six saved branches, native action/state and rendered bytes."""
    from PIL import Image
    from experiments.spectral_reachability import agent_experiment1_common_controller_physical_consume_20261028_v1 as prior
    require(report['schema']==NAME+'.report' and report['contract']==p['contract']==contract() and
            report['sourcepack_sha256']==SOURCEPACK_SHA and report['source_prepare_id']==p['source_prepare_id'] and
            report['fixed_branch_denominator']==6 and report['training'] is False and
            report['new_neural_calls']==0 and report['scientific_admission'] is report['P1_admission'] is False,
            'six-slot report contract')
    require(events and events[0]['kind']=='run_started' and events[0]['prepare']==p and
            events[0]['execution_binding']==dict(prepared=report['prepared'],prepare_sha256=report['prepare_sha256']) and
            events[-1]['kind']=='run_finished' and events[-1]['status']==report['status'] and
            events[-1]['counts']==report['counts'] and events[-1]['errors']==report['errors'] and
            events[-1]['event_sha256']==report['journal_chain_tail'],'run boundaries')
    rows=report['rows'];require(len(rows)==len(p['plans'])==6 and
        [(x['root_slot'],x['arm']) for x in rows]==list(SLOTS),'fixed six rows')
    ops=validate_operation_ledger(events,p,report)
    require([e['receipt'] for e in events if e['kind']=='branch_closed']==[x for x in rows if x['status']!='not_started'],'durable rows')
    names={'execution_binding.json','journal.jsonl','report.json'};checkpoints=[];artifact_hashes={}
    for i,(row,plan) in enumerate(zip(rows,p['plans'],strict=True)):
        require(row['branch_index']==i and row['selected_plan']==plan and row['selection_digest']==plan['selection_digest']==
                old.source.digest({k:v for k,v in plan.items() if k!='selection_digest'}),'selected plan exact')
        ev=[e for e in events if e.get('branch')==i]
        steps=[x for x in ops if x['branch']==i and x['resource']=='public_step']
        own=[x for x in steps if x.get('phase')=='plan'];warm=[x for x in steps if x.get('phase')=='warmup']
        require(row['own_attempted']==len(own) and row['own_returned']==sum(x['status']=='returned' for x in own) and
                row['warmup_returned']==sum(x['status']=='returned' for x in warm) and
                row['full_horizon']==(row['own_returned']==10),'actual step counts')
        for group,commands in ((own,plan['commands']),(warm,p['contract']['warmup_raw'])):
            require(len(group)<=10 and [x['step'] for x in group]==list(range(1,len(group)+1)) and
                    all(x['raw_action']==commands[j] for j,x in enumerate(group)),'literal contiguous actions')
        observations=[e['receipt'] for e in ev if e['kind']=='fresh_observation']
        tracking=[e['receipt'] for e in ev if e['kind']=='tracking_observation']
        require(row['PNG']==len(observations) and row['tracking']==tracking,'frame/tracking census')
        for cp in observations:
            name=cp['png'];require(Path(name).name==name and name not in names and old.sha(output/name)==cp['file_sha256'],'PNG file bytes')
            with Image.open(output/name) as image:
                require(image.size==(224,224) and image.mode=='RGB','PNG RGB ABI');pixels=np.asarray(image)
            require(old.kernel.moving.image_sha(pixels)==cp['pixel_sha256'],'RGB pixel bytes');names.add(name);artifact_hashes[name]=cp['file_sha256']
            guards=[e for e in ev if e['kind']=='guard_receipt' and e['phase']==cp['phase'] and e['step']==cp['step']]
            require(len(guards)==1,'one actual guard')
            g=guards[0];state=prior.vector(cp['state'])
            require(next(e['index'] for e in ev if e['kind']=='fresh_observation' and e['receipt']==cp)<g['index'], 'observation precedes guard')
            accepted=(int(np.count_nonzero(np.all(pixels==[65,105,225],axis=-1)))>0 and
                      bool(((state[:4]>=0)&(state[:4]<=512)).all()) and
                      np.array_equal(state,prior.vector(cp['observation_state'])) and g['render_preserved_dynamics'])
            require(g['canonical_agent_pixels']==int(np.count_nonzero(np.all(pixels==[65,105,225],axis=-1))) and
                    g['accepted']==bool(accepted),'guard arithmetic')
        planobs=[x for x in observations if x['phase']=='plan']
        require([x['step'] for x in tracking]==[x['step'] for x in planobs],'all tracking slots')
        for cp,tr in zip(planobs,tracking,strict=True):
            require(tr==dict(step=cp['step'],png=cp['png'],pixel_sha256=cp['pixel_sha256'],state=cp['state'],
                native=prior.metric(row['tracking_goal'],cp['state']),target_sha256=producer.core.old.core.digest(plan['target']),
                model_prefix_target_cost=plan['prefix_target_cost'],model_terminal_cost=plan['model_terminal_cost'],
                observed_latent192=None,latent_tracking_error=None,
                model_successor=plan['predicted_contexts'][cp['step']//5-1] if cp['step'] in (5,10) else None),'tracking exact')
        transitions=[e for e in ev if e['kind']=='returned_transition']
        for cp in observations:
            if cp['step']>0:
                matched=[e for e in transitions if e['phase']==cp['phase'] and e['step']==cp['step']]
                require(len(matched)==1 and cp['state']==matched[0]['state'],'every frame matches returned state')
        for phase,commands,returned in (('warmup',p['contract']['warmup_raw'],row['warmup_returned']),('plan',plan['commands'],row['own_returned'])):
            group=[e for e in transitions if e['phase']==phase]
            require([e['step'] for e in group]==list(range(1,returned+1)),'returned transitions complete')
            for e in group:
                if phase=='plan':
                    goal=row['goal'];attached=[x for x in ev if x['kind']=='goal_attached']
                    require(len(attached)==1 and e['command']==commands[e['step']-1] and e['returned_goal_state']==goal and
                            e['returned_goal_rgb_sha256']==attached[0]['goal_rgb_sha256'],'plan action and goal')
                else:
                    placements=[x for x in ev if x['kind']=='placement_request'];require(len(placements)==1,'placement')
                    goal=placements[0]['original_goal']
                metric=prior.metric(goal,e['state'])
                require(type(e['terminated']) is bool and type(e['truncated']) is bool and
                        e['terminated']==metric['native_success'] and e['reward']==-metric['native_distance'],'native transition metric')
        ref=data['references'][plan['root_slot']]
        if row['initial_exact']:
            require(old.kernel.moving.exact(np.asarray(row['root_state']),ref['states'][-1]) and
                    old.kernel.moving.exact(np.asarray(row['goal']),ref['goal']) and
                    row['goal']==row['tracking_goal'] and row['initial_metrics']==prior.metric(row['goal'],row['root_state']) and
                    old.source.root_hash(row['root_state'],row['goal'])==plan['root_sha256'],'native source root/goal')
            for phase,step,j in [('warmup',0,0),('warmup',5,1),('warmup',10,2),('plan',0,2)]:
                matches=[x for x in observations if x['phase']==phase and x['step']==step]
                require(len(matches)==1 and old.kernel.moving.exact(np.asarray(matches[0]['state']),ref['states'][j]) and
                        matches[0]['snapshot']==ref['dynamics'][j],'source state/snapshot')
                with Image.open(output/matches[0]['png']) as image: rgb=np.asarray(image)
                require(old.kernel.moving.exact(rgb,ref['history'][j]),'source RGB exact')
            attached=[e for e in ev if e['kind']=='goal_attached']
            require(len(attached)==1 and attached[0]['goal']==row['goal'] and
                    attached[0]['goal_rgb_sha256']==old.kernel.moving.image_sha(ref['goal_rgb']),'source goal RGB')
        for cp in row['checkpoints']:
            require(cp in planobs and cp['step'] in (0,5,10),'real checkpoint')
            guards=[e for e in ev if e['kind']=='guard_receipt' and e['phase']=='plan' and e['step']==cp['step']]
            require(len(guards)==1 and guards[0]['accepted'] is True,'checkpoint guard')
            owntrans=[e for e in transitions if e['phase']=='plan']
            require(cp['state']==(row['root_state'] if cp['step']==0 else owntrans[cp['step']-1]['state']), 'checkpoint transition state')
            checkpoints.append(dict(branch=i,root_slot=plan['root_slot'],arm=plan['arm'],checkpoint=cp))
        if row['status'] in ('horizon_completed','native_success','native_stopped'):
            require(row['initial_exact'] and [x['step'] for x in planobs]==list(range(row['own_returned']+1)) and
                    [x['step'] for x in row['checkpoints']]==[x for x in (0,5,10) if x<=row['own_returned']],'complete actual frames')
            require(all(e['accepted'] is True for e in ev if e['kind']=='guard_receipt'),'all clean guards')
        if row['status']=='horizon_completed':
            require(row['full_horizon'] and row['terminated'] is False and row['truncated'] is False,'full horizon status')
        if row['status']=='native_success':
            require(row['native_success'] is True and row['terminated'] is True and row['truncated'] is False,'success status')
        if row['final_state'] is not None:
            owntrans=[e for e in transitions if e['phase']=='plan']
            require(owntrans and row['final_state']==owntrans[-1]['state'] and
                    row['final_metrics']==prior.metric(row['goal'],row['final_state']) and
                    row['native_success']==row['final_metrics']['native_success'] and row['terminated']==owntrans[-1]['terminated'] and
                    row['truncated']==owntrans[-1]['truncated'],'final native state')
        for e in ev:
            if e['kind']=='invalid_render' and e['receipt'].get('raw_render'):
                cp=e['receipt'];name=cp['raw_render'];require(Path(name).name==name and name not in names and old.sha(output/name)==cp['file_sha256'],'invalid render bytes');names.add(name);artifact_hashes[name]=cp['file_sha256']
    require({x.name for x in output.iterdir()}==names,'exact output file census')
    require(all(old.sha(output/name)==h for name,h in artifact_hashes.items()),'final physical artifact hashes')
    require(report['full_horizons']==sum(x['full_horizon'] for x in rows) and
            report['native_successes']==sum(x['status']=='native_success' for x in rows) and
            report['engineering_failures']==sum(x['status']=='failed' for x in rows),'fixed denominator outcomes')
    require(report['status'] in ('six_slot_native_recorded','partial_or_failed'),'known report status')
    if report['status']=='six_slot_native_recorded':
        c=report['counts'];steps=sum(x['warmup_returned']+x['own_returned'] for x in rows)
        require(not report['errors'] and all(x['status'] in ('horizon_completed','native_success','native_stopped') for x in rows) and
                all(x['attempted']==x['completed'] for x in c.values()) and
                c['environment_construct']['completed']==c['environment_close']['completed']==6 and
                c['reset']['completed']==12 and c['set_state']['completed']==24 and c['goal_render']['completed']==12 and
                c['goal_bind']['completed']==6 and c['public_step']['completed']==steps and
                c['physics_step']['completed']==24+10*steps and c['explicit_render']['completed']==sum(x['PNG'] for x in rows),
                'clean native census')
    return checkpoints


def run(prepared,expected,output,factory,*,authorization):
    require(authorization==AUTH,'explicit MAIN physical authorization')
    p=verify(prepared,expected);data=source_data()
    store=Store(output,p['contract']['wall_seconds'])
    binding=dict(prepared=old.absolute(str(prepared)),prepare_sha256=expected)
    try:
        store.put('execution_binding.json',binding)
        return execute(p,data,store,factory,authorization=authorization,execution_binding=binding)
    except BaseException:
        if not (store.path/'emergency.json').exists():store.emergency('uncaught_run_failure')
        raise


def consume(output,expected):
    output=Path(old.absolute(str(output)));old.primitives.hash_literal(expected)
    files=list(output.iterdir());require(all(x.is_file() and not x.is_symlink() for x in files) and
        sum(x.stat().st_size for x in files)<=old.CAP and old.sha(output/'report.json')==expected,'bounded saved report')
    report=old.read(output/'report.json',old.CAP);p=verify(report['prepared'],report['prepare_sha256'])
    require(old.read(output/'execution_binding.json')==dict(prepared=report['prepared'],prepare_sha256=report['prepare_sha256']),'execution binding')
    events=verify_journal(output/'journal.jsonl',report['journal_sha256'],report['journal_chain_tail'])
    checkpoints=validate_saved(p,source_data(),report,events,output)
    for path,h in p['bindings'].items():require(old.sha(path)==h,'source closure')
    require(old.sha(output/'report.json')==expected,'final report identity')
    return dict(schema=NAME+'.consume',status=report['status'],report_sha256=expected,
                rows=report['rows'],counts=report['counts'],checkpoints=checkpoints,
                fixed_branch_denominator=6,scientific_admission=False,P1_admission=False,
                evidence_boundary='saved physical events, source and images; no neural re-execution')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('prepare','verify','run','consume'))
    for name in ('prepared','prepare-sha','output','report-sha','receipt'):ap.add_argument('--'+name)
    ap.add_argument('--authorize-physics',action='store_true');a=ap.parse_args()
    if a.command=='prepare':result=prepare(a.output)
    elif a.command=='verify':verify(a.prepared,a.prepare_sha);result=dict(status='verified_no_execution')
    elif a.command=='run':
        require(a.authorize_physics,'MAIN authorization required')
        def factory():
            import gymnasium as gym
            import stable_worldmodel
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False,render_action=False)
        report=run(a.prepared,a.prepare_sha,a.output,factory,authorization=AUTH)
        report_path=Path(a.output)/'report.json';emergency=Path(a.output)/'emergency.json'
        result=dict(status=report['status'],report=str(report_path) if report_path.exists() else None,
                    report_sha256=old.sha(report_path) if report_path.exists() else None,
                    emergency=str(emergency) if emergency.exists() else None)
        print(old.source.canonical(result),flush=True)
        require(report['status']=='six_slot_native_recorded','partial physical run saved')
        return
    else:
        receipt=consume(a.output,a.report_sha);path=Path(old.absolute(a.receipt));data=old.raw(receipt)
        with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        result=dict(receipt=str(path),receipt_sha256=old.sha(path),status=receipt['status'])
    print(old.source.canonical(result),flush=True)


if __name__=='__main__':main()
