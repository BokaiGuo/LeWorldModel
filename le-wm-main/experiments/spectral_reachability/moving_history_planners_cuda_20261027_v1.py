"""Two explicit moving-history exports -> unchanged GN/D2P CUDA primitives.

Own schemas/loader/consume; never replace old module globals or call zero gates.
No model/physics work in prepare, verify, consume or RAM tests. MAIN CUDA only.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_constructive_blind_planners_cuda_20261027_v1 as blind
from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as moving

NAME='moving_history_planners_cuda_20261027_v1'
ROOT,ARMS=blind.ROOT,blind.ARMS
CAP,FLOOR,FINAL=blind.CAP,blind.FLOOR,blind.FINAL
BLIND_SHA='e56bee610f4eb412cc03fbe49b3df91054d0defe23d24b6fd9c77e6a7f1e9f27'
ENCODER_PREPARE_SHA='4386cf3babd40df0eb21abdf8bed52913596f012fb017ddb4986af96ddafa599'
LOADER_SHA='246b9194fd31c874d7f0da67c4dec17365f93294e6ed460fe46e2509673e1b63'
EXPORTS=[str(ROOT/f'results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_{s:02d}.json.gz') for s in (0,1)]
EXPORT_SHAS=['a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b',
             '101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74']
core=blind.encode.core
require,sha,read,canonical=blind.require,blind.sha,blind.read,blind.canonical
Store=blind.Store


def fixed_raw():return np.asarray([[.1,0]]*5+[[0,.1]]*5,np.float32)


def interface(value):
    core.fields(value,'gn_task d2p_packet q task_id','moving public planner interface')
    i=deepcopy(value);t=i['gn_task'];core.validate_task(t)
    require(t['environment']=='swm/PushT-v1' and np.array_equal(t['projection'],np.eye(192)) and
        t['packet_logit_cap']==float(.75*np.sqrt(5)) and
        np.array_equal(t['root_raw_action_history'],fixed_raw().astype(float)), 'actual fixed moving raw history, not zero/reordered/scaled')
    require(np.array_equal(t['action_mean'],blind.d2p.ACTION_MEAN.astype(float)) and
        np.array_equal(t['action_std'],blind.d2p.ACTION_STD.astype(float)) and
        np.array_equal(t['root_context']['action_history'],core.normalized_tokens(t,fixed_raw())), 'exact float32 public normalized two-packet history')
    require(t['model_sha256']==moving.MODELS[str(moving.POLICY/'weights.pt')] and
        t['model_config_sha256']==moving.MODELS[str(moving.POLICY/'config.json')], 'original model files')
    require(i['task_id']==core.task_id(t) and i['q']==t['projection'] and
        i['d2p_packet']==dict(root_context=t['root_context'],goal=t['goal']), 'same task/context/goal/Q both arms')
    z=np.asarray(t['root_context']['latent_history']+[t['goal']])
    require(np.array_equal(z,z.astype(np.float32)), 'native float32 latent exports')
    return i


def load_export(path,expected):
    core.hash_value(expected);path=Path(path).absolute()
    require(path.name.startswith('task_') and path.name.endswith(('.json','.json.gz')), 'explicit own task exports only')
    require(sha(path)==expected,'external moving task SHA')
    i=interface(moving.load_task(path,expected))  # Own loader, NEVER blind.interface/load_export.
    e=read(path,2*1024**2)
    require(e['schema']==moving.NAME+'.task_export' and type(e['source_slot']) is int and
        e['source_slot'] in (0,1) and e['task_id']==i['task_id'], 'own export schema/source slot')
    require(sha(path)==expected,'export final TOCTOU');return i


def contract():
    c=deepcopy(blind.contract(2))
    c.update(roots=2,history=dict(observed_steps=[0,5,10],raw=fixed_raw().astype(float).tolist(),
        normalized='float32 subtract/divide then two time-major 5x2 packets; never zero padding'),
        interface_source=moving.NAME,context_evidence='actual CUDA predictor full latent inputs plus action-encoder input tokens; every forward',
        GN_budget_clarification='78predict+78action_encoder/root; 156predict is TWO roots, not per root; no padding',
        global_freshness='reused recipes, NEW moving post-warmup tasks; not old zero-history roots',
        own_schema=True,P2_admission=False)
    return c


def build(task_exports,task_shas,loader_sha):
    core.hash_value(loader_sha)
    require(loader_sha==LOADER_SHA and sha(blind.__file__)==BLIND_SHA and sha(moving.__file__)==loader_sha,'frozen primitive and MAIN final moving loader SHA')
    require(len(task_exports)==len(task_shas)==2,'exactly two explicit ordered exports and external SHAs')
    require(list(task_shas)==EXPORT_SHAS,'MAIN actual fixed two export SHAs')
    slots=[]
    for s,(path,h) in enumerate(zip(task_exports,task_shas,strict=True)):
        path=Path(path).absolute();require(str(path)==EXPORTS[s],'MAIN fixed two explicit moving export paths')
        i=load_export(path,h);e=read(path)
        require(type(e['source_slot']) is int and e['source_slot']==s and
            e['task_contract']['prepare_sha256']==ENCODER_PREPARE_SHA,'fixed original moving slots0 then1 and actual encoder prepare')
        slots.append(dict(order=s,source_slot=s,task_export=str(path),task_sha256=h,task_id=i['task_id'],blind_sha256=e['blind_sha256']))
    for k in ('task_export','task_sha256','task_id','blind_sha256'):
        require(len({s[k] for s in slots})==2,'no duplicated task/export/blind identity')
    norm=blind.encode.native_metadata()  # Only frozen model/config/public constants, no old task loader.
    require(norm==moving.normalization(),'same frozen normalizer')
    files=set(blind.source_files()+moving.source_files()+[Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')])
    bindings={str(f.resolve()):sha(f) for f in files};bindings.update(moving.MODELS)
    bindings.update({s['task_export']:s['task_sha256'] for s in slots})
    for path,h in bindings.items():require(sha(path)==h,'source/input/model final TOCTOU')
    return dict(schema=NAME+'.prepare',arguments=dict(task_exports=[s['task_export'] for s in slots],task_shas=list(task_shas),loader_sha=loader_sha),
        slots=slots,contract=contract(),bindings_sha256=bindings,encoder_prepare_sha256=ENCODER_PREPARE_SHA,
        policy=str(moving.POLICY),expected_model_hash=blind.encode.MODEL_TENSOR_SHA)


def prepare(output,**arguments):
    p=build(**arguments);path=Path(output).absolute();raw=canonical(p)
    require(not path.exists() and len(raw)<256*1024 and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'NEW small prepare +200MiB floor')
    with path.open('xb') as f:require(f.write(raw)==len(raw),'short prepare');f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw),contract=p['contract'])


def verify(path,expected):
    core.hash_value(expected);require(sha(path)==expected,'external own prepare SHA');p=read(path)
    require(p==build(**p['arguments']),'own moving interface/source/task/model reconstruction');return p


def incoming(t,before,raw):
    core.context(before);token=core.normalized_tokens(t,np.asarray(raw)).reshape(1,10)
    return dict(latent_history=deepcopy(before['latent_history']),action_history=np.concatenate((before['action_history'],token)).tolist())


def advance(t,before,raw,after):
    core.context(after);want=incoming(t,before,raw)
    require(after['latent_history'][:2]==before['latent_history'][1:] and
        after['action_history']==want['action_history'][-2:],'exact actual float32 history/latent chronology')


@contextmanager
def context_trace(model,sink):
    """Read-only module pre-hooks; no forward/global replacement or new neural calls."""
    hooks=[];pending=[];index=0;primary=None
    def values(x,shape):
        require(getattr(getattr(x,'device',None),'type',None)=='cuda' and tuple(x.shape)==shape,'actual full CUDA context')
        a=x.detach().cpu().numpy();require(a.dtype==np.float32 and np.isfinite(a).all(),'native float32 context');return a[0].astype(float).tolist()
    def action(module,args):
        require(not pending,'one action-encoder input per predictor');pending.append(values(args[0],(1,3,10)))
    def predictor(module,args):
        nonlocal index
        require(len(pending)==1,'predictor requires actual action-encoder input')
        c=dict(latent_history=values(args[0],(1,3,192)),action_history=pending.pop())
        sink(dict(kind='native_context_attempt',index=index,context=c));index+=1
    try:
        hooks.append(model.action_encoder.register_forward_pre_hook(action));hooks.append(model.predictor.register_forward_pre_hook(predictor))
        yield
        require(not pending,'no unmatched context inputs')
    except BaseException as e:primary=e;raise
    finally:
        failures=[]
        for h in reversed(hooks):
            try:h.remove()
            except BaseException as e:failures.append(e)
        if failures:
            if primary is not None:primary.add_note('context hook cleanup: '+repr(failures))
            else:raise failures[0]


def history_check(arm,i,result,events):
    """Pure-array verification of ALL actual model inputs, not duplicate receipts."""
    t=i['gn_task'];initial=t['root_context'];expected=[]
    def step(before,raw,after):
        expected.append(incoming(t,before,raw));advance(t,before,raw,after)
    if arm==ARMS[0]:
        request=None;path=[];incumbent=None;adpath=[];evaluations=0;derivatives=0;completed=[]
        frozen=None;trials=[];chosen=None;updates=[]
        for e in events:
            k=e['kind']
            if k=='evaluation_request':
                require(request is None,'no overlapping evaluation');request=e;path=[]
                require(np.array_equal(e['commands'],np.tanh(np.asarray(e['logits'],np.float32)).reshape(10,2)),'each candidate raw/logit binding')
                if '_scale_' in e['evaluation_id']:
                    require(frozen is not None and len(trials)<5 and e['logits']==frozen['logits'][len(trials)],'all predeclared line proposals, no reselection')
            elif k=='native_packet':
                require(request is not None and e['packet']==len(path) and e['mode']==request['mode'],'all candidate packet order')
                n=len(path);step(initial if n==0 else path[-1],np.asarray(request['commands'])[5*n:5*n+5],e['context']);path.append(e['context'])
            elif k=='evaluation_complete':
                require(request is not None and len(path)==2,'full H2 candidate only')
                want=dict(task_id=i['task_id'],logits=request['logits'],commands=request['commands'],contexts=path,cost=core.model_cost(t,path[-1]['latent_history'][-1]))
                require(e['prediction']==want,'all candidate costs/contexts retained');completed.append(want)
                if '_scale_' in request['evaluation_id']:trials.append(want)
                evaluations+=1;request=None
            elif k in ('initial_verified','update_verified'):
                incumbent=e['incumbent'] if k=='initial_verified' else e['prediction'];core.check_prediction(t,incumbent)
                require(len(completed)>=3 and completed[-3]==completed[-2]==completed[-1] and incumbent['contexts']==completed[-1]['contexts'],'independent triple incumbent from actual candidate replays')
                if k=='update_verified':
                    require(chosen is not None and incumbent['contexts']==chosen['contexts'] and incumbent['commands']==chosen['commands'] and incumbent['cost']==chosen['cost'],'selected trial actually replayed')
            elif k=='derivative_complete':core.check_derivative(t,e['derivative'],incumbent)
            elif k=='line_proposals_frozen':
                require(e['iteration']==len(updates)+1 and e['incumbent_logits']==incumbent['logits'] and e['scales']==list(blind.gn.SCALES),'fixed GN proposal order/scales')
                endpoint=blind.gn.screen.projected_logits(np.asarray(incumbent['logits'])+np.asarray(e['delta']).reshape(2,10),t['packet_logit_cap'])
                require(e['endpoint']==endpoint.tolist() and e['logits']==[a.tolist() for a in blind.gn.line_trials(incumbent['logits'],endpoint,t['packet_logit_cap'])],'all original projected line candidates')
                frozen=e;trials=[]
            elif k=='selection_complete':
                index=blind.gn.select(incumbent,trials);chosen=([incumbent]+trials)[index]
                want=dict(iteration=len(updates)+1,incumbent_cost=incumbent['cost'],trial_costs=[c['cost'] for c in trials],selected_index=index,
                    selected_scale=None if index==0 else blind.gn.SCALES[index-1],selected_cost=chosen['cost'])
                require({key:e[key] for key in want}==want,'stable all-five-trial selection');updates.append(want)
            elif k=='jacobian_forward_packet':
                require(incumbent is not None and e['packet']==len(adpath),'full joint AD packet order')
                n=len(adpath);step(initial if n==0 else adpath[-1],np.asarray(incumbent['commands'])[5*n:5*n+5],e['context']);adpath.append(e['context'])
            elif k=='jacobian_forward_complete':
                require(len(adpath)==2 and e['terminal']==adpath[-1]['latent_history'][-1] and e['logits']==incumbent['logits'],'AD consumes original nonzero context')
                adpath=[];derivatives+=1
        require(request is None and not adpath and evaluations==35 and derivatives==4,'all35 evaluated H2 +4 AD H2')
        require(incumbent==result['final_prediction'] and updates==result['steps'],'final selected incumbent and all4 frozen updates')
    else:
        # Explicit primitive schedule, including root/suffix AD forwards and ALL replays.
        expected.extend([incoming(t,initial,np.zeros((5,2)))]*2)
        prefixes=[e for e in events if e['kind']=='prefix']
        require([e['index'] for e in prefixes]==list(range(10)),'all10 conditional prefixes')
        def replay(before,raw,terminal):
            a=np.asarray(raw).reshape(-1,5,2)
            for n,u in enumerate(a):
                if n==len(a)-1:after=terminal
                else:
                    after=dict(before,latent_history=before['latent_history'][1:]+[terminal['latent_history'][-2]],
                        action_history=incoming(t,before,u)['action_history'][-2:])
                step(before,u,after);before=after
        for c,e in zip(result['candidates'],prefixes,strict=True):
            index=c['index'];require(c['prefix_params']==result['bank']['params'][index] and
                c['suffix_params']==c['suffix_solve']['params'] and
                np.array_equal(c['prefix_commands'],np.tanh(np.asarray(c['prefix_params'],np.float32)).reshape(5,2)) and
                np.array_equal(c['suffix_commands'],np.tanh(np.asarray(c['suffix_params'],np.float32)).reshape(5,2)) and
                c['commands']==c['prefix_commands']+c['suffix_commands'] and
                c['nonlinear_terminal_cost']==core.model_cost(t,c['context']['latent_history'][-1]),'all conditional raw packet concatenations and same-Q terminal scores')
            after=e['context'];step(initial,c['prefix_commands'],after)
            require(after['latent_history'][-1]==c['predicted_prefix'],'conditional prefix identity')
            expected.extend([incoming(t,after,np.zeros((5,2)))]*2)
            step(after,c['suffix_commands'],c['context']);replay(initial,c['commands'],c['context'])
        w=result['winner'];replay(initial,w['commands'],result['whole_replay_context'])
        replay(initial,w['commands'],result['segmented_replay_context'])
        replays=[e for e in events if e['kind']=='replay' and e.get('completed') is True]
        require([e['phase'] for e in replays]==['candidate_'+str(n) for n in range(10)]+['winner_whole','winner_segment_0','winner_segment_1'],'all13 actual replay receipts')
        for c,e in zip(result['candidates'],replays[:10],strict=True):require(e['context']==c['context'] and e['commands']==c['commands'],'candidate actual replay')
        require(replays[10]['context']==result['whole_replay_context'] and replays[12]['context']==result['segmented_replay_context'],'winner actual replay')
        prefix=prefixes[w['index']]['context']
        require(replays[11]['context']==prefix and replays[10]['commands']==w['commands'] and
            replays[11]['commands']==w['prefix_commands'] and replays[12]['commands']==w['suffix_commands'],'winner whole/segmented literal actions and boundary')
        nominals=[e for e in events if e['kind']=='nominal'];derivative=[e for e in events if e['kind']=='derivative']
        phases=['root']+['suffix_'+str(n) for n in range(10)]
        require([e['phase'] for e in nominals]==[e['phase'] for e in derivative]==phases and
            [e['nominal'] for e in nominals]==[result['root_nominal']]+[c['nominal_terminal'] for c in result['candidates']] and
            [e['jacobian'] for e in derivative]==[result['root_jacobian']]+[c['jacobian'] for c in result['candidates']], 'every root/conditional nominal and derivative preserved')
    traces=[e for e in events if e['kind']=='native_context_attempt']
    require([e['index'] for e in traces]==list(range(blind.budgets()[arm]['predict'])) and
        [e['context'] for e in traces]==expected,'ALL native model-input latents/tokens match nonzero-root candidate/AD/replay chronology')
    return dict(native_contexts=len(traces),all_actual_nonzero_history_paths_verified=True)


def ledger_check(arm,events):
    """Reconcile original primitive attempt/return journals, no inferred refunds."""
    def pairs(attempts,returns,wanted):
        aa={e['index']:e for e in attempts};rr={e['index']:e for e in returns}
        require(len(aa)==len(attempts)==len(rr)==len(returns) and set(aa)==set(rr),'unique complete ledger attempts/returns')
        for k,e in aa.items():
            require(e['completed'] is False and rr[k]==dict(e,completed=True),'exact actual ledger return')
        require({resource:sum(e['resource']==resource for e in attempts) for resource in wanted}==wanted and
            all(e['resource'] in wanted for e in attempts),'exact resource journal census')
    if arm==ARMS[0]:
        pairs([e['event'] for e in events if e['kind']=='operation_attempted'],
            [e['event'] for e in events if e['kind']=='operation_completed'],blind.budgets()[arm])
    else:
        pairs([e['entry'] for e in events if e['kind']=='native_module_attempt'],
            [e['entry'] for e in events if e['kind']=='native_module_completed'],dict(predict=66,action_encoder=66))
        math=[e for e in events if e['kind'] in ('jacobian','svd')]
        norm=lambda e:dict(index=e['index'],resource=e['kind'],completed=e['completed'])
        pairs([norm(e) for e in math if e['completed'] is False],
            [norm(e) for e in math if e['completed'] is True],dict(jacobian=11,svd=12))
    pairs([e['entry'] for e in events if e['kind']=='AD_API_attempt'],
        [e['entry'] for e in events if e['kind']=='AD_API_completed'],contract()['AD_API_per_root'][arm])


def plan_all(p,store,runner,rows):
    for slot in p['slots']:
        i=load_export(slot['task_export'],slot['task_sha256']);require(i['task_id']==slot['task_id'],'moving task identity')
        for arm in ARMS:
            row=next(r for r in rows if r['order']==slot['order'] and r['arm']==arm);row['status']='attempted';events=[]
            def sink(e):
                events.append(deepcopy(e));store.event(dict(root_order=slot['order'],task_id=i['task_id'],arm=arm,event=e))
            sink(dict(kind='arm_started',task_sha256=slot['task_sha256'],root_context=i['gn_task']['root_context'],raw_history=fixed_raw().astype(float).tolist()))
            result=runner(arm,i,sink,row);chosen=blind.selected(arm,i,result)
            blind.exact_counts(row['resources'],blind.budgets()[arm]);blind.exact_counts(row['AD_API_resources'],contract()['AD_API_per_root'][arm])
            ledger_check(arm,events);row['history_verification']=history_check(arm,i,result,events)
            stem=f'root_{slot["order"]}_{arm}';store.put(stem+'.result.json.gz',result)
            plan=dict(schema=NAME+'.model_plan',source_slot=slot['source_slot'],order=slot['order'],arm=arm,
                blind_sha256=slot['blind_sha256'],task_export=slot['task_export'],task_sha256=slot['task_sha256'],
                task_contract_sha256=i['gn_task']['task_contract_sha256'],with_target=False,physical_execution=False,
                selection_uses_physics=False,P1_admission=False,**chosen)
            plan['selection_sha256']=core.digest(plan);store.put(stem+'.plan.json',plan)
            sink(dict(kind='model_selection_sealed',selection_sha256=plan['selection_sha256'],commands=plan['commands']))
            row.update(status='model_plan_sealed',result_file=stem+'.result.json.gz',plan_file=stem+'.plan.json',selection_sha256=plan['selection_sha256'])


def run(prepared,expected,output):
    p=verify(prepared,expected);store=Store(output);errors=[];model=None;before=after=host=None;source_ok=False
    rows=[dict(order=s['order'],source_slot=s['source_slot'],task_id=s['task_id'],arm=a,status='not_started',resources={},AD_API_resources={}) for s in p['slots'] for a in ARMS]
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        blind.encode.frozen(model);before=blind.encode.tensor_hash(model);require(before==p['expected_model_hash'],'same frozen model')
        def runner(arm,i,sink,row):
            with context_trace(model,sink):return blind.execute_arm(arm,i,model,device,torch,sink,row)
        with sdpa_kernel(SDPBackend.MATH):plan_all(p,store,runner,rows)
    except BaseException as e:
        errors.append(blind.encode.error('execution',e))
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as e:errors.append(blind.encode.error('error_journal',e))
    finally:
        for row in rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:blind.encode.frozen(model);after=blind.encode.tensor_hash(model)
            except BaseException as e:errors.append(blind.encode.error('final_model',e))
        try:verify(prepared,expected);source_ok=True
        except BaseException as e:errors.append(blind.encode.error('final_source',e))
        try:store.close()
        except BaseException as e:errors.append(blind.encode.error('journal_finalize',e))
        complete=not errors and source_ok and before is not None and before==after and all(r['status']=='model_plan_sealed' for r in rows)
        hashes={}
        try:files=list(store.output.iterdir())
        except BaseException as e:errors.append(blind.encode.error('artifact_listing',e));complete=False;files=[]
        for f in files:
            try:
                require(f.is_file(),'flat artifact census');hashes[f.name]=sha(f)
            except BaseException as e:errors.append(blind.encode.error('artifact_hash',e));complete=False
        result=dict(schema=NAME+'.report',status='moving_history_plans_sealed_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,errors=errors,first_error=errors[0] if errors else None,
            artifact_sha256=hashes,source_verified=source_ok,model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before is not None else None,
            host=host,contract=p['contract'],commands_released=complete,new_images=0,new_physics=0,training=False,P1_admission=False,P2_admission=False)
        try:store.put('report.json',result,final=True)
        except BaseException as e:errors.append(blind.encode.error('final_report',e));complete=False;print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial preserved; no retry: '+str(errors));return result


def consume(output,report_sha):
    """Own saved-array command release for a future NEW physical consumer only."""
    core.hash_value(report_sha);output=Path(output).absolute();require(sha(output/'report.json')==report_sha,'external own actual report SHA')
    r=read(output/'report.json');require(r['schema']==NAME+'.report' and r['status']=='moving_history_plans_sealed_not_admitted' and
        r['commands_released'] is True and r['errors']==[] and r['first_error'] is None and r['source_verified'] is True and r['model_unchanged'] is True,'whole run complete release')
    p=verify(r['prepared'],r['prepare_sha256']);require(r['contract']==p['contract'] and
        r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'],'same model/protocol')
    names={'binding.json','events.jsonl.gz'}|{f'root_{s}_{a}.{tail}' for s in (0,1) for a in ARMS for tail in ('result.json.gz','plan.json')}
    require(set(r['artifact_sha256'])==names and {f.name for f in output.iterdir()}==names|{'report.json'},'exact own artifact census')
    for n,h in r['artifact_sha256'].items():require(sha(output/n)==h,'artifact SHA')
    require(read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'own prepare binding')
    plain,eof=blind.inflate((output/'events.jsonl.gz').read_bytes());require(eof and plain.endswith(b'\n'),'complete journal')
    groups=[]
    for line in plain.splitlines():
        e=blind.decode(line);core.fields(e,'root_order task_id arm event','own scoped journal')
        require(type(e['root_order']) is int,'literal root order')
        key=(e['root_order'],e['arm'])
        if not groups or groups[-1][0]!=key:groups.append((key,[]))
        groups[-1][1].append(e)
    census=[(s,a) for s in (0,1) for a in ARMS]
    require([k for k,_ in groups]==[(r['order'],r['arm']) for r in r['rows']]==census,'all four ordered branches')
    plans=[]
    for row,((s,arm),group) in zip(r['rows'],groups,strict=True):
        slot=p['slots'][s];i=load_export(slot['task_export'],slot['task_sha256']);events=[e['event'] for e in group];stem=f'root_{s}_{arm}'
        require(row['status']=='model_plan_sealed' and row['task_id']==i['task_id'] and type(row['order']) is int and type(row['source_slot']) is int and row['source_slot']==s and
            row['result_file']==stem+'.result.json.gz' and row['plan_file']==stem+'.plan.json' and all(e['task_id']==i['task_id'] for e in group),'literal own branch identity')
        blind.exact_counts(row['resources'],blind.budgets()[arm]);blind.exact_counts(row['AD_API_resources'],contract()['AD_API_per_root'][arm])
        result=read(output/row['result_file']);chosen=blind.selected(arm,i,result)
        ledger_check(arm,events)
        require(row['history_verification']==history_check(arm,i,result,events),'actual nonzero input history verified again')
        require(events[0]==dict(kind='arm_started',task_sha256=slot['task_sha256'],root_context=i['gn_task']['root_context'],raw_history=fixed_raw().astype(float).tolist()),'literal original nonzero root')
        plan=read(output/row['plan_file']);ident=plan.pop('selection_sha256')
        wanted=dict(schema=NAME+'.model_plan',source_slot=s,order=s,arm=arm,blind_sha256=slot['blind_sha256'],task_export=slot['task_export'],task_sha256=slot['task_sha256'],
            task_contract_sha256=i['gn_task']['task_contract_sha256'],with_target=False,physical_execution=False,selection_uses_physics=False,P1_admission=False,**chosen)
        require(plan==wanted and core.digest(plan)==ident==row['selection_sha256'] and
            events[-1]==dict(kind='model_selection_sealed',selection_sha256=ident,commands=chosen['commands']),'own complete action selection seal')
        plans.append(dict(plan,selection_sha256=ident,whole_run_report_sha256=report_sha))
    require(sha(output/'report.json')==report_sha,'report final TOCTOU')
    for path,h in p['bindings_sha256'].items():require(sha(path)==h,'closure final TOCTOU')
    for n,h in r['artifact_sha256'].items():require(sha(output/n)==h,'artifact final TOCTOU')
    return dict(schema=NAME+'.consume',plans=plans,status='moving_commands_verified_not_admitted',new_physics=0,P1_admission=False,P2_admission=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run','consume'))
    p.add_argument('--task-exports',nargs=2);p.add_argument('--task-shas',nargs=2)
    for k in ('loader-sha','prepared','prepare-sha','output','report-sha'):p.add_argument('--'+k)
    a=p.parse_args()
    if a.command=='prepare':result=prepare(a.output,task_exports=a.task_exports,task_shas=a.task_shas,loader_sha=a.loader_sha)
    elif a.command=='verify':result=dict(status='verified_no_execution',roots=len(verify(a.prepared,a.prepare_sha)['slots']))
    elif a.command=='consume':result=consume(a.output,a.report_sha)
    else:result=run(a.prepared,a.prepare_sha,a.output)
    print(canonical(result).decode(),end='')


if __name__=='__main__':main()
