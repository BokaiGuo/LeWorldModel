"""Actual cp5 RGB suffix recompile; MAIN HOST CUDA only, fixed72/32.

Frozen numerical core + exact frozen midpoint authority, no outcome selection.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import inspect
import math
import os
from pathlib import Path
import time
import numpy as np
from experiments.spectral_reachability import agent_nonlinear_suffix_observed_recompile_20261028_v1 as numeric
from experiments.spectral_reachability import agent_nonlinear_suffix_midpoint_rgb_20261028_v1 as midpoint

NAME='agent_nonlinear_suffix_observed_recompile_20261028_v2'
AUTH='MAIN_OBSERVED_SUFFIX4_RECOMPILE72_HOST_CUDA_ONLY'
F=numeric.frozen
core,signal,transport=F.core,F.signal,F.transport
ROOT=F.ROOT
require,sha,digest,raw=core.require,signal.sha,core.digest,core.canonical
PINNED=((numeric,'231bf17d98f4e5dcb61845a8bff4bed1ebbc108b58f6fbeddb722c340ce5b1df'),
    (F,'0a3e475af9271a0769a3829b4e714fa4e2d6f00eaed38ec8c3ea5d7b9a163fbc'),
    (midpoint,'b1371a0de4d805bac1b4b64cfb625b7d327a3772d61d68c092804430f94a7ece'))
ARGS={'midpoint_output','midpoint_report_sha','midpoint_receipt','midpoint_receipt_sha'}
BOOL_FIELDS={'production_authority_verified','postcast_norm_is_diagnostic_only','parent_prediction_used_as_new_prediction',
    'P1_admission','freshness','physical_release','scientific_admission'}


def contract():
    c=deepcopy(numeric.contract())
    c.update(schema=NAME+'.contract',stage='production_adapter_requires_actual_frozen_midpoint_strict',
        production_ABI_bound=True,production_ready='only complete actual4 cp5 histories',production_run_authorized=False,
        per_slot=dict(score_forward=9,gradient_forward=8,winner_replay=1,predict=18,scalar_reverse=8),
        total=dict(score_forward=36,gradient_forward=32,winner_replay=4,predict=72,scalar_reverse=32),independent_replay=4,
        cost_scope='actual native per-operation ledgers; all upstream encoding/physical/compiler costs inherited separately',
        wall_seconds=1800,output_bytes=128*1024**2,final_reserve_bytes=16*1024**2,disk_floor_bytes=200*1024**2,
        cuda_allocated_bytes=3*1024**3,cuda_reserved_bytes=4*1024**3,
        CUDA_only=True,TF32=False,autocast=False,SDP='MATH',retries=0)
    return c


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def scalar(x,name,nonnegative=True):
    require(type(x) in (int,float) and math.isfinite(x) and (not nonnegative or x>=0),'literal finite numeric scalar: '+name)
    return x


def integer(x,name,upper):require(type(x) is int and 0<=x<upper,'literal integer: '+name)


def typed_tree(v,key=''):
    """JSON scalar types before NumPy coercion or Python bool==1 equality."""
    if type(v) is dict:
        require(all(type(k) is str for k in v),'literal JSON object keys')
        for k,x in v.items():typed_tree(x,k)
    elif type(v) is list:
        for x in v:typed_tree(x,key)
    elif type(v) is bool:require(key in BOOL_FIELDS,'bool masquerading as numeric: '+key)
    elif type(v) in (int,float):scalar(v,key,False)
    else:require(v is None or type(v) is str,'literal JSON numeric/context types')


def validate_request(q):
    typed_tree(q);numeric.validate_request(q)
    integer(q['slot'],'slot',4);integer(q['root_slot'],'root_slot',2)
    inc=q['incumbent'];integer(inc['parent_iteration'],'parent_iteration',9);integer(inc['parent_prefix_index'],'parent_prefix_index',42)
    domain=inc['original_ridge_domain']
    core.keys(domain,('solver_float64_norm','postcast_float64_norm','cap','postcast_norm_is_diagnostic_only','original_raw_bits'))
    for k in ('solver_float64_norm','postcast_float64_norm','cap'):scalar(domain[k],k)
    require(domain['cap']==F.CAP and domain['solver_float64_norm']<=F.CAP
        and domain['postcast_norm_is_diagnostic_only'] is True,'original ridge legality (postcast norm diagnostic ONLY)')
    return q['task']


def parent_incumbent(c,r):
    typed_tree(c);typed_tree(r)
    integer(c['index'],'original prefix',42);integer(r['index'],'parent prefix',42)
    integer(r['winner_iteration'],'parent winner',9)
    for i,v in enumerate(r['iterates']):
        integer(v['iteration'],'parent iterate',9);require(v['iteration']==i,'parent iterate order');scalar(v['Q'],'parent Q')
    scalar(r['selected_Q'],'parent selected Q')
    return numeric.selected_incumbent(c,r)


def check_pins():
    for module,h in PINNED:require(sha(module.__file__)==h,'frozen source identity: '+module.__name__)


def material(p,r,accepted,compiled,parent,load):
    """Join ONLY exact real report ABI, then whitelist model-only requests.

    Caller must first run midpoint.consume and compare its entire saved receipt.
    Native states/outcomes in upstream rows are never copied into optimizer inputs.
    """
    require(p['schema']==midpoint.NAME+'.prepare' and p['contract']==midpoint.contract()
        and r['schema']==midpoint.NAME+'.report' and r['contract']==p['contract']
        and accepted['schema']==midpoint.NAME+'.consume' and accepted['status']=='actual_midpoint_suffix4_verified_not_admitted'
        and accepted['fixed_denominator']==4 and accepted['known_roots']==2 and accepted['fresh_roots']==0,
        'exact original actual midpoint consumer ABI, four slots/two known roots')
    require(len(p['rows'])==len(r['rows'])==len(accepted['rows'])==len(compiled['slots'])==len(parent['finals'])==4
        and len(parent['rows'])==168 and compiled['schema']==F.NAME+'.prepare' and parent['schema']==F.NAME+'.report',
        'full4 cp5 and complete original168 producer')
    authority=p['physical_authority']
    require(authority['producer']==F.NAME and authority['complete_all168'] is True and authority['full_source_build_verified'] is True
        and authority['persisted_receipt_equal'] is True and authority['selection_uses_physics'] is False,
        'real complete original library strict, no forged/observed selector receipt')
    rows=[]
    for i,(spec,row,end,s,final) in enumerate(zip(p['rows'],r['rows'],accepted['rows'],compiled['slots'],parent['finals'],strict=True)):
        for obj,key in ((spec,'index'),(row,'index'),(end,'index'),(s,'index'),(final,'slot')):
            integer(obj[key],key,4);require(obj[key]==i,'exact slot position')
        require(all((x['root_slot'],x['condition'])==numeric.ORDER[i] for x in (spec,row,end,s)),'same known root/renderer condition')
        plan=final['plan'];task=core.validate_task(s['task']);typed_tree(task)
        require(final['status']=='sealed' and plan['commands']==spec['full10']==row['full10'] and task['goal']==spec['goal192']
            and spec['model_task_binding']==task['binding'] and spec['goal_sha256']==task['binding']['goal_sha256']
            and spec['action_mean']==task['action_mean'] and spec['action_std']==task['action_std'],
            'same original full10/model binding/goal192/normalizer, no goal replacement')
        integer(plan['prefix_index'],'selected prefix',42);integer(plan['iterate_index'],'selected iterate',9)
        j=plan['prefix_index'];c=s['candidates'][j];old=parent['rows'][42*i+j]
        require(old['slot']==i and old['prefix_index']==j and old['status']=='sealed','actual selected prefix result from all168')
        inc=parent_incumbent(c,old['result'])
        require(inc['parent_iteration']==plan['iterate_index'] and inc['prefix']+inc['commands']==plan['commands']
            and plan['selection_digest']==digest([F.NAME,task['binding'],old['result']])
            and spec['selected_index']==j and spec['goal192']==task['goal'],'exact selected parent logits/actions/witness')
        base=dict(index=i,root_slot=numeric.ORDER[i][0],condition=numeric.ORDER[i][1],parent_selection_digest=plan['selection_digest'],
            parent_plan_sha256=digest(plan),midpoint_row_binding_sha256=spec['row_binding_sha256'],request=None,
            status='missing_actual_history',missing_history=deepcopy(spec['missing_history']),
            observation_provenance=[None if f is None else {k:deepcopy(f[k]) for k in
                ('phase','step','path','file_sha256','pixel_sha256','condition')} for f in spec['frames']],
            original_compiler_task_sha256=digest(task),parent_candidate_sha256=digest(c),parent_result_sha256=digest(old['result']))
        require(type(spec['eligible']) is bool and row['eligible'] is spec['eligible'],'literal actual-history eligibility')
        if not spec['eligible']:
            require(row['actual_full3'] is None and row['prediction'] is None and spec['missing_history'],'missing stays null, fixed denominator')
            rows.append(base);continue
        require(row['status']=='predicted' and not spec['missing_history'],'genuine complete actual cp5 history')
        full=load(row['actual_full3']);inp={k:load(v) for k,v in row['feedback']['inputs'].items()};tokens=midpoint.tokens(spec)
        require(full.dtype==np.float32 and full.shape==(1,3,192) and np.isfinite(full).all(),'actual frozen RGB full3 tensor')
        require(set(inp)=={'context',*tokens} and np.array_equal(inp['context'],full),'complete causal actual feedback inputs')
        for k,v in tokens.items():require(inp[k].dtype==v.dtype and np.array_equal(inp[k],v),'same original saved raw/token input '+k)
        root_raw=np.asarray(midpoint.producer.moving.warmup_commands(),np.float32)
        actual_raw=np.concatenate((root_raw,np.asarray(inc['prefix'],np.float32)))[-10:]
        observed=dict(task['root_context'],latent_history=full[0].tolist(),action_history=inp['incoming'][0].tolist())
        q=numeric.request(i,task,observed,root_raw.tolist(),actual_raw.tolist(),inc,c,old['result']);validate_request(q)
        pred=load(row['prediction']);require(pred.dtype==np.float32 and pred.shape==(1,3,192) and np.isfinite(pred).all(),'original same-suffix native prediction3')
        expected_context=dict(observed,latent_history=observed['latent_history'][1:]+[pred[0,-1].tolist()],action_history=tokens['outgoing'][0,-2:].tolist())
        require(F.cost(task['goal'],expected_context)==end['conditioned_final_Q'],'original actual-mid same-suffixQ last-row semantic')
        base.update(status='ready',request=q,observed_incumbent_prediction=expected_context,observed_incumbent_Q=end['conditioned_final_Q'])
        rows.append(base)
    return rows


def build(midpoint_output,midpoint_report_sha,midpoint_receipt,midpoint_receipt_sha):
    check_pins();args=dict(midpoint_output=midpoint_output,midpoint_report_sha=midpoint_report_sha,
        midpoint_receipt=midpoint_receipt,midpoint_receipt_sha=midpoint_receipt_sha)
    require(set(args)==ARGS,'actual explicit midpoint arguments')
    for k,v in args.items():
        if k.endswith('_sha'):core.hash_value(v)
        else:midpoint.producer.mixed.absolute(v)
    out=Path(midpoint_output);receipt=Path(midpoint_receipt)
    require(sha(out/'report.json')==midpoint_report_sha and sha(receipt)==midpoint_receipt_sha,'actual external report and receipt SHA')
    accepted=midpoint.consume(out,midpoint_report_sha)
    require(raw(accepted)==raw(midpoint.read(receipt,64*1024**2)),'entire persisted midpoint strict receipt reconstructed')
    report=midpoint.read(out/'report.json',64*1024**2);p=midpoint.read(report['prepared'],64*1024**2)
    require(sha(report['prepared'])==report['prepare_sha256'],'actual midpoint prepare')
    a=p['physical_authority'];compiled=midpoint.read(a['producer_prepared'],16*1024**2)
    parent=midpoint.read(Path(a['producer_output'])/'report.json',64*1024**2)
    for f,h in ((a['producer_prepared'],a['producer_prepare_sha256']),
        (Path(a['producer_output'])/'report.json',a['producer_report_sha256']),
        (a['producer_receipt'],a['producer_receipt_sha256'])):
        require(sha(f)==h==p['bindings'][str(f)],'exact inherited original compiler/strict authority')
    rows=material(p,report,accepted,compiled,parent,lambda ref:midpoint.load_saved_array(out,ref))
    bindings=deepcopy(p['bindings'])
    def merge(add):
        for f,h in add.items():require(f not in bindings or bindings[f]==h,'input/source hash conflict');bindings[f]=h
    merge({str(out/'report.json'):midpoint_report_sha,str(receipt):midpoint_receipt_sha,report['prepared']:report['prepare_sha256']})
    merge({str(out/n):h for n,h in report['artifact_sha256'].items()})
    for module,h in PINNED:merge({str(Path(module.__file__).resolve()):h})
    for f in own_files():merge({str(f):sha(f)})
    # Actual installed classes omitted by old v1 are sealed here too, no skip.
    merge({v['path']:v['sha256'] for v in F.installed_class_sources().values()})
    F.check_bindings(bindings)
    return dict(schema=NAME+'.prepare',arguments=args,contract=contract(),rows=rows,bindings=bindings,
        policy=p['policy'],expected_model_hash=p['expected_model_hash'],midpoint_authority=dict(report_sha256=midpoint_report_sha,
            receipt_sha256=midpoint_receipt_sha,whole_strict_reconstructed=True),original_physical_authority=deepcopy(a),
        inherited_resources=dict(before_midpoint=p['inherited_resources'],midpoint=accepted['counts']),
        planned_new_compute=F.caps(72,32),actual_new_compute=None,production_ready=all(r['status']=='ready' for r in rows),
        P1_admission=False,freshness=False,physical_release=False)


def verify(prepared,expected):
    core.hash_value(expected);require(sha(prepared)==expected,'actual own prepare SHA');p=signal.read(prepared,32*1024**2)
    require(raw(p)==raw(build(**p['arguments'])) and sha(prepared)==expected,'whole real midpoint/parent/source preparation reconstructed');return p


def write(path,value):
    path=Path(path).absolute();require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_'),'own exclusive metadata prefix')
    # Frozen writer enforces all original bytes/fsync/caps; no globals patched.
    payload=raw(value);require(not path.exists() and not path.is_symlink() and path.resolve()==path and len(payload)<=32*1024**2,'new bounded metadata')
    import shutil
    require(shutil.disk_usage(path.parent).free>=contract()['disk_floor_bytes']+len(payload),'disk floor')
    with path.open('xb') as f:require(f.write(payload)==len(payload),'full metadata');f.flush();os.fsync(f.fileno())
    signal.fsync_dir(path.parent);return dict(path=str(path),sha256=sha(path),status=value.get('status','prepared'))


class Store(F.Store):
    def __init__(self,path,started):
        import shutil
        self.path=Path(path).absolute();self.started=started;self.finalizing=False
        require(self.path.parent==ROOT/'results' and self.path.name.startswith(NAME+'_') and self.path.resolve()==self.path
            and not self.path.exists() and shutil.disk_usage(self.path.parent).free>=contract()['output_bytes']+contract()['disk_floor_bytes'],'exclusive own output')
        self.path.mkdir();signal.fsync_dir(self.path.parent)
    def check(self,size=4096):
        import shutil
        require(self.finalizing or time.monotonic()-self.started<contract()['wall_seconds'],'whole1800s including verify boundary watchdog')
        used=sum(f.stat().st_size for f in self.path.rglob('*') if f.is_file())
        require(used+size<=contract()['output_bytes']-(0 if self.finalizing else contract()['final_reserve_bytes'])
            and shutil.disk_usage(self.path).free>=contract()['disk_floor_bytes']+size,'output128MiB/reserve16MiB/floor200MiB')


class Backend(F.Backend):
    """Identical native implementation; explicit labels for frozen auditor ABI."""
    def step(self,c,a,label):return super().step(c,a,label.removeprefix('observed_'))
    def gradient(self,c,a,label):return super().gradient(c,a,label.removeprefix('observed_'))


def strict_numeric(q,r,events):
    validate_request(q);typed_tree(r);typed_tree(events)
    integer(r['slot'],'result slot',4);integer(r['winner_iteration'],'new winner',9)
    require(raw(r['scheduled_cost'])==raw(numeric.contract()['per_slot']),'literal numeric planned costs (not booleans)')
    scalar(r['new_suffix_Q'],'newQ');scalar(r['observed_incumbent_Q'],'incumbentQ')
    for i,v in enumerate(r['iterates']):
        integer(v['iteration'],'iterate',9);require(v['iteration']==i,'all9 ordered iterates');scalar(v['Q'],'iterateQ')
    for d in r['gradients']:scalar(d['native_float32_loss'],'native float32 scalar loss')
    return numeric.verify_numeric(q,r,events)


def compile_row(spec,backend,journal):
    q=spec['request'];validate_request(q);numeric_events=[]
    journal.event(dict(kind='compile_intent',request_sha256=digest(q),midpoint_row_binding_sha256=spec['midpoint_row_binding_sha256']))
    def sink(e):
        typed_tree(e);numeric_events.append(deepcopy(e));journal.event(dict(kind='numeric_event',event=e))
    r=numeric.optimize(q,backend,sink);strict_numeric(q,r,numeric_events)
    require(r['iterates'][0]['context']==spec['observed_incumbent_prediction'] and
        r['observed_incumbent_Q']==spec['observed_incumbent_Q'],'original actual-cp5 same-suffix prediction exactly reproduced before replacement')
    journal.event(dict(kind='winner_replay_intent',iteration=r['winner_iteration'],commands=r['commands'][5:]))
    after=backend.step(q['observed_context'],r['commands'][5:],'winner_replay')
    require(after==r['selected_context'],'independent selected suffix H1 replay exact')
    plan=dict(schema=NAME+'.plan',root_slot=spec['root_slot'],condition=spec['condition'],slot=spec['index'],
        task_id=q['task']['binding']['task_id'],model_task_binding=deepcopy(q['task']['binding']),
        commands=deepcopy(r['commands']),raw_float32_bits=F.bits(r['commands']),prefix_bits=q['incumbent']['prefix_bits'],
        parent_selection_digest=spec['parent_selection_digest'],parent_plan_sha256=spec['parent_plan_sha256'],
        parent_result_sha256=q['parent_result_sha256'],parent_candidate_sha256=q['parent_candidate_sha256'],
        parent_prediction_not_new_claim=True,observed_context_sha256=digest(q['observed_context']),
        goal_sha256=digest(q['task']['goal']),new_suffix_Q=r['new_suffix_Q'],observed_incumbent_Q=r['observed_incumbent_Q'],
        new_terminal_context=after,new_witness_sha256=digest(r),winner_iteration=r['winner_iteration'],
        midpoint_row_binding_sha256=spec['midpoint_row_binding_sha256'],physical_release=False,P1_admission=False,freshness=False)
    plan['selection_digest']=digest(plan)
    journal.event(dict(kind='compile_return',numeric_sha256=digest(r),plan_sha256=digest(plan)))
    return dict(numeric=r,numeric_events=numeric_events,plan=plan)


def schedule(q,r):
    c=q['observed_context'];v=r['iterates'];out=[('iterate_0',c,v[0]['commands'])]
    for i in range(1,9):out.extend([(f'gradient_{i}',c,v[i-1]['commands']),(f'iterate_{i}',c,v[i]['commands'])])
    out.append(('winner_replay',c,r['commands'][5:]));return out


def audit_row(spec,result,receipt,events):
    core.keys(result,('numeric','numeric_events','plan'))
    q=spec['request'];r=result['numeric'];strict_numeric(q,r,result['numeric_events'])
    require(r['iterates'][0]['context']==spec['observed_incumbent_prediction'] and r['observed_incumbent_Q']==spec['observed_incumbent_Q'],'actual-mid incumbent agreement')
    control={'compile_intent','numeric_event','winner_replay_intent','compile_return'}
    native_events=[e for e in events if e['kind'] not in control]
    starts,ends,linked=F.audit_native(q['task'],receipt,native_events,schedule(q,r))
    require(all(e['kind'] not in ('candidate_intent','candidate_selection','candidate_return','library_selection') for e in native_events),'no legacy algorithm events masquerading as observed compiler')
    indices={id(e):i for i,e in enumerate(events)}
    actual_control=[e for e in events if e['kind'] in control]
    wanted=[dict(kind='compile_intent',request_sha256=digest(q),midpoint_row_binding_sha256=spec['midpoint_row_binding_sha256'])]
    wanted += [dict(kind='numeric_event',event=e) for e in result['numeric_events']]
    wanted += [dict(kind='winner_replay_intent',iteration=r['winner_iteration'],commands=r['commands'][5:]),
        dict(kind='compile_return',numeric_sha256=digest(r),plan_sha256=digest(result['plan']))]
    require(raw(actual_control)==raw(wanted) and events[0]==wanted[0] and events[-1]==wanted[-1],'all actual observed compiler event ABI/bookends')
    controls=[e for e in events if e['kind']=='numeric_event']
    require(indices[id(controls[0])]<indices[id(starts[0])],'request intent before any forward')
    # Each score is returned before its numeric iterate; each numeric AD intent
    # brackets the associated actual forward+reverse and model derivative output.
    for i in range(9):
        phase=0 if i==0 else 2*i;it=next(e for e in controls if e['event']['kind']=='observed_iterate' and e['event']['iterate']['iteration']==i)
        require(indices[id(ends[phase])]<indices[id(it)],'actual score returns before numeric iterate')
        require(ends[phase]['context']==r['iterates'][i]['context'],'actual native score context equals saved iterate')
        if i<8:
            gi=next(e for e in controls if e['event']['kind']=='observed_gradient_intent' and e['event']['iteration']==i+1)
            gr=next(e for e in controls if e['event']['kind']=='observed_gradient_return' and e['event']['iteration']==i+1)
            gphase=f'f{2*i+1:03}';ad=[e for e in native_events if e['kind'].startswith('AD_API_') and e['phase']==gphase]
            expected_ad=dict(phase=gphase,scalar_outputs=1,input_shape=[5,2],retain_graph=False,create_graph=False)
            require(raw(ad)==raw([dict(kind='AD_API_attempt',**expected_ad),dict(kind='AD_API_completed',**expected_ad)]),
                'exact runtime scalar reverse ABI, not vector or retained graph')
            require(indices[id(it)]<indices[id(gi)]<indices[id(starts[2*i+1])]
                and indices[id(ad[-1])]<indices[id(gr)]<indices[id(starts[2*i+2])],'numeric AD intent/actual return before next score')
            d=r['gradients'][i];require(ends[2*i+1]['context']==d['context'],'fresh gradient native context')
            op=next(x for x in receipt['events'] if x['resource']=='grad' and x['scope']['forward']==gphase)
            a,b=linked[op['index']];core.keys(a['arrays'],('grad:loss','grad:raw'));core.keys(b['arrays'],('grad:output',))
            loss,leaf,grad=(signal.unblob(x) for x in (a['arrays']['grad:loss'],a['arrays']['grad:raw'],b['arrays']['grad:output']))
            cbop=next(x for x in receipt['events'] if x['resource']=='raw_backward' and x['scope']['forward']==gphase)
            cb=signal.unblob(linked[cbop['index']][0]['arrays']['value'])
            require(loss.shape==(1,) and leaf.shape==grad.shape==cb.shape==(5,2) and float(loss[0])==d['native_float32_loss']
                and np.array_equal(leaf,r['iterates'][i]['commands']) and np.array_equal(grad,d['raw_gradient']) and np.array_equal(cb,grad),
                'strict runtime scalar transport[1]/rawleaf/actual gradient/callback, no vector objective')
    selected=controls[-1];replay=actual_control[-2]
    require(indices[id(ends[-2])]<indices[id(selected)]<indices[id(replay)]<indices[id(starts[-1])]
        and ends[-1]['context']==r['selected_context'],'selection after ALL9 actual scores, before independent winner replay')
    plan=result['plan'];body={k:v for k,v in plan.items() if k!='selection_digest'}
    require(plan['selection_digest']==digest(body) and plan['commands']==r['commands'] and F.bits(plan['commands'][:5])==q['incumbent']['prefix_bits']
        and plan['new_suffix_Q']==r['new_suffix_Q'] and plan['observed_incumbent_Q']==r['observed_incumbent_Q']
        and plan['new_terminal_context']==r['selected_context'] and plan['new_witness_sha256']==digest(r),'new commands/context/Q/witness indivisible')
    # Reconstruct exact plan through a side-effect-free sink/callback replay of
    # saved numeric outputs? No: enforce explicit complete expected metadata.
    expected=dict(schema=NAME+'.plan',root_slot=spec['root_slot'],condition=spec['condition'],slot=spec['index'],
        task_id=q['task']['binding']['task_id'],model_task_binding=q['task']['binding'],commands=r['commands'],raw_float32_bits=F.bits(r['commands']),
        prefix_bits=q['incumbent']['prefix_bits'],parent_selection_digest=spec['parent_selection_digest'],parent_plan_sha256=spec['parent_plan_sha256'],
        parent_result_sha256=q['parent_result_sha256'],parent_candidate_sha256=q['parent_candidate_sha256'],parent_prediction_not_new_claim=True,
        observed_context_sha256=digest(q['observed_context']),goal_sha256=digest(q['task']['goal']),new_suffix_Q=r['new_suffix_Q'],
        observed_incumbent_Q=r['observed_incumbent_Q'],new_terminal_context=r['selected_context'],new_witness_sha256=digest(r),
        winner_iteration=r['winner_iteration'],midpoint_row_binding_sha256=spec['midpoint_row_binding_sha256'],physical_release=False,P1_admission=False,freshness=False)
    expected['selection_digest']=digest(expected);require(raw(plan)==raw(expected),'complete exact new plan provenance/metadata')
    return plan


def roster():return [dict(index=i,status='not_started',result=None,resources=None) for i in range(4)]


def dispatch(p,store,rows,check,factory):
    require(len(p['rows'])==len(rows)==4 and p['production_ready'] is True,'complete actual cp5 all4 before any compile')
    for spec,row in zip(p['rows'],rows,strict=True):
        require(row['index']==spec['index'] and spec['status']=='ready','all4 slot-major ready')
        row['status']='attempted';ns=f'r{row["index"]}'
        journal=transport.NativeJournal(store,ns,check);ledger=core.Ledger(F.caps(18,8),sink=journal.event)
        try:
            b=factory(spec['request']['task'],journal,ledger)
            with b.observe():row['result']=compile_row(spec,b,journal)
            require(ledger.attempted==ledger.completed==F.caps(18,8),'18 actual native forwards/8reverse per slot')
            row['status']='sealed'
        finally:
            row['resources']=ledger.receipt();journal.close()
            row.update(journal_file=ns+'/events.jsonl',journal_sha256=sha(journal.path),journal_tail=journal.tail)
            if row['status']!='sealed':row['status']='failed_partial'
            store.put(ns+'_result.json',row)


def precision(torch):
    value=dict(matmul_TF32=torch.backends.cuda.matmul.allow_tf32,cudnn_TF32=torch.backends.cudnn.allow_tf32,
        autocast=torch.is_autocast_enabled('cuda'),math=torch.backends.cuda.math_sdp_enabled(),
        flash=torch.backends.cuda.flash_sdp_enabled(),efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),cudnn_sdp=torch.backends.cuda.cudnn_sdp_enabled())
    require(value==dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,math=True,flash=False,efficient=False,cudnn_sdp=False),'original exact MATH precision')
    return value


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit HOST CUDA authorization before any IO')
    started=time.monotonic();p=verify(prepared,expected);require(p['production_ready'] is True,'missing cp5 history keeps4 denominator; no incomplete production')
    store=Store(output,started);rows=roster();errors=[];model=None;before=after=host=None;flags=None;source_ok=False;pb=pa=None;peaks=None
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        def check():
            store.check();precision(torch)
            require(torch.cuda.memory_allocated(device)<=contract()['cuda_allocated_bytes']
                and torch.cuda.memory_reserved(device)<=contract()['cuda_reserved_bytes'],'CUDA allocated/reserved caps')
        with torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            check();model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False);check()
            before=signal.model_hash(model);require(before['original_state_dict']==p['expected_model_hash'],'original checkpoint model tensor identity')
            for module in model.modules():require(str(Path(inspect.getfile(type(module))).resolve()) in p['bindings'],'all presealed loaded model class sources, no builtin skip')
            pb=precision(torch);dispatch(p,store,rows,check,lambda task,journal,ledger:Backend(model,device,task,journal,ledger,check));pa=precision(torch)
            torch.cuda.synchronize(device);check()
    except BaseException as exc:errors.append(dict(stage='run',error=repr(exc)))
    finally:
        store.finalizing=True
        try:
            if model is not None:
                midpoint.bridge.native.frozen(model);after=signal.model_hash(model);require(before==after,'all model parameters/persistent/nonpersistent buffers unchanged')
                peaks=[torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)]
            F.check_bindings(p['bindings']);require(sha(prepared)==expected,'own prepare unchanged');source_ok=True
        except BaseException as exc:errors.append(dict(stage='closure',error=repr(exc)))
        wall=time.monotonic()-started
        if wall>=contract()['wall_seconds']:errors.append(dict(stage='wall',error='whole-stage cap'))
        if flags is not None:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
        complete=not errors and source_ok and before==after and before is not None and all(r['status']=='sealed' for r in rows)
        artifacts={str(f.relative_to(store.path)):sha(f) for f in store.path.rglob('*') if f.is_file()}
        report=dict(schema=NAME+'.report',status='observed_suffix4_recompiled_not_admitted' if complete else 'partial_failclosed',
            contract=contract(),prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,errors=errors,
            fixed_denominator=4,known_roots=2,source_verified=source_ok,source_manifest_sha256=digest(p['bindings']),
            model_before=before,model_after=after,model_unchanged=before is not None and before==after,host=host,
            precision_before=pb,precision_after=pa,CUDA_peak_bytes=peaks,wall_seconds=wall,artifact_sha256=artifacts,
            P1_admission=False,scientific_admission=False,freshness=False,physical_release=False)
        store.put('report.json',report,final=True)
    return dict(status=report['status'],report_sha256=sha(store.path/'report.json'),output=str(store.path),errors=errors)


def consume(output,expected):
    output=Path(output).absolute();core.hash_value(expected);require(sha(output/'report.json')==expected,'external actual observed-recompile report SHA')
    r=signal.read(output/'report.json',32*1024**2);p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['contract']==contract() and len(r['rows'])==r['fixed_denominator']==4 and r['known_roots']==2
        and all(r[k] is False for k in ('P1_admission','scientific_admission','freshness','physical_release')),'honest all4 denominator/gates')
    require(signal.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'actual own execution binding')
    files={str(f.relative_to(output)) for f in output.rglob('*') if f.is_file()}
    require(files==set(r['artifact_sha256'])|{'report.json'} and sum(f.stat().st_size for f in output.rglob('*') if f.is_file())<=contract()['output_bytes'],'whole output census/cap')
    for name,h in r['artifact_sha256'].items():
        path=Path(name);require(not path.is_absolute() and all(v not in ('..','.') for v in path.parts)
            and (output/path).resolve()==output/path and sha(output/path)==h,'complete artifact path/hash/no aliases')
    receipts=[];plans=[]
    for i,(spec,row) in enumerate(zip(p['rows'],r['rows'],strict=True)):
        integer(row['index'],'row index',4);require(row['index']==i==spec['index'],'fixed all4 positions')
        if row['resources'] is None:require(row['status']=='not_started' and row['result'] is None,'unreached keeps null');continue
        require(signal.read(output/f'r{i}_result.json',16*1024**2)==row,'durable row equality')
        events=F.load_journal(output,row);transport.audit_resources(row['resources'],events);receipts.append(row['resources'])
        if row['status']=='sealed':plans.append(audit_row(spec,row['result'],row['resources'],events))
        else:require(row['status']=='failed_partial','unknown completed status rejected')
    counts={kind:{k:sum(q[kind][k] for q in receipts) for k in core.AXES} for kind in ('attempted','completed')}
    complete=r['status']=='observed_suffix4_recompiled_not_admitted' and not r['errors'] and len(plans)==4
    if complete:
        require(counts['attempted']==counts['completed']==F.caps(72,32),'all72 actual forward/32same reverse accounted')
        require(r['source_verified'] is r['model_unchanged'] is True and r['model_before']==r['model_after']
            and r['model_before']['original_state_dict']==p['expected_model_hash'] and r['source_manifest_sha256']==digest(p['bindings']),'complete model/source closure')
        require(r['precision_before']==r['precision_after']==dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,math=True,flash=False,efficient=False,cudnn_sdp=False),'actual original precision')
        require(0<=scalar(r['wall_seconds'],'wall')<contract()['wall_seconds'] and len(r['CUDA_peak_bytes'])==2
            and all(type(x) is int and 0<=x<=contract()[k] for x,k in zip(r['CUDA_peak_bytes'],('cuda_allocated_bytes','cuda_reserved_bytes'),strict=True)),'wall/CUDA caps')
    F.check_bindings(p['bindings']);require(sha(output/'report.json')==expected,'final output seal')
    return dict(schema=NAME+'.consume',status='observed_suffix4_verified_science_blocked' if complete else 'partial_failclosed',
        complete=complete,report_sha256=expected,fixed_denominator=4,known_roots=2,plans=plans if complete else [],rows=r['rows'],
        actual_new_compute=counts,inherited_resources=p['inherited_resources'],midpoint_authority=p['midpoint_authority'],
        original_physical_authority=p['original_physical_authority'],total_FLOPs=None,physical_metrics=None,
        P1_admission=False,scientific_admission=False,freshness=False,physical_release=False)


def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('midpoint-output','midpoint-report-sha','midpoint-receipt','midpoint-receipt-sha','output','prepared','prepare-sha','report-sha','receipt','authorization'):q.add_argument('--'+k)
    a=q.parse_args()
    if a.command=='prepare':v=write(a.output,build(a.midpoint_output,a.midpoint_report_sha,a.midpoint_receipt,a.midpoint_receipt_sha))
    elif a.command=='verify':v=dict(verified=True,production_ready=verify(a.prepared,a.prepare_sha)['production_ready'])
    elif a.command=='run':v=run(a.prepared,a.prepare_sha,a.output,a.authorization)
    else:require(a.receipt,'own durable saved receipt');v=write(a.receipt,consume(a.output,a.report_sha))
    print(raw(v).decode(),flush=True)


if __name__=='__main__':main()
