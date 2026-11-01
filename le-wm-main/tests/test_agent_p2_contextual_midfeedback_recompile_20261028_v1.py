"""RAM/native-interface doubles only; no neural, CUDA or real physics."""
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_contextual_midfeedback_recompile_20261028_v1 as m
import test_agent_p2_contextual_physical_rgb_encode_20261028_v1 as rgbtest
import test_agent_p2_contextual_composition_physical_handoff_20261028_v1 as phystest
import test_moving_history_observed_suffix_cuda_20261027_v1 as coretest


def inputs():
    p=rgbtest.prepared();spec=p['rows'][0]
    _,_,planner=rgbtest.fixture();plan=rgbtest.fixture()[0]['plans'][0]
    u=np.linspace(-.2,.25,20);commands=np.tanh(u).astype(np.float32).reshape(10,2)
    spec['full10']=commands.tolist();spec['raw_float32_bits']=m.physical.bits(commands.tolist())
    spec['suffix_bits']=commands[5:].tobytes().hex();plan['commands']=commands.tolist()
    spec['root_context'].update(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10)
    spec['frames'][1]['source']['raw_action_history10']=np.concatenate((m.kernel.moving.warmup_commands()[5:],commands[:5])).tolist()
    plan['encoder_identity'].update(action_mean=m.F.d2p.ACTION_MEAN.astype(float).tolist(),action_std=m.F.d2p.ACTION_STD.astype(float).tolist())
    return spec,plan,u


def branch():
    spec,plan,u=inputs();return m.branch(spec,plan,rgbtest.z(.25),rgbtest.z(.5),u)


def core_replay():
    b=branch();b,s,row,_,_=coretest.execute(b)
    row['predict_API_resources']=dict(predict=dict(attempted=5,completed=5))
    row['predict_API_ledger']=[dict(index=i,stage=stage,completed=True) for i,stage in enumerate(m.F.STAGES)]
    plain,eof=m.F.blind.inflate(s.files['events.jsonl.gz']);assert not eof  # open RAM journal, not a production receipt
    events=[m.F.blind.decode(x) for x in plain.splitlines()]
    full=[]
    for event in events:
        k=event['kind'];e=event.get('entry',{})
        if k=='module_attempt' and e['resource']=='predict':
            full.append(dict(kind='predict_API_attempt',entry=dict(index=m.F.STAGES.index(e['stage']),stage=e['stage'],completed=False)))
        if k=='math_return' and e['resource']=='jacobian':
            full += [dict(kind='AD_API_attempt',entry=dict(row['AD_API_ledger'][1],completed=False)),
                     dict(kind='AD_API_completed',entry=row['AD_API_ledger'][1]),
                     dict(kind='AD_API_completed',entry=row['AD_API_ledger'][0])]
        full.append(event)
        if k=='math_attempt' and e['resource']=='jacobian':
            full.append(dict(kind='AD_API_attempt',entry=dict(row['AD_API_ledger'][0],completed=False)))
        if k=='module_return' and e['resource']=='pred_proj':
            full.append(dict(kind='predict_API_return',entry=dict(index=m.F.STAGES.index(e['stage']),stage=e['stage'],completed=True)))
    return b,s,row,full


def test_exact_actual_context_original_prefix_domain_not_rounded_inverse():
    spec,plan,u=inputs();b=m.branch(spec,plan,rgbtest.z(.25),rgbtest.z(.5),u)
    assert b['prefix_commands']==plan['commands'][:5] and b['old_suffix_commands']==plan['commands'][5:]
    assert b['observed_context']['latent_history']==[spec['root_context']['latent_history'][1],rgbtest.z(.25),rgbtest.z(.5)]
    assert b['packet_logit_cap']==np.sqrt((.75*np.sqrt(10))**2-float(u[:10]@u[:10]))
    assert b['goal']==plan['goal'] and np.array_equal(b['q'],np.eye(192))
    assert not any(k in b for k in ('state','snapshot','reward','native_success','actual_terminal'))


@pytest.mark.parametrize('fault',['raw','logits','shape','radius'])
def test_wrong_prefix_identity_or_domain_blocks(fault):
    spec,plan,u=inputs()
    if fault=='raw':plan['commands'][0][0]=0.
    elif fault=='logits':u[0]+=.01
    elif fault=='shape':u=u[:10]
    else:
        u[:10]=1.;commands=np.tanh(u).astype(np.float32).reshape(10,2)
        plan['commands']=commands.tolist();spec['raw_float32_bits']=m.physical.bits(commands.tolist())
    with pytest.raises(ValueError):m.branch(spec,plan,rgbtest.z(.25),rgbtest.z(.5),u)


@pytest.fixture(scope='module')
def replay():return core_replay()


def test_original_H1_arrays_solve_all_five_calls_and_full_ABI_events(replay):
    b,s,row,events=replay;m.check_compile(b,row,lambda e:coretest.array(s,e));m.check_events(b,row,events)
    assert row['dispatch']=='new_suffix_unconditionally' and len(row['module_ledger'])==15
    assert len([e for e in events if e['kind']=='module_attempt'])==15


@pytest.mark.parametrize('fault',['unknown','drop_attempt','drop_return','swap','input','context','AD','extra_stage','drop_freeze'])
def test_complete_events_fail_closed(replay,fault):
    b,s,row,original=replay;events=deepcopy(original)
    if fault=='unknown':events.insert(1,dict(kind='not_a_native_ABI'))
    elif fault in ('drop_attempt','drop_return'):
        name='module_attempt' if fault=='drop_attempt' else 'module_return';events.remove(next(e for e in events if e['kind']==name))
    elif fault=='swap':
        i=next(i for i,e in enumerate(events) if e['kind']=='module_return');events[i],events[i-1]=events[i-1],events[i]
    elif fault=='input':next(e for e in events if e['kind']=='module_inputs_saved')['entry']['inputs']=[]
    elif fault=='context':next(e for e in events if e['kind']=='stage_started')['expected_input']['latent_history'][0][0]+=1
    elif fault=='AD':next(e for e in events if e['kind']=='AD_API_attempt')['entry']['is_grads_batched']=True
    elif fault=='extra_stage':events.insert(0,deepcopy(events[0]))
    else:events.remove(next(e for e in events if e['kind']=='new_suffix_frozen_before_evaluation'))
    with pytest.raises(ValueError):m.check_events(b,row,events)


@pytest.mark.parametrize('field',['resources','solve','goal','module_input','new_suffix'])
def test_saved_native_buffers_cost_and_solve_not_just_receipt(replay,field):
    b,s,original,_=replay;b=deepcopy(b);row=deepcopy(original)
    if field=='resources':row['resources']['predict']['completed']-=1
    elif field=='solve':row['solve']['params'][0]+=.01
    elif field=='goal':b['goal'][0]=1.
    elif field=='module_input':row['module_ledger'][3]['inputs']=[]
    else:row['new_suffix_commands'][0][0]+=.01
    with pytest.raises((ValueError,KeyError,IndexError)):m.check_compile(b,row,lambda e:coretest.array(s,e))


class GateRAM(m.ReplayGateStore,phystest.MemoryStore):
    """Production ReplayGateStore.emit runs atop the original RAM durable Store."""
    def __init__(self,midpoints):
        phystest.MemoryStore.__init__(self);self.midpoints=midpoints;self.matched=set();self.started=m.time.monotonic()


@pytest.fixture(scope='module')
def native_baseline():
    with pytest.MonkeyPatch.context() as patch:
        r,s,_,p,data=phystest.execute(patch);strict=phystest.assemble(r,s,p,data)
    return r,s,p,data,[row['rgb_encoding_inputs'][1] for row in strict['rows']]


def test_production_gate_emit_accepts_real_native_fields_then_strict_assemble(monkeypatch,native_baseline):
    _,_,p,data,mid=native_baseline;s=GateRAM(mid)
    r,s,envs,pp,dd=phystest.execute(monkeypatch,p=deepcopy(p),data=data,store=s)
    strict=phystest.assemble(r,s,pp,dd)
    assert s.matched==set(range(8)) and len(strict['rows'])==8
    assert r['counts']['public_step']==dict(attempted=160,completed=160)
    assert r['counts']['physics_step']==dict(attempted=1632,completed=1632)
    assert all(len(e.commands)==20 for e in envs)


@pytest.mark.parametrize('field',['file_sha256','pixel_sha256','state','snapshot'])
def test_mismatch_saved_mid_stops_all8_before_first_new_suffix(monkeypatch,native_baseline,field):
    _,_,p,data,mid=native_baseline;mid=deepcopy(mid)
    for f in mid:
        if field=='state':f['state'][0]+=1e-10
        elif field=='snapshot':f['snapshot']['state_float64_bits']='BAD'
        else:f[field]='0'*64
    s=GateRAM(mid);r,s,envs,pp,dd=phystest.execute(monkeypatch,p=deepcopy(p),data=data,store=s)
    assert r['status']=='partial_or_failed' and len(r['rows'])==8
    assert all(row['own_returned']==5 for row in r['rows'])
    assert all(len(e.commands)==15 and e.closed for e in envs)
    assert not s.matched


def test_step6_entry_requires_gate_even_no_selected_observation(native_baseline):
    _,_,_,_,mid=native_baseline;s=GateRAM(mid)
    with pytest.raises(ValueError,match='new suffix blocked'):
        s.emit('operation_attempted',resource='public_step',phase='plan',step=6,branch=0)
    assert s.events[-1]['kind']=='operation_attempted'  # durable intent is NOT execution


def test_AUTH_before_IO_no_neural_or_physical_import(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('must not verify'))
    monkeypatch.setattr(m,'physical_material',lambda *a:pytest.fail('must not read'))
    with pytest.raises(ValueError):m.compile_run('/NONE','0'*64,'/NONE','wrong')
    with pytest.raises(ValueError):m.run_physical('/NONE','0'*64,'/NONE',lambda:pytest.fail('factory'),'wrong')


def test_contract_counts_keep_all8_and_unknown_FLOPs():
    c=m.contract();assert c['plans']==8 and c['known_roots']==2 and c['total_compile']['predict']==40
    assert c['mathematical_VJP_directions']==1536 and c['total_FLOPs'] is None
    assert not c['equal_FLOPs'] and c['equal_execution_horizon']
    assert not c['P1_admission'] and not c['P2_admission'] and not c['physical_release']


def test_compile_to_physical_provenance_no_old_Q_or_graph_claim(replay):
    b,s,row,_=replay;_,original,_=inputs();old=deepcopy(original)
    p=m.seal_plan(b,row,original,'a'*64)
    assert original==old and 'predicted_Q' not in p and 'witness' not in p
    assert p['parent_predicted_Q']==old['predicted_Q'] and p['parent_witness']==old['witness']
    assert p['new_suffix_Q']==row['evaluations'][1]['cost']
    assert p['new_suffix_context']==row['evaluations'][1]['context']
    assert p['new_suffix_witness']['input_context']==b['observed_context']
    assert p['new_suffix_witness']['raw5']==p['commands'][5:]
    assert p['commands'][:5]==old['commands'][:5] and p['goal']==old['goal']
    assert p['raw_float32_bits']==m.physical.bits(p['commands'])
    assert p['selection_digest']==m.physical.saved.digest({k:v for k,v in p.items() if k!='selection_digest'})
    assert p['compiler_selection_digest']!=p['parent_compiler_selection_digest']


def test_new_suffix_not_selected_from_old_diagnostic_even_if_old_is_better():
    b=branch();reference=coretest.Backend(b).step(b['observed_context'],b['old_suffix_commands'])
    b['goal']=reference['latent_history'][-1];b['branch_id']=m.F.core.digest({k:v for k,v in b.items() if k!='branch_id'})
    b,s,row,_,_=coretest.execute(b)
    assert row['evaluations'][0]['cost']==0 and row['evaluations'][1]['cost']>0
    _,original,_=inputs();plan=m.seal_plan(b,row,original,'a'*64)
    assert plan['commands'][5:]==row['new_suffix_commands']
