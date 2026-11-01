"""RAM-only v2 ABI + native tensor/guard doubles; no model or physics."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_midfeedback_terminal_rgb_20261028_v2 as m
import test_agent_p2_midfeedback_terminal_rgb_20261028_v1 as prior


def fixture(missing=(),incumbents=(0,3,6)):
    plans,strict,previous=prior.fixture(missing)
    original=prior.old.fixture()[0]['plans']
    strict['schema']=m.intervention.NAME+'.physical_consume'
    for i,(plan,pair,old) in enumerate(zip(plans,strict['rows'],previous['rows'],strict=True)):
        index=0 if i in incumbents else 2
        if index==0:
            plan['commands'][5:]=deepcopy(original[i]['commands'][5:])
            plan['new_suffix_witness']['raw5']=deepcopy(plan['commands'][5:])
        sg=dict(incumbent_Q=plan['new_suffix_Q'] if index==0 else plan['new_suffix_Q']+.125,
            selected_Q=plan['new_suffix_Q'],selected_index=index,no_modeled_worsening=True,
            chart=m.intervention.contract()['chart'],domain=dict(legal=True))
        plan['safeguard']=sg;pair['safeguard']=deepcopy(sg);old['real_mid_same_suffix_predicted_Q']=sg['incumbent_Q']
        plan['raw_float32_bits']=m.physical.bits(plan['commands'])
        plan['raw_sha256']=m.hashlib.sha256(bytes.fromhex(plan['raw_float32_bits'])).hexdigest()
        plan['compiler_selection_digest']=prior.old.h(('RAM_v2_selected',i,index))
        plan['selection_digest']=m.physical.saved.digest({k:v for k,v in plan.items() if k!='selection_digest'})
        actual=pair['intervention'];actual['selection_digest']=plan['selection_digest'];actual['compiler_selection_digest']=plan['compiler_selection_digest']
        for f in actual['rgb_encoding_inputs']:
            f.update(compiler_selection_digest=plan['compiler_selection_digest'],raw_action_history10=deepcopy(plan['commands']),
                raw_action_history_bits=plan['raw_float32_bits'])
    return plans,strict,previous


def prepared(missing=(),incumbents=(0,3,6)):
    n=8-len(missing)
    return dict(schema=m.NAME+'.prepare',contract=m.contract(),rows=m.fixed_rows(*fixture(missing,incumbents),'/RAM_ONLY/v2physical'),
        expected_calls={k:n if k in ('encoder','projector') else 0 for k in m.bridge.KINDS})


def execute(monkeypatch,missing=(),incumbents=(0,3,6),tokens=257,forbidden=None):
    p=prepared(missing,incumbents);rows=deepcopy(p['rows']);s=prior.old.Memory();model=prior.Model(tokens);torch=prior.fake_torch();errors=[]
    monkeypatch.setattr(m.bridge,'load_pixels',prior.pixels)
    trace=m.bridge.Trace(model,torch,s,lambda:None,p['expected_calls'])
    guard=m.bridge.native.Ledger(2,lambda e:s.event(dict(kind='guard_event',event=e)),no_grad=lambda:True)
    def encode(pix,row,scope,ordinal):
        guard.image=ordinal
        with m.physical.saved.overlay.traced_scope(trace,scope,row):
            if forbidden=='predict':model.predict(None)
            if forbidden=='AD':torch.autograd.functional.jacobian(None)
            if forbidden=='action_encoder':model.action_encoder(None)
            pre=m.bridge.pixels.preprocess_pixels(pix)
            row['preprocessed']=trace.save(prior.Tensor(pre),f'r{row["index"]}_preprocessed')
            return model.projector(model.encoder(prior.Tensor(pre.reshape(1,3,224,224)))[:,0]).a[0].copy()
    with trace,m.bridge.native.guard(model,torch,guard,errors):m.process(p,rows,s,encode)
    assert not errors and not guard.blocked
    return p,dict(schema=m.NAME+'.report',rows=rows,contract=m.contract(),errors=errors,blocked=guard.blocked,
        modules=trace.entries,guards=guard.entries,counts=m.bridge.counts(trace.entries)),s


@pytest.fixture(scope='module')
def replay():
    with pytest.MonkeyPatch.context() as patch:return execute(patch)


def test_full8_selected_including_incumbents_no_dedup(replay,monkeypatch):
    p,r,s=replay;monkeypatch.setattr(m.bridge,'load_pixels',prior.pixels)
    rows=m.validate(p,r,s.events,s.load)
    assert len(rows)==8 and len(r['modules'])==len(r['guards'])==16
    assert r['counts']['encoder']==r['counts']['projector']==dict(attempted=8,completed=8)
    assert [v['index'] for v in rows if v['incumbent_selected']]==[0,3,6]
    assert all(v['new_suffix_predicted_Q']==v['safeguard']['selected_Q'] for v in rows)
    assert not any(v['physical_benefit_claim'] or v['model_safeguard_guarantees_actual_Q'] or
        v['model_safeguard_guarantees_prediction_ranking'] or v['model_safeguard_guarantees_native_task'] for v in rows)


def test_all8_incumbents_still_freshly_encode8(monkeypatch):
    p,r,s=execute(monkeypatch,incumbents=tuple(range(8)));rows=m.validate(p,r,s.events,s.load)
    assert all(row['incumbent_selected'] for row in rows) and r['counts']['encoder']['completed']==8
    assert len({row['latent']['file'] for row in r['rows']})==8


@pytest.mark.parametrize('missing',[(0,),(3,7),(0,3,6),tuple(range(8))])
def test_missing_keeps_null_all8_and_does_not_reuse_parent(monkeypatch,missing):
    p,r,s=execute(monkeypatch,missing=missing);rows=m.validate(p,r,s.events,s.load)
    assert len(rows)==8 and r['counts']['encoder']['completed']==8-len(missing)
    for i in missing:assert rows[i]['new_suffix_actual_Q'] is rows[i]['new_endpoint_MSE'] is None


@pytest.mark.parametrize('fault',['old_schema','guard','index','cost','incumbentQ','domain','chart','modeled_worse','plan_context','raw','drop'])
def test_v2_authority_and_selected_witness_mutations(fault):
    plans,s,rgb=fixture();p=plans[0];pair=s['rows'][0]
    if fault=='old_schema':s['schema']=m.core.intervention.NAME+'.physical_consume'
    elif fault=='guard':pair['safeguard']['selected_index']=4
    elif fault=='index':p['safeguard']['selected_index']=5;pair['safeguard']=deepcopy(p['safeguard'])
    elif fault=='cost':p['new_suffix_Q']+=1
    elif fault=='incumbentQ':rgb['rows'][0]['real_mid_same_suffix_predicted_Q']+=1
    elif fault=='domain':p['safeguard']['domain']['legal']=False;pair['safeguard']=deepcopy(p['safeguard'])
    elif fault=='chart':p['safeguard']['chart']='float32 tanh';pair['safeguard']=deepcopy(p['safeguard'])
    elif fault=='modeled_worse':p['safeguard']['incumbent_Q']=-1;pair['safeguard']=deepcopy(p['safeguard'])
    elif fault=='plan_context':p['new_suffix_context']['latent_history'][-1][0]+=1
    elif fault=='raw':pair['intervention']['rgb_encoding_inputs'][0]['raw_action_history_bits']='f'*160
    else:s['rows'].pop()
    with pytest.raises((ValueError,KeyError)):m.fixed_rows(plans,s,rgb,'/RAM_ONLY/v2physical')


@pytest.mark.parametrize('fault',['schema','contract','event','drop_attempt','pixels','197','native_output','selected','guard','count','missing','blocked'])
def test_shared_tensor_consumer_strict_complete_v2(replay,monkeypatch,fault):
    p,original,store=replay;p=deepcopy(p);r=deepcopy(original);events=deepcopy(store.events);s=prior.old.Memory();s.files=dict(store.files)
    monkeypatch.setattr(m.bridge,'load_pixels',prior.pixels);row=r['rows'][0]
    if fault=='schema':r['schema']=m.core.NAME+'.report'
    elif fault=='contract':r['contract']=m.core.contract()
    elif fault=='event':events.insert(0,dict(kind='unknown'))
    elif fault=='drop_attempt':events.remove(next(e for e in events if e['kind']=='module_attempt'))
    elif fault=='pixels':row['pixels']=s.array('badpixels',np.zeros((224,224,3),np.uint8))
    elif fault=='197':row['modules'][0]['output']=s.array('bad197',np.zeros((1,197,192),np.float32))
    elif fault=='native_output':row['latent']=s.array('badlatent',np.zeros(192,np.float32))
    elif fault=='selected':row['selected_index']=2
    elif fault=='guard':r['guards'].pop()
    elif fault=='count':r['counts']['encoder']['completed']-=1
    elif fault=='missing':row['source']=None
    else:r['blocked']=['predict']
    with pytest.raises((ValueError,KeyError)):m.validate(p,r,events,s.load)


@pytest.mark.parametrize('kind',['predict','AD','action_encoder'])
def test_no_extra_neural_operations(monkeypatch,kind):
    with pytest.raises(ValueError):execute(monkeypatch,forbidden=kind)


def test_prepare_reconstructs_v2_strict_not_hash_only(monkeypatch):
    monkeypatch.setattr(m,'sha',lambda p:m.INTERVENTION_SHA if str(p)==m.intervention.__file__ else m.CORE_SHA if str(p)==m.core.__file__ else '1'*64)
    calls=[];monkeypatch.setattr(m.intervention,'consume_physical',lambda *a:(calls.append(a) or dict(real='v2')))
    monkeypatch.setattr(m.physical,'read',lambda *a:dict(real='mismatched'))
    with pytest.raises(ValueError,match='whole actual v2 physical strict reconstruction'):
        m.build('/RAM_ONLY/v2physical','2'*64,'/RAM_ONLY/receipt','1'*64)
    assert len(calls)==1


def test_authorization_before_IO_and_no_old_globals_changed(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('unexpected IO'))
    with pytest.raises(ValueError):m.run('/NONE','0'*64,'/NONE','wrong')
    assert m.core.NAME=='agent_p2_midfeedback_terminal_rgb_20261028_v1'
    assert m.core.intervention.NAME=='agent_p2_contextual_midfeedback_recompile_20261028_v1'
    assert m.contract()['physical_schema']==m.intervention.NAME+'.physical_consume'


def test_pure_adapters_do_not_mutate_authority_or_report(replay,monkeypatch):
    plans,s,rgb=fixture();before=deepcopy((plans,s,rgb));m.fixed_rows(plans,s,rgb,'/RAM_ONLY/v2physical')
    assert (plans,s,rgb)==before
    p,r,store=replay;before=deepcopy((p,r,store.events));monkeypatch.setattr(m.bridge,'load_pixels',prior.pixels)
    m.validate(p,r,store.events,store.load);assert (p,r,store.events)==before


def test_prepare_full_v2_paired_schema_without_control_report_sha(monkeypatch,tmp_path):
    plans,strict,oldrgb=fixture();assert 'control_report_sha' not in strict
    strict['counts']=dict(public_step=dict(attempted=160,completed=160))
    physical=tmp_path/'v2physical';compiled=tmp_path/'v2compiled';physical.mkdir();compiled.mkdir()
    def put(path,value):path.write_bytes(m.raw(value));return m.sha(path)
    oldfile=tmp_path/'oldrgb.json';put(oldfile,oldrgb)
    prepfile=tmp_path/'v2prepare.json';prep_sha=put(prepfile,dict(RAM_ONLY=True))
    rows=[dict(resources={k:dict(attempted=v,completed=v) for k,v in m.intervention.budget().items()},
        AD_API_resources={k:dict(attempted=1,completed=1) for k in ('jacobian','grad')},
        predict_API_resources=dict(predict=dict(attempted=7,completed=7))) for i in range(8)]
    compiled_sha=put(compiled/'report.json',dict(schema=m.intervention.NAME+'.compile',status='all8_safeguarded_science_blocked',
        rows=rows,elapsed_seconds=1.25))
    binding=dict(feedback_compile_output=str(compiled),feedback_compile_sha256=compiled_sha,prepared=str(prepfile),prepare_sha256=prep_sha)
    physical_sha=put(physical/'report.json',dict(execution_binding=binding))
    receipt=tmp_path/'actualreceipt.json';receipt_sha=put(receipt,strict)
    p=dict(schema=m.intervention.NAME+'.prepare',contract=m.intervention.contract(),arguments=dict(rgb_receipt=str(oldfile)),
        bindings={},policy='/RAM/model',expected_model_hash='f'*64,inherited_RGB_counts={},inherited_compiler_resources={},inherited_physical_counts={})
    calls=[]
    def consume(output,expected):
        assert str(output)==str(physical) and expected==physical_sha;calls.append('strict');return deepcopy(strict)
    def material(output,expected):
        assert output==str(compiled) and expected==compiled_sha;calls.append('material');return p,dict(plans=plans,bindings={}),None
    monkeypatch.setattr(m.intervention,'consume_physical',consume)
    monkeypatch.setattr(m.intervention,'physical_material',material)
    monkeypatch.setattr(m.bridge,'load_pixels',prior.pixels)
    monkeypatch.setattr(m.bridge,'verify_bindings',lambda *a:None)  # RAM metadata seam, not real certification
    monkeypatch.setattr(m.bridge.rgb,'MODELS',{})
    monkeypatch.setattr(m.bridge.rgb,'read',lambda *a:{})
    monkeypatch.setattr(m.bridge.rgb,'config_shapes',lambda *a:{})
    result=m.build(str(physical),physical_sha,str(receipt),receipt_sha)
    assert calls==['strict','material'] and result['schema']==m.NAME+'.prepare' and len(result['rows'])==8
    assert result['actual_new_compiler_counts']['resources']['predict']==dict(attempted=56,completed=56)
    assert result['expected_calls']['encoder']==result['expected_calls']['projector']==8
    assert result['bindings'][str(receipt)]==receipt_sha and result['bindings'][str(compiled/'report.json')]==compiled_sha
