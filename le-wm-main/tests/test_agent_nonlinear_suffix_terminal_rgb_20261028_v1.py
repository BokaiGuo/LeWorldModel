"""RAM arrays, metadata and native hook doubles; no neural/physics/production build."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib
import json
import subprocess
import sys
import numpy as np
import pytest
import test_agent_p2_midfeedback_terminal_rgb_20261028_v1 as h
from test_agent_nonlinear_suffix_physical_20261028_v1 import fixture as physical_fixture
from experiments.spectral_reachability import agent_nonlinear_suffix_terminal_rgb_20261028_v1 as m


def fixture(missing=()):
    prep,data,model,compiled,consumed,original=physical_fixture()
    # Preserve actual upstream schema, full168 and exact model-only Q/context arithmetic.
    for i,slot in enumerate(compiled['slots']):
        context=deepcopy(slot['task']['root_context'])
        context['latent_history'][-1]=(np.asarray(slot['task']['goal'],np.float32)+np.float32(.5)).tolist()
        q=m.bridge.mse(context['latent_history'][-1],slot['task']['goal'])
        for j,row in enumerate(model['rows'][42*i:42*i+42]):
            row['result']['selected_context']=deepcopy(context)
            row['result']['selected_Q']=q+abs(j-i)
        winner=model['rows'][42*i+i]['result'];chosen=model['finals'][i]['plan']
        chosen.update(context=deepcopy(context),Q=q,
            selection_digest=m.suffix.digest([m.suffix.NAME,slot['task']['binding'],winner]))
        consumed['plans'][i]=dict(chosen,root_slot=slot['root_slot'],condition=slot['condition'],
            task_id=slot['task']['binding']['task_id'],planner_report_sha256=prep['arguments']['producer_report_sha256'])
    consumed['rows']=deepcopy(model['rows'])
    prep['plans']=m.producer.plans_from_saved(model,compiled,consumed,original,data,prep['arguments'])
    prep['planner_authority']['plans_sha256']=m.producer.mixed.core.digest(prep['plans'])
    actual=[]
    for i,plan in enumerate(prep['plans']):
        native=prep['source_roots'][plan['root_slot']];state=deepcopy(native['root_state'])
        cp=[];trajectory=[]
        for step in ([0,5] if i in missing else [0,5,10]):
            f=dict(step=step,png=f'RAM_r{i}_plan_{step}.png',file_sha256='a'*64,pixel_sha256='b'*64,
                state=deepcopy(state),snapshot=dict(state=deepcopy(state),RAM_ONLY=True))
            cp.append(f);trajectory.append(dict(deepcopy(f),phase='plan',condition=plan['condition'],
                overlay_index=1 if plan['condition']=='on' else 0))
        actual.append(dict(index=i,root_slot=plan['root_slot'],condition=plan['condition'],arm=plan['arm'],
            task_id=plan['task_id'],selection_digest=plan['selection_digest'],plan=deepcopy(plan),
            status='native_stopped' if i in missing else 'horizon_completed',initial_exact=True,
            own_returned=9 if i in missing else 10,checkpoints=cp,trajectory=trajectory,
            missing_checkpoints=[10] if i in missing else [],final_state=state,final_metrics=dict(RAM_ONLY=True),
            native_success=i in missing,terminated=i in missing,truncated=False,root_state=deepcopy(native['root_state']),
            root_sha256=native['root_sha256'],goal=deepcopy(native['goal']),goal_pose=deepcopy(native['goal_pose']),
            goal_pixel_sha256=native['goal_pixels'][plan['condition']]))
    strict=dict(schema=m.producer.NAME+'.consume',status=m.producer.RECORDED,errors=[],fixed_denominator=4,
        known_roots=2,fresh_roots=0,inherited=0,rows=actual,planner_authority=deepcopy(prep['planner_authority']),
        P1_admission=False,scientific_admission=False,freshness=False,independent_confirmation=False,equal_FLOPs=False,
        counts={k:dict(attempted=v,completed=v) for k,v in m.producer.contract()['caps'].items()},bindings={})
    return prep,strict,compiled,model


def prepared(missing=()):
    rows=m.fixed_rows(*fixture(missing),'/RAM_ONLY/physical')
    return dict(schema=m.NAME+'.prepare',rows=rows,contract=m.contract(),
        shapes=dict(hidden=[1,257,192]),
        expected_calls={k:4-len(missing) if k in ('encoder','projector') else 0 for k in m.bridge.KINDS})


def test_own_physical_schema_four_exact_terminal_joins():
    p=prepared();assert len(p['rows'])==4
    assert [r['condition'] for r in p['rows']]==['off','on','off','on']
    assert all(r['source']['step']==10 and r['raw_history10']==r['full10'] for r in p['rows'])
    assert all(Path(r['source']['path']).name==r['source']['png'] for r in p['rows'])
    assert all('selector_analysis_sha256' not in row for row in p['rows'])


@pytest.mark.parametrize('fault',['old_schema','task','raw','goal','context','Q','selection','checkpoint_early',
    'checkpoint_missing','frame_condition','frame_state','snapshot','drop_row','prediction','authority',
    'plan_digest','missing_without_stop','native_goal','final_count','partial168','prefix','normalizer'])
def test_physical_model_joins_reject_tampering(fault):
    p,s,c,r=fixture()
    if fault=='old_schema':s['schema']=m.rgb_base.producer.NAME+'.consume'
    elif fault=='task':p['plans'][0]['task_sha256']='0'*64
    elif fault=='raw':s['rows'][0]['plan']['commands'][0][0]=.9
    elif fault=='goal':c['slots'][0]['task']['goal'][0]+=1.
    elif fault=='context':c['slots'][0]['task']['root_context']['latent_history'][0][0]+=1.
    elif fault=='Q':c['slots'][0]['task']['projection'][0][0]=2.
    elif fault=='selection':r['finals'][0]['plan']['prefix_index']=1
    elif fault=='checkpoint_early':s['rows'][0]['own_returned']=9
    elif fault=='checkpoint_missing':s['rows'][0]['missing_checkpoints']=[10]
    elif fault=='frame_condition':s['rows'][0]['trajectory'][-1]['condition']='on'
    elif fault=='frame_state':s['rows'][0]['checkpoints'][-1]['state'][0]+=1
    elif fault=='snapshot':s['rows'][0]['checkpoints'][-1]['snapshot']['state'][0]+=1
    elif fault=='drop_row':s['rows'].pop()
    elif fault=='prediction':r['finals'][0]['plan']['context']['latent_history'][-1][0]+=2.
    elif fault=='authority':s['planner_authority']['producer']='fake'
    elif fault=='plan_digest':p['plans'][0]['selection_digest']='0'*64
    elif fault=='missing_without_stop':
        s['rows'][0]['checkpoints'].pop();s['rows'][0]['missing_checkpoints']=[10]
    elif fault=='native_goal':s['rows'][0]['goal'][0]+=1.
    elif fault=='final_count':r['finals'].pop()
    elif fault=='partial168':r['rows'][0]['status']='failed'
    elif fault=='prefix':c['slots'][0]['candidates'][0]['prefix'][0][0]+=.1
    elif fault=='normalizer':c['slots'][0]['task']['action_std'][0]+=1.
    with pytest.raises((ValueError,KeyError)):m.fixed_rows(p,s,c,r,'/RAM_ONLY/physical')


def execute(monkeypatch,missing=(),forbidden=None,tokens=257):
    p=prepared(missing);rows=deepcopy(p['rows']);store=h.old.Memory()
    monkeypatch.setattr(m.bridge,'load_pixels',h.pixels)
    model=h.Model(tokens);torch=h.fake_torch();errors=[]
    trace=m.bridge.Trace(model,torch,store,lambda:None,p['expected_calls'])
    guard=m.bridge.native.Ledger(2,lambda e:store.event(dict(kind='guard_event',event=e)),no_grad=lambda:True)
    def encode(pix,row,scope,ordinal):
        guard.image=ordinal
        with m.bridge.physical.saved.overlay.traced_scope(trace,scope,row):
            if forbidden=='predict':model.predict(None)
            if forbidden=='AD':torch.autograd.functional.jacobian(None)
            if forbidden=='action_encoder':model.action_encoder(None)
            pre=m.bridge.pixels.preprocess_pixels(pix)
            row['preprocessed']=trace.save(h.Tensor(pre),f'r{row["index"]}_preprocessed')
            return model.projector(model.encoder(h.Tensor(pre.reshape(1,3,224,224)))[:,0]).a[0].copy()
    with trace,m.encode_guard(model,torch,guard,errors):m.process(p,rows,store,encode)
    return p,dict(schema=m.NAME+'.report',rows=rows,contract=m.contract(),errors=errors,blocked=guard.blocked,
        modules=trace.entries,guards=guard.entries,counts=m.bridge.counts(trace.entries),precision_before=m.precision_contract(),precision_after=m.precision_contract()),store


@pytest.fixture(scope='module')
def replay():
    with pytest.MonkeyPatch.context() as patch:return execute(patch)


def test_complete_encoder_projector_native_arrays_four(replay,monkeypatch):
    p,r,s=replay;monkeypatch.setattr(m.bridge,'load_pixels',h.pixels)
    out=m.validate(p,r,s.events,s.load)
    assert len(out)==4 and len(r['modules'])==len(r['guards'])==8
    assert r['counts']['encoder']==dict(attempted=4,completed=4)
    assert all(x['actual_Q'] is not None and not x['P1_admission'] for x in out)


@pytest.mark.parametrize('missing',[(0,),(1,3),(0,1,2,3)])
def test_missing_cp10_keeps_four_null_no_padding(monkeypatch,missing):
    p,r,s=execute(monkeypatch,missing);out=m.validate(p,r,s.events,s.load)
    assert len(out)==4 and sum(x['latent_missing'] for x in out)==len(missing)
    assert all(out[i]['actual_Q'] is None for i in missing)


@pytest.mark.parametrize('fault',['event','event_tail','preprocess','pixels','latent','hidden197','plan','guard','count','blocked'])
def test_strict_saved_validation_no_unknown_filter_or_preprocess_shortcut(replay,monkeypatch,fault):
    p,r0,s0=replay;r=deepcopy(r0);events=deepcopy(s0.events);s=h.old.Memory();s.files=dict(s0.files)
    monkeypatch.setattr(m.bridge,'load_pixels',h.pixels)
    if fault=='event':events.insert(0,dict(kind='UNKNOWN'))
    elif fault=='event_tail':events.append(dict(kind='UNKNOWN'))
    elif fault=='preprocess':r['rows'][0]['preprocessed']=s.array('badpre',np.zeros((1,1,3,224,224),np.float32))
    elif fault=='pixels':r['rows'][0]['pixels']=s.array('badpixels',np.zeros((224,224,3),np.uint8))
    elif fault=='latent':r['rows'][0]['latent']=s.array('badlatent',np.zeros(192,np.float32))
    elif fault=='hidden197':r['rows'][0]['modules'][0]['output']=s.array('bad197',np.zeros((1,197,192),np.float32))
    elif fault=='plan':r['rows'][0]['goal192'][0]=1.
    elif fault=='guard':r['guards'].pop()
    elif fault=='count':r['counts']['encoder']['completed']-=1
    else:r['blocked']=['predict']
    with pytest.raises((ValueError,KeyError)):m.validate(p,r,events,s.load)


@pytest.mark.parametrize('forbidden',['predict','AD','action_encoder'])
def test_frozen_guard_forbids_non_encoder_computation(monkeypatch,forbidden):
    with pytest.raises(ValueError):execute(monkeypatch,forbidden=forbidden)


def test_cpu_authority_refused_before_io():
    with pytest.raises(ValueError):m.run('/not/read','a'*64,'/not/created',authorization='CPU')


def test_encode_guard_never_replaces_torch_globals_and_restores_profile():
    import sys
    model=h.Model();torch=h.fake_torch();ledger=m.bridge.native.Ledger(1,lambda e:None);errors=[]
    original=(torch.autograd.grad,torch.autograd.backward,torch.autograd.functional.jacobian,
        torch.Tensor.backward,torch.func.jacrev,sys.getprofile(),model.predict,model.encoder.forward)
    with m.encode_guard(model,torch,ledger,errors):
        assert (torch.autograd.grad,torch.autograd.backward,torch.autograd.functional.jacobian,
            torch.Tensor.backward,torch.func.jacrev)==original[:5]
        with pytest.raises(ValueError,match='forbidden'):torch.autograd.functional.jacobian(None)
    assert (torch.autograd.grad,torch.autograd.backward,torch.autograd.functional.jacobian,
        torch.Tensor.backward,torch.func.jacrev,sys.getprofile(),model.predict,model.encoder.forward)==original
    assert errors==[] and ledger.blocked


def test_thread_probe_unwraps_decorators_not_shared_wrapper_code():
    from functools import wraps
    def wrapper(fn):
        @wraps(fn)
        def shared(*a,**kw):return fn(*a,**kw)
        return shared
    def ad(*a,**kw):pytest.fail('AD must be stopped before body')
    torch=h.fake_torch();torch.autograd.grad=wrapper(ad)
    safe=wrapper(lambda:123);ledger=m.bridge.native.Ledger(1,lambda e:None)
    with m.encode_guard(h.Model(),torch,ledger,[]):
        assert safe()==123
        with pytest.raises(ValueError,match='forbidden'):torch.autograd.grad(None)


@pytest.mark.parametrize('field',list(m.precision_contract()))
def test_actual_precision_flags_are_checked_with_RAM_double(field):
    value=m.precision_contract()
    def mock():
        return NS(backends=NS(cuda=NS(matmul=NS(allow_tf32=value['matmul_TF32']),
            math_sdp_enabled=lambda:value['math_SDP'],flash_sdp_enabled=lambda:value['flash_SDP'],
            mem_efficient_sdp_enabled=lambda:value['memory_efficient_SDP']),cudnn=NS(allow_tf32=value['cudnn_TF32'])),
            is_autocast_enabled=lambda device:value['autocast'],is_grad_enabled=lambda:value['grad'])
    assert m.checked_precision(mock())==m.precision_contract()
    value[field]=not value[field]
    with pytest.raises(ValueError,match='precision guard'):m.checked_precision(mock())


@pytest.mark.parametrize('fault',['before','after','config'])
def test_saved_precision_or_config_mutation_rejected(replay,monkeypatch,fault):
    p0,r0,s=replay;p=deepcopy(p0);r=deepcopy(r0)
    monkeypatch.setattr(m.bridge,'load_pixels',h.pixels)
    if fault=='config':p['shapes']['hidden']=[1,197,192]
    else:r['precision_'+fault]['matmul_TF32']=True
    with pytest.raises(ValueError):m.validate(p,r,s.events,s.load)


def test_bound_config_derives257_and_refuses197():
    config=dict(encoder=dict(image_size=224,patch_size=14,pretrained=False),
        projector=dict(input_dim=192,output_dim=192))
    assert m.bridge.rgb.config_shapes(config)['hidden']==[1,257,192]
    config['encoder']['patch_size']=16
    with pytest.raises(ValueError):m.bridge.rgb.config_shapes(config)


def test_actual_frozen_helper_budget_and_array_ABI_metadata_only():
    assert m.CAP==m.bridge.CAP==m.rgb_base.CAP==64*1024**2
    assert m.FINAL==m.bridge.FINAL==m.rgb_base.FINAL==8*1024**2
    assert m.FLOOR==m.bridge.FLOOR==m.rgb_base.FLOOR==200*1024**2
    assert m.WALL==m.bridge.WALL==m.rgb_base.WALL==600
    assert m.bridge.KINDS==('encoder','projector','action_encoder','predict_API','predictor','pred_proj')
    for module,expected in m.FROZEN_HELPERS:assert m.sha(module.__file__)==expected


def test_cold_import_and_help_without_neural_or_native_imports():
    code="""
import builtins
old=builtins.__import__
def guard(name,*a,**kw):
    if name.split('.')[0] in ('torch','gymnasium','pymunk','stable_worldmodel'):raise AssertionError(name)
    return old(name,*a,**kw)
builtins.__import__=guard
from experiments.spectral_reachability import agent_nonlinear_suffix_terminal_rgb_20261028_v1 as m
assert m.contract()['max_calls']['predict']==0
try:m.main(['--help'])
except SystemExit as e:assert e.code==0
"""
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr


def setup_build(monkeypatch):
    p,s,c,r=fixture()
    args=dict(physical_output='/RAM/physical',physical_report_sha='1'*64,physical_prepared='/RAM/physical_prepare.json',
        physical_prepare_sha='2'*64,physical_receipt='/RAM/physical_receipt.json',physical_receipt_sha='3'*64)
    s['physical_report_sha256']=args['physical_report_sha']
    a=p['planner_authority'];c['policy']=str(m.bridge.rgb.POLICY);c['expected_model_hash']=m.bridge.rgb.MODEL_TENSOR_SHA
    # Update fixture model hash everywhere before recomputing exact source/task joins.
    for slot in c['slots']:slot['task']['binding']['model_sha256']=c['expected_model_hash']
    for plan,slot in zip(p['plans'],c['slots'],strict=True):
        plan['model_task_binding']=deepcopy(slot['task']['binding'])
        plan['selection_digest']=m.producer.mixed.core.digest({k:v for k,v in plan.items() if k!='selection_digest'})
    for actual,plan in zip(s['rows'],p['plans'],strict=True):
        actual['plan']=deepcopy(plan);actual['selection_digest']=plan['selection_digest']
    p['planner_authority']['plans_sha256']=m.producer.mixed.core.digest(p['plans'])
    s['planner_authority']=deepcopy(p['planner_authority'])
    hashes={args['physical_output']+'/report.json':args['physical_report_sha'],args['physical_prepared']:args['physical_prepare_sha'],
        args['physical_receipt']:args['physical_receipt_sha'],a['producer_prepared']:a['producer_prepare_sha256'],
        a['producer_receipt']:a['producer_receipt_sha256'],a['producer_output']+'/report.json':a['producer_report_sha256']}
    hashes.update({str(Path(mod.__file__).resolve()):value for mod,value in m.FROZEN_HELPERS})
    s['bindings']={k:v for k,v in hashes.items() if k.startswith('/RAM/')}
    report=dict(execution_binding=dict(prepared=args['physical_prepared'],prepare_sha256=args['physical_prepare_sha']),rows=s['rows'])
    records={args['physical_receipt']:s,args['physical_output']+'/report.json':report,args['physical_prepared']:p,
        a['producer_prepared']:c,a['producer_output']+'/report.json':r}
    calls=[]
    monkeypatch.setattr(m,'sha',lambda path:hashes.get(str(path),hashlib.sha256(str(path).encode()).hexdigest()))
    monkeypatch.setattr(m,'read',lambda path,*a:deepcopy(records[str(path)]))
    monkeypatch.setattr(m.producer,'consume',lambda output,expected:calls.append((str(output),expected)) or deepcopy(s))
    monkeypatch.setattr(m.bridge,'verify_bindings',lambda b:None)
    monkeypatch.setattr(m.bridge,'load_pixels',h.pixels)
    monkeypatch.setattr(m.suffix.source.kernel,'validate_normalizer_source',lambda root:None)
    monkeypatch.setattr(m.bridge.rgb,'read',lambda *a:dict(encoder=dict(image_size=224,patch_size=14,pretrained=False),
        projector=dict(input_dim=192,output_dim=192)))
    return args,records,s,calls


def test_build_verify_full_physical_consume_exact_receipt_not_status_only(monkeypatch):
    args,records,strict,calls=setup_build(monkeypatch)
    p=m.build(**args)
    assert calls==[(args['physical_output'],args['physical_report_sha'])]
    assert p['physical_authority']['complete_all168'] is True
    assert p['actual_new_physical_counts']==strict['counts']
    assert p['inherited_suffix_resources']==strict['planner_authority']['actual_new_compute']
    assert p['expected_calls']['predict_API']==0 and p['expected_calls']['encoder']==4
    records['/RAM/rgb_prepare.json']=p
    assert m.verify('/RAM/rgb_prepare.json',m.sha('/RAM/rgb_prepare.json'))==p and len(calls)==2
    records[args['physical_receipt']]=dict(strict,status='fake')
    with pytest.raises(ValueError,match='entire real physical strict receipt'):m.build(**args)


def test_build_wrong_external_sha_refused_before_actual_consume(monkeypatch):
    args,records,strict,calls=setup_build(monkeypatch);args['physical_report_sha']='4'*64
    with pytest.raises(ValueError,match='external physical seals'):m.build(**args)
    assert not calls


def test_cli_actual_physical_six_arguments_no_future_defaults(monkeypatch):
    args=dict(physical_output='/RAM/out',physical_report_sha='a'*64,physical_prepared='/RAM/p',
        physical_prepare_sha='b'*64,physical_receipt='/RAM/r',physical_receipt_sha='c'*64)
    calls=[]
    monkeypatch.setattr(m,'build',lambda *values:calls.append(dict(zip(args,values,strict=True))) or dict(RAM=True))
    monkeypatch.setattr(m,'write',lambda output,value:dict(RAM=True))
    command=['prepare','--output','/RAM/new']
    for k,v in args.items():command+=['--'+k.replace('_','-'),v]
    assert m.main(command)==0 and calls==[args]

