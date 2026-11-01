"""RAM metadata/native array doubles only: no torch, physics or production build."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib
import json
import subprocess
import sys
import numpy as np
import pytest
import test_agent_nonlinear_suffix_terminal_rgb_20261028_v1 as prior
import test_agent_nonlinear_suffix_physical_20261028_v1 as phys
import test_agent_p2_midfeedback_terminal_rgb_20261028_v1 as h
from experiments.spectral_reachability import agent_nonlinear_suffix_midpoint_rgb_20261028_v1 as m


def fixture():
    # Correct the RAM source packet before tasks/hashes are constructed. This is
    # fixture injection only; production never changes a dependency's globals.
    original=phys.model_slot
    def slot(i):
        s=original(i);warm=np.asarray(m.producer.moving.warmup_commands(),np.float32)
        s['packet']['root_context']['action_history'][1]=m.suffix.source.kernel.normalized_history(warm[5:])[0].tolist()
        return s
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(phys,'model_slot',slot)
        pp,strict,compiled,report=prior.fixture()
    physical=dict(schema=m.producer.NAME+'.report',status=m.producer.RECORDED,errors=[],rows=deepcopy(strict['rows']))
    for row in physical['rows']:
        f=deepcopy(row['trajectory'][0]);f.update(phase='warmup',step=5,png=f'RAM_r{row["index"]}_warmup5.png')
        row['trajectory'].insert(0,f)
    tp=dict(schema=m.terminal.NAME+'.prepare',contract=m.terminal.contract(),
        physical_authority=deepcopy(pp['planner_authority']),
        action_mean=m.suffix.source.kernel.ACTION_MEAN.tolist(),action_std=m.suffix.source.kernel.ACTION_STD.tolist(),
        rows=m.terminal.fixed_rows(pp,strict,compiled,report,'/RAM_ONLY/physical'))
    accepted=dict(schema=m.terminal.NAME+'.consume',status='nonlinear_suffix4_RGB_verified_not_admitted',
        fixed_denominator=4,known_roots=2,fresh_roots=0,P1_admission=False,scientific_admission=False,freshness=False,
        physical_authority=deepcopy(pp['planner_authority']),rows=[])
    accepted.update(actual_new_physical_counts=deepcopy(strict['counts']),
        inherited_suffix_resources=deepcopy(pp['planner_authority']['actual_new_compute']),
        inherited_original_resources=deepcopy(pp['planner_authority']['inherited_compute']),
        counts={k:dict(attempted=4 if k in ('encoder','projector') else 0,
            completed=4 if k in ('encoder','projector') else 0) for k in m.KINDS})
    for row in tp['rows']:
        z=np.full(192,.25,np.float32)
        accepted['rows'].append(dict(deepcopy(row),actual192=z.tolist(),actual_Q=m.bridge.mse(z,row['goal192']),
            endpoint_MSE=m.bridge.mse(z,row['predicted192'])))
    return tp,accepted,physical,pp,compiled


def prepared(missing=()):
    rows=m.fixed_rows(*fixture(),'/RAM_ONLY/physical')
    # Native-stop null cases here exercise process/validator, not missing-asset
    # source validation (covered separately below).
    for i in missing:
        rows[i].update(eligible=False,status='missing_actual_history',missing_history=[['plan',5]],
            frames=rows[i]['frames'][:2]+[None])
        rows[i]['row_binding_sha256']=m.producer.mixed.core.digest({k:v for k,v in rows[i].items() if k!='row_binding_sha256'})
    return dict(schema=m.NAME+'.prepare',contract=m.contract(),rows=rows,
        expected_calls={k:4-len(missing) for k in m.KINDS},expected_encode_API=4-len(missing))


def pixels(f):
    return np.full((224,224,3),32+f['step']+(64 if f['phase']=='plan' else 0),np.uint8)


class Model:
    def __init__(self,tokens=257):
        self.encoder=h.Module(lambda x:NS(last_hidden_state=h.Tensor(np.broadcast_to(
            x.a.mean(axis=(1,2,3))[:,None,None],(3,tokens,192)).copy())))
        self.projector=h.Module(lambda x:h.Tensor(x.a+np.float32(.125)))
        self.action_encoder=h.Module(lambda x:h.Tensor(np.repeat(x.a.sum(-1,keepdims=True),192,axis=-1)))
        self.predictor=h.Module(lambda z,a:h.Tensor(z.a+a.a))
        self.pred_proj=h.Module(lambda x:h.Tensor(x.a+np.float32(.125)))
    def encode(self,info):
        x=info['pixels'];hidden=self.encoder(h.Tensor(x.a.reshape(3,3,224,224))).last_hidden_state
        info['emb']=h.Tensor(self.projector(hidden[:,0]).a[None]);return info
    def predict(self,z,a):return h.Tensor(self.pred_proj(self.predictor(z,a)[0]).a[None])


def execute(monkeypatch,missing=(),tokens=257):
    p=prepared(missing);rows=deepcopy(p['rows']);store=h.old.Memory();model=Model(tokens)
    monkeypatch.setattr(m.bridge,'load_pixels',pixels)
    trace=m.Trace(model,h.fake_torch(),store,lambda:None,p['expected_calls']);calls=[]
    def encode(pix,enc,scope):
        with m.bridge.physical.saved.overlay.traced_scope(trace,scope,enc):
            pre=np.concatenate([m.bridge.pixels.preprocess_pixels(x) for x in pix],axis=1)
            enc['preprocessed']=trace.save(h.Tensor(pre),'pre_'+str(scope['row']))
            e=dict(index=len(calls),scope=scope,completed=False);calls.append(e)
            store.event(dict(kind='encode_API_attempt',entry=deepcopy(e)))
            result=model.encode(dict(pixels=h.Tensor(pre)))
            enc['api_pixels']=trace.save(result['pixels'],'api_pixels_'+str(scope['row']))
            enc['api_emb']=trace.save(result['emb'],'api_emb_'+str(scope['row']))
            e['completed']=True;store.event(dict(kind='encode_API_return',entry=deepcopy(e)))
            return result['emb'].a.copy()
    def predict(inp,fb,scope):
        with m.bridge.physical.saved.overlay.traced_scope(trace,scope,fb):
            return model.predict(h.Tensor(inp['context']),model.action_encoder(h.Tensor(inp['outgoing']))).a.copy()
    with trace:m.process(p,rows,store,encode,predict)
    r=dict(schema=m.NAME+'.report',contract=m.contract(),rows=rows,errors=[],blocked=[],modules=trace.entries,
        encode_API=calls,counts=m.resource_counts(trace.entries,calls),
        precision_before=m.terminal.precision_contract(),precision_after=m.terminal.precision_contract())
    return p,r,store


@pytest.fixture(scope='module')
def replay():
    with pytest.MonkeyPatch.context() as patch:return execute(patch)


def test_true_history_roles_actions_and_context_are_saved():
    p=prepared()
    for r in p['rows']:
        assert [(f['phase'],f['step']) for f in r['frames']]==list(m.ROLES)
        assert len(r['original_root_context']['latent_history'])==3
        b=m.tokens(r)
        np.testing.assert_array_equal(b['outgoing'][0,0],r['original_root_context']['action_history'][1])
        np.testing.assert_array_equal(b['raw_packets'][1:],np.asarray(r['full10'],np.float32).reshape(2,5,2))
    assert 'entire actual history3' in p['contract']['intervention']


@pytest.mark.parametrize('fault',['task','goal','raw','authority','rowcount','frame_condition','snapshot',
    'same_frame','missing_warmup','missing_plan5','root_action','actualQ','source_time','partial168_schema'])
def test_real_metadata_schema_join_refuses(fault):
    tp,a,r,pp,c=fixture()
    if fault=='task':a['rows'][0]['task_sha256']='0'*64
    elif fault=='goal':a['rows'][0]['goal192'][0]+=.1
    elif fault=='raw':tp['rows'][0]['full10'][0][0]+=.1
    elif fault=='authority':a['physical_authority']['producer']='fake'
    elif fault=='rowcount':r['rows'].pop()
    elif fault=='frame_condition':r['rows'][0]['trajectory'][0]['condition']='on'
    elif fault=='snapshot':r['rows'][0]['trajectory'][0]['snapshot']['state'][0]+=1
    elif fault=='same_frame':r['rows'][0]['trajectory'][0]['png']=r['rows'][0]['trajectory'][1]['png']
    elif fault=='missing_warmup':r['rows'][0]['trajectory'].pop(0)
    elif fault=='missing_plan5':r['rows'][0]['trajectory']=[f for f in r['rows'][0]['trajectory'] if not(f['phase']=='plan' and f['step']==5)]
    elif fault=='root_action':c['slots'][0]['task']['root_context']['action_history'][1][0]+=1
    elif fault=='actualQ':a['rows'][0]['actual_Q']+=1
    elif fault=='source_time':tp['rows'][0]['source']['step']=5
    else:tp['schema']='old.prepare'
    with pytest.raises((ValueError,KeyError)):m.fixed_rows(tp,a,r,pp,c,'/RAM_ONLY/physical')


def test_full_native_array_replay_and_separate_cost_axes(replay,monkeypatch):
    p,r,s=replay;monkeypatch.setattr(m.bridge,'load_pixels',pixels)
    rows=m.validate(p,r,s.events,s.load)
    assert len(rows)==4 and len(r['modules'])==24
    assert r['counts']['encoder']==dict(attempted=4,completed=4,input_axis0_samples=12,input_temporal_rows=12)
    assert r['counts']['predict_API']==dict(attempted=4,completed=4,input_axis0_samples=4,input_temporal_rows=12)
    assert all(x['conditioned_final192'] and not x['P1_admission'] and not x['physical_benefit_claim'] for x in rows)
    assert all(s.load(row['actual_full3']).shape==(1,3,192) for row in r['rows'])
    assert len(set(s.load(r['rows'][0]['actual_full3'])[0,:,0]))==3


@pytest.mark.parametrize('fault',['unknown_event','extra_event','missing_event','module','count','encode_API',
    'precision','blocked','raw','root_context','row_binding','context','incoming','normalizer','preprocess',
    'pixels','hidden197','CLS','prediction','missingrow','outgoing_extra'])
def test_full_strict_rejects_mutation_before_diagnostic(replay,monkeypatch,fault):
    p0,r0,s0=replay;p=deepcopy(p0);r=deepcopy(r0);events=deepcopy(s0.events);s=h.old.Memory();s.files=dict(s0.files)
    monkeypatch.setattr(m.bridge,'load_pixels',pixels);row=r['rows'][0]
    if fault=='unknown_event':events[0]['kind']='unknown'
    elif fault=='extra_event':events.append(dict(kind='unknown'))
    elif fault=='missing_event':events.pop()
    elif fault=='module':r['modules'].pop()
    elif fault=='count':r['counts']['predict_API']['completed']-=1
    elif fault=='encode_API':r['encode_API'].pop()
    elif fault=='precision':r['precision_before']['matmul_TF32']=True
    elif fault=='blocked':r['blocked']=['AD']
    elif fault=='raw':row['full10'][0][0]+=.1
    elif fault=='root_context':row['original_root_context']['latent_history'][0][0]+=.1
    elif fault=='row_binding':row['row_binding_sha256']='0'*64
    elif fault in ('context','incoming','normalizer'):
        k={'context':'context','incoming':'incoming','normalizer':'action_mean'}[fault]
        x=s.load(row['feedback']['inputs'][k]).copy();x.flat[0]+=1
        row['feedback']['inputs'][k]=s.array('bad_'+k,x)
    elif fault=='preprocess':row['encoding']['preprocessed']=s.array('badpre',np.zeros((1,3,3,224,224),np.float32))
    elif fault=='pixels':row['encoding']['pixels']=s.array('badpix',np.zeros((3,224,224,3),np.uint8))
    elif fault=='hidden197':row['encoding']['modules'][0]['output']=s.array('bad197',np.zeros((3,197,192),np.float32))
    elif fault=='CLS':row['encoding']['modules'][1]['inputs'][0]=s.array('badCLS',np.zeros((3,192),np.float32))
    elif fault=='prediction':row['prediction']=s.array('badpred',np.zeros((1,3,192),np.float32))
    elif fault=='missingrow':r['rows'].pop()
    else:row['feedback']['inputs']['reward']=row['feedback']['inputs']['context']
    with pytest.raises((ValueError,KeyError)):m.validate(p,r,events,s.load)


@pytest.mark.parametrize('missing',[(0,),(1,3),(0,1,2,3)])
def test_null_denominator_never_pads(monkeypatch,missing):
    p,r,s=execute(monkeypatch,missing);out=m.validate(p,r,s.events,s.load)
    assert len(out)==4 and len(r['modules'])==6*(4-len(missing))
    assert all(out[i]['conditioned_final192'] is None for i in missing)


def test_197_is_rejected_at_actual_hook(monkeypatch):
    with pytest.raises(ValueError,match='hidden257'):execute(monkeypatch,tokens=197)


def test_guard_refuses_AD_before_body_and_restores_profile():
    torch=h.fake_torch();s=h.old.Memory();blocked=[];old=sys.getprofile();grad=torch.autograd.grad
    with m.no_AD(torch,blocked,s):
        assert torch.autograd.grad is grad
        with pytest.raises(ValueError,match='AD forbidden'):torch.autograd.grad(None)
    assert sys.getprofile() is old and blocked


def test_wrong_authority_before_any_input_read():
    with pytest.raises(ValueError,match='authority'):m.run('/not/read','a'*64,'/not/create','CPU')


def test_preregistered_main_protocol_binding_readonly():
    # Small saved metadata only. No production build, checkpoint or root files.
    assert m.sha(m.PROTOCOL)==m.PROTOCOL_SHA
    p=m.read(m.PROTOCOL)
    assert p['protocol']['primary']=='endpoint MSE to actual terminal encoding, all four rows'
    assert p['protocol']['secondary']=='absolute error of predicted goal Q versus actual goal Q'
    assert p['protocol']['selection'] is False


def test_public_array_loader_verifies_sha_and_shape(tmp_path,replay):
    _,r,s=replay;ref=r['rows'][0]['actual_full3']
    for name,payload in s.files.items():
        if name in str(ref): (tmp_path/name).write_bytes(payload)
    np.testing.assert_array_equal(m.load_saved_array(str(tmp_path),ref),s.load(ref))
    for f in tmp_path.iterdir():f.write_bytes(b'broken')
    with pytest.raises((ValueError,Exception)):m.load_saved_array(str(tmp_path),ref)


def test_cold_help_never_imports_neural_or_physics():
    code="""
import builtins,sys
old=builtins.__import__
def checked(name,*a,**k):
    if name.split('.')[0] in {'torch','gym','gymnasium','pymunk','stable_worldmodel'}:
        raise AssertionError('forbidden runtime import '+name)
    return old(name,*a,**k)
builtins.__import__=checked
from experiments.spectral_reachability import agent_nonlinear_suffix_midpoint_rgb_20261028_v1 as m
assert m.contract()['P1_admission'] is False
assert not any(n in sys.modules for n in ('torch','gym','gymnasium','pymunk','stable_worldmodel'))
m.main(['--help'])
"""
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr


def test_both_diagnostics_retained_without_rank_claim(replay):
    p,r,s=replay;out=m.analyze(p['rows'][0],r['rows'][0],s.load)
    z=s.load(r['rows'][0]['prediction'])[0,-1];actual=p['rows'][0]['actual_terminal192'];goal=p['rows'][0]['goal192']
    assert out['conditioned_endpoint_MSE']==m.bridge.mse(z,actual)
    assert out['conditioned_goalQ_absolute_error']==abs(m.bridge.mse(z,goal)-m.bridge.mse(actual,goal))
    assert out['endpoint_MSE_change']==out['conditioned_endpoint_MSE']-out['imagined_endpoint_MSE']
    assert 'rank_repaired' not in out


def setup_build(monkeypatch):
    tp,a,r,pp,c=fixture();c['expected_model_hash']=m.bridge.rgb.MODEL_TENSOR_SHA
    for slot,plan,base,end,actual in zip(c['slots'],pp['plans'],tp['rows'],a['rows'],r['rows'],strict=True):
        slot['task']['binding']['model_sha256']=c['expected_model_hash']
        plan['model_task_binding']=deepcopy(slot['task']['binding'])
        plan['selection_digest']=m.producer.mixed.core.digest({k:v for k,v in plan.items() if k!='selection_digest'})
        actual['plan']=deepcopy(plan);actual['selection_digest']=plan['selection_digest']
        for row in (base,end):
            row['model_task_binding']=deepcopy(plan['model_task_binding']);row['selection_digest']=plan['selection_digest']
    pp['planner_authority']['plans_sha256']=m.producer.mixed.core.digest(pp['plans'])
    tp['physical_authority']=deepcopy(pp['planner_authority']);a['physical_authority']=deepcopy(pp['planner_authority'])
    args=dict(terminal_output='/RAM/terminal',terminal_report_sha='1'*64,
        terminal_receipt='/RAM/terminal_receipt.json',terminal_receipt_sha='2'*64)
    tp.update(policy=str(m.bridge.rgb.POLICY),expected_model_hash=c['expected_model_hash'],
        arguments=dict(physical_output='/RAM/physical',physical_report_sha='3'*64,
            physical_prepared='/RAM/physical_prepare.json',physical_prepare_sha='4'*64))
    r['execution_binding']=dict(prepared=tp['arguments']['physical_prepared'],prepare_sha256='4'*64)
    tr=dict(prepared='/RAM/terminal_prepare.json',prepare_sha256='5'*64,artifact_sha256={})
    hashes={'/RAM/terminal/report.json':'1'*64,'/RAM/terminal_receipt.json':'2'*64,
        '/RAM/physical/report.json':'3'*64,'/RAM/physical_prepare.json':'4'*64,'/RAM/terminal_prepare.json':'5'*64,
        pp['planner_authority']['producer_prepared']:pp['planner_authority']['producer_prepare_sha256'],
        str(m.PROTOCOL):m.PROTOCOL_SHA}
    hashes.update({str(Path(mod.__file__).resolve()):value for mod,value in m.FROZEN})
    tp['bindings']={k:v for k,v in hashes.items() if k.startswith('/RAM/')}
    protocol=m.read(m.PROTOCOL)
    records={'/RAM/terminal/report.json':tr,'/RAM/terminal_receipt.json':a,'/RAM/terminal_prepare.json':tp,
        '/RAM/physical/report.json':r,'/RAM/physical_prepare.json':pp,
        pp['planner_authority']['producer_prepared']:c,str(m.PROTOCOL):protocol}
    calls=[]
    monkeypatch.setattr(m,'sha',lambda path:hashes.get(str(path),hashlib.sha256(str(path).encode()).hexdigest()))
    monkeypatch.setattr(m,'read',lambda path,*a:deepcopy(records[str(path)]))
    monkeypatch.setattr(m.terminal,'consume',lambda out,expected:calls.append((str(out),expected)) or deepcopy(a))
    monkeypatch.setattr(m.bridge,'load_pixels',pixels)
    monkeypatch.setattr(m.bridge,'verify_bindings',lambda b:None)
    config=dict(encoder=dict(image_size=224,patch_size=14,pretrained=False),projector=dict(input_dim=192,output_dim=192),
        action_encoder=dict(_target_='stable_worldmodel.wm.lewm.module.Embedder',input_dim=10,emb_dim=192),
        predictor=dict(_target_='stable_worldmodel.wm.lewm.module.Predictor',num_frames=3,input_dim=192,output_dim=192),
        pred_proj=dict(_target_='stable_worldmodel.wm.lewm.module.MLP',input_dim=192,output_dim=192))
    monkeypatch.setattr(m.bridge.rgb,'read',lambda *a:deepcopy(config))
    return args,records,hashes,calls,config


def test_prepare_full_terminal_consume_protocol_and_verify(monkeypatch):
    args,records,hashes,calls,config=setup_build(monkeypatch)
    p=m.build(**args)
    assert calls==[(args['terminal_output'],args['terminal_report_sha'])]
    assert p['bindings'][str(m.PROTOCOL)]==m.PROTOCOL_SHA
    assert p['analysis_protocol']==records[str(m.PROTOCOL)]['protocol']
    assert p['expected_encode_API']==4 and all(v==4 for v in p['expected_calls'].values())
    assert p['prediction_abi']['predict_output']==[1,3,192]
    records['/RAM/own.json']=p;assert m.verify('/RAM/own.json',m.sha('/RAM/own.json'))==p
    assert len(calls)==2


@pytest.mark.parametrize('fault',['receipt','external_sha','protocol','physical_prepare','config197','predictor_abi'])
def test_prepare_fails_closed_on_real_envelope_source_and_ABI(monkeypatch,fault):
    args,records,hashes,calls,config=setup_build(monkeypatch)
    if fault=='receipt':records[args['terminal_receipt']]['status']='relabel'
    elif fault=='external_sha':args['terminal_report_sha']='6'*64
    elif fault=='protocol':hashes[str(m.PROTOCOL)]='6'*64
    elif fault=='physical_prepare':records['/RAM/physical/report.json']['execution_binding']['prepare_sha256']='6'*64
    elif fault=='config197':config['encoder']['patch_size']=16
    else:config['predictor']['num_frames']=2
    with pytest.raises(ValueError):m.build(**args)


def saved_fixture(tmp_path,replay,monkeypatch):
    p,r0,s=replay;r=deepcopy(r0);p=deepcopy(p)
    p.update(arguments=dict(terminal_report_sha='1'*64),expected_model_hash='2'*64,bindings={},inherited_resources={})
    output=tmp_path/'output';output.mkdir();prepared=tmp_path/'prepare.json';prepared.write_bytes(m.raw(p))
    r.update(prepared=str(prepared),prepare_sha256=m.sha(prepared),model_unchanged=True,source_verified=True,
        model_hash_before='2'*64,model_hash_after='2'*64,elapsed_seconds=1.,
        P1_admission=False,scientific_admission=False,freshness=False,
        status='actual_midpoint_suffix4_predicted_not_admitted',event_count=len(s.events))
    files=dict(s.files)
    files.update({f'event_{i:04}.json':m.raw(e) for i,e in enumerate(s.events)})
    files['binding.json']=m.raw(dict(prepared=str(prepared),prepare_sha256=r['prepare_sha256']))
    files['initial_roster.json']=m.raw(p['rows'])
    files['native_seal.json']=m.raw({k:r[k] for k in ('rows','modules','encode_API','errors','blocked','precision_before','precision_after')})
    r['native_seal_sha256']=hashlib.sha256(files['native_seal.json']).hexdigest()
    r['artifact_sha256']={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    for n,b in files.items():(output/n).write_bytes(b)
    (output/'report.json').write_bytes(m.raw(r))
    monkeypatch.setattr(m,'verify',lambda path,expected:deepcopy(p))
    monkeypatch.setattr(m.bridge,'verify_bindings',lambda b:None)
    monkeypatch.setattr(m.bridge,'load_pixels',pixels)
    return output,r


def test_full_saved_consume_reads_all_artifacts_and_native_seal(tmp_path,replay,monkeypatch):
    output,r=saved_fixture(tmp_path,replay,monkeypatch)
    result=m.consume(str(output),m.sha(output/'report.json'))
    assert len(result['rows'])==4 and result['analysis_protocol_sha256']==m.PROTOCOL_SHA
    assert result['consume_new_neural_calls']==result['consume_new_physics']==0
    np.testing.assert_array_equal(m.load_saved_array(str(output),r['rows'][0]['actual_full3']),replay[2].load(r['rows'][0]['actual_full3']))


@pytest.mark.parametrize('fault',['extra_file','artifact_bytes','report_hash','model','admission','seal'])
def test_saved_consume_refuses_durable_mutations(tmp_path,replay,monkeypatch,fault):
    output,r=saved_fixture(tmp_path,replay,monkeypatch)
    if fault=='extra_file':(output/'extra').write_bytes(b'extra')
    elif fault=='artifact_bytes':(output/'event_0000.json').write_bytes(b'{}')
    elif fault=='model':r['model_hash_after']='3'*64
    elif fault=='admission':r['P1_admission']=True
    elif fault=='seal':r['native_seal_sha256']='3'*64
    (output/'report.json').write_bytes(m.raw(r));expected=m.sha(output/'report.json')
    if fault=='report_hash':expected='3'*64
    with pytest.raises(ValueError):m.consume(str(output),expected)


def test_frozen_helpers_and_budget_ABI():
    for module,expected in m.FROZEN:assert m.sha(module.__file__)==expected
    assert m.CAP==64*1024**2 and m.FINAL==8*1024**2 and m.FLOOR==200*1024**2 and m.WALL==600
    assert m.contract()['total_FLOPs'] is None and not m.contract()['equal_FLOPs']


def test_cli_uses_actual_terminal_report_and_receipt_no_future_defaults(monkeypatch):
    args=dict(terminal_output='/RAM/out',terminal_report_sha='a'*64,
        terminal_receipt='/RAM/receipt',terminal_receipt_sha='b'*64);calls=[]
    monkeypatch.setattr(m,'build',lambda *values:calls.append(dict(zip(args,values,strict=True))) or dict(RAM=True))
    monkeypatch.setattr(m,'write',lambda output,value:dict(RAM=True))
    command=['prepare','--output','/RAM/new']
    for k,v in args.items():command+=['--'+k.replace('_','-'),v]
    assert m.main(command)==0 and calls==[args]
