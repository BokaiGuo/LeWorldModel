"""NumPy/device-labelled doubles and saved metadata only. No neural/physics."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_safeguard_factorial_cuda_20261028_v1 as m
import test_agent_p2_contextual_midfeedback_recompile_20261028_v2 as prior
import test_agent_p2_midfeedback_raw_jacobian_fd_20261028_v1 as fdtest
import test_agent_p2_midfeedback_terminal_rgb_20261028_v1 as visual


def after(b,commands):
    incoming=m.F.planner.incoming(b,b['observed_context'],commands)
    z=np.asarray(b['observed_context']['latent_history'][-1],np.float32).copy()
    z[:10]+=(np.asarray(commands,np.float32).reshape(10)*.5)
    return dict(b['observed_context'],latent_history=b['observed_context']['latent_history'][1:]+[z.astype(float).tolist()],
        action_history=incoming['action_history'][-2:])


def manifest():
    rows=[];requests=[];b=prior.branch();j=np.zeros((192,10));j[:10]=np.eye(10)*.5
    incafter=after(b,b['old_suffix_commands'])
    inc=dict(commands=b['old_suffix_commands'],raw_bits=m.source.v2.suffix_bits(b['old_suffix_commands']),
        prediction=dict(context=incafter,cost=m.source.cost(b,incafter),authorities=[dict(RAM=True)]))
    for i in range(8):
        axes=[]
        for anchor in ('zero','incumbent'):
            z=b['observed_context']['latent_history'][-1] if anchor=='zero' else incafter['latent_history'][-1]
            proposal=m.source.proposals(b,anchor,z,j);items=[]
            for k,u in enumerate(proposal['logits']):
                commands=m.source.v2.chart(u).astype(float).tolist();a=after(b,commands)
                pred=None if anchor=='zero' else dict(context=a,cost=m.source.cost(b,a),authorities=[dict(RAM=True)])
                items.append(dict(index=k+1,commands=commands,raw_bits=m.source.v2.suffix_bits(commands),prediction=pred))
                if anchor=='zero':requests.append(dict(request_index=4*i+k,branch=i,anchor=anchor,alpha_index=k+1,alpha=m.source.v2.ALPHAS[k],
                    logits=u,domain=m.source.v2.legality(b,u),commands=commands,raw_bits=m.source.v2.suffix_bits(commands),
                    input_context=deepcopy(b['observed_context']),context_sha256=m.F.core.digest(b['observed_context']),
                    raw_history10=deepcopy(b['observed_raw_history']),goal=deepcopy(b['goal']),action_mean=b['action_mean'],action_std=b['action_std'],
                    native_input=m.F.planner.incoming(b,b['observed_context'],commands),dispatch=dict(RAM=True)))
            axes.append(dict(anchor=anchor,candidates=items))
        rows.append(dict(index=i,root_slot=i//4,condition='off' if i%4<2 else 'on',arm=('joint80','conditional40')[i%2],task_id=str(i)*64,
            context_sha256=m.F.core.digest(b['observed_context']),goal_sha256=m.F.core.digest(b['goal']),
            prefix_raw_bits=m.source.v2.suffix_bits(b['prefix_commands']),incumbent=deepcopy(inc),controlled_axes=axes))
    return dict(schema=m.source.NAME+'.prepare',status='blocked_missing_model_candidates',contract=m.source.contract(),rows=rows,requests=requests)


@pytest.fixture(scope='module')
def prepared():
    return dict(schema=m.NAME+'.prepare',contract=m.contract(),**m.material(manifest()),bindings={},policy='RAM_ONLY',expected_model_hash='a'*64,
        arguments=dict(factorial_prepared='/RAM_ONLY',factorial_sha=m.FACTORIAL_SHA),inherited_compute=[],
        planned_new_compute=dict(predict=32),full_upstream_rebuilt_here=False,authority='pinned MAIN actual complete factorial prepare and every upstream binding SHA')


def execute(q,store):
    meter=NS(stage=None,expected_input=None,last_native=None);modules=m.Modules(NS(),store,f'r{q["request_index"]}',meter,m.schedule(q))
    point=dict(context=q['input_context'],raw5=np.zeros((5,2),np.float32).tolist(),anchor_context=q['input_context'],
        action_mean=q['action_mean'],action_std=q['action_std'])
    backend=fdtest.Backend(point,modules,meter);api=m.PredictAPI(backend.model,meter,store,lambda:None,m.schedule(q))
    row=dict(index=q['request_index'],status='not_started',AD_API_ledger=[],forbidden_ledger=[])
    with api:context=m.evaluate(q,backend,meter,store,row)
    row.update(module_ledger=deepcopy(modules.entries),predict_API_ledger=deepcopy(api.entries));m.counts(row)
    return row,context


@pytest.fixture(scope='module')
def replay(prepared):
    q=prepared['requests'][0];store=prior.Memory();row,context=execute(q,store)
    return q,row,context,store


def test_actual_native_hooks_predict_API_all_buffers_and_original_Q(replay):
    q,row,context,store=replay
    assert m.check_request(q,row,store.events,store.load)==context
    assert row['resources']=={k:dict(attempted=1,completed=1) for k in m.Modules.ORDER}
    assert row['predict_API_resources']==dict(predict=dict(attempted=1,completed=1))


@pytest.mark.parametrize('fault',['unknown','drop_intent','drop_input','drop_return','order','API','raw','Q','AD','SVD','pending','module','token','endpoint'])
def test_strict_native_trace_and_resource_mutations_failclosed(replay,fault):
    original,row0,_,store0=replay;q=deepcopy(original);row=deepcopy(row0);store=prior.Memory();store.files=dict(store0.files);events=deepcopy(store0.events)
    if fault=='unknown':events.insert(0,dict(kind='unknown'))
    elif fault.startswith('drop_'):
        kind={'drop_intent':'evaluation_intent','drop_input':'module_inputs_saved','drop_return':'module_return'}[fault]
        events.remove(next(e for e in events if e['kind']==kind))
    elif fault=='order':events[1],events[2]=events[2],events[1]
    elif fault=='API':row['predict_API_resources']['predict']['completed']=0
    elif fault=='raw':q['commands'][0][0]=float(np.float32(.99))
    elif fault=='Q':row['model_Q']+=1
    elif fault=='AD':row['AD_API_ledger']=[dict(resource='jacobian',completed=False)]
    elif fault=='SVD':row['forbidden_ledger']=[dict(kind='forbidden_operation',resource='SVD')]
    elif fault=='pending':row['status']='attempted'
    elif fault=='module':row['module_ledger'][1]['inputs']=[]
    else:
        key='token_array' if fault=='token' else 'latent_array';a=store.load(row[key]);a[-1,0]+=1
        row[key]=m.F.save_array(store,'mutated.npy',a)
    with pytest.raises((ValueError,IndexError)):m.check_request(q,row,events,store.load)


@pytest.mark.parametrize('fault',['drop','order','alpha','domain','chart','context','normalizer','goal','candidate'])
def test_material_literal_ABI_and_denominators(fault):
    p=manifest()
    if fault=='drop':p['requests'].pop()
    elif fault=='order':p['requests'][0],p['requests'][1]=p['requests'][1],p['requests'][0]
    elif fault=='alpha':p['requests'][0]['alpha']=.9
    elif fault=='domain':p['requests'][0]['domain']['legal']=False
    elif fault=='chart':p['requests'][0]['logits'][0]+=.1
    elif fault=='context':p['requests'][0]['input_context']['latent_history'][0][0]+=1
    elif fault=='normalizer':p['requests'][0]['action_mean'][0]+=1
    elif fault=='goal':p['rows'][0]['goal_sha256']='0'*64
    else:p['rows'][0]['controlled_axes'][1]['candidates'][0]['prediction']=None
    with pytest.raises(ValueError):m.material(p)


def test_all32_cell_seals_original_choose_no_physics_leak(prepared):
    contexts=[after(m.branch(q),q['commands']) for q in prepared['requests']]
    cells=m.seal(prepared,contexts)
    assert len(cells)==32 and [c['index'] for c in cells]==list(range(32))
    assert [(c['anchor'],c['guard']) for c in cells[:4]]==[('zero',False),('zero',True),('incumbent',False),('incumbent',True)]
    assert all(c['P1_admission'] is c['P2_admission'] is c['physical_release'] is False for c in cells)
    assert all(c['selection_digest']==m.F.core.digest({k:v for k,v in c.items() if k!='selection_digest'}) for c in cells)
    with pytest.raises(ValueError):m.seal(prepared,contexts[:31])
    p=deepcopy(prepared);p['selector_rows'][0]['axes'][0]['candidates'][0]['native_success']=True
    with pytest.raises(ValueError):m.seal(p,contexts)


@pytest.mark.parametrize('operation',['SVD','encoder','projector'])
def test_forbidden_operation_hooks_restore(operation):
    torch=visual.fake_torch();torch.svd=lambda *a:pytest.fail('SVD executed');torch.linalg=NS(svd=torch.svd,svdvals=torch.svd)
    model=visual.Model();events=[];oldnp=np.linalg.svd;oldtorch=torch.svd
    with pytest.raises(ValueError):
        with m.Forbidden(torch,model,events.append):
            if operation=='SVD':np.linalg.svd(np.eye(2))
            else:getattr(model,operation)(visual.Tensor(np.zeros((1,3,224,224),np.float32)))
    assert np.linalg.svd is oldnp and torch.svd is oldtorch and events[-1]['resource']==operation
    assert not model.encoder.pre and not model.projector.pre


@pytest.mark.parametrize('operation',['grad','jacobian','hessian','backward'])
def test_original_NoAD_hooks(operation):
    t=visual.fake_torch();events=[]
    with pytest.raises(ValueError):
        with m.NoAD(t,m.F.blind.ARMS[1],events.append,lambda:None):
            fn=getattr(t.autograd,operation) if operation in ('grad','backward') else getattr(t.autograd.functional,operation)
            fn(None)
    assert events[-1]['kind']=='forbidden_AD_API'


def test_pinned_authority_rejected_before_IO_and_sourcehash_before_bindings(monkeypatch):
    with pytest.raises(ValueError):m.build('/NOT_READ','0'*64)
    old=m.sha
    monkeypatch.setattr(m,'sha',lambda p:'0'*64 if str(p)==m.source.__file__ else old(p))
    monkeypatch.setattr(m.physical,'read',lambda *a:pytest.fail('bad source reached manifest'))
    with pytest.raises(ValueError):m.build('/NOT_READ',m.FACTORIAL_SHA)


def test_actual_saved_manifest_metadata_and_binding_tamper(monkeypatch):
    path=m.ROOT/'results/agent_p2_safeguard_factorial_20261028_v1_prepare_MAIN01.json'
    if not path.exists():pytest.skip('actual MAIN metadata unavailable, no substitute')
    assert m.sha(path)==m.FACTORIAL_SHA
    original=m.physical.read(path);assert len(m.material(original)['requests'])==32
    bad=deepcopy(original);bad['bindings'][str(Path(m.source.__file__).resolve())]='0'*64
    monkeypatch.setattr(m.physical,'read',lambda *a:bad)
    with pytest.raises(ValueError):m.build(path,m.FACTORIAL_SHA)


@pytest.fixture(scope='module')
def durable(tmp_path_factory,prepared):
    root=tmp_path_factory.mktemp('factorial');output=root/(m.NAME+'_RAM');store=m.Store(output,m.time.monotonic())
    prep=root/(m.NAME+'_prepare_RAM.json');prep.write_bytes(m.raw(prepared));expected=m.sha(prep)
    store.put('binding.json',dict(prepared=str(prep),prepare_sha256=expected));store.emit('run_intent',requests=32,contract=m.contract())
    rows=[];contexts=[]
    for i,q in enumerate(prepared['requests']):
        store.request_index=i;store.native=[];store.emit('request_intent',request_index=i,request_sha256=m.F.core.digest(q),budget={k:1 for k in m.Modules.ORDER})
        row,context=execute(q,store);assert m.check_request(q,row,store.native,lambda e:m.source.v1.load_npy(output,e))==context
        store.emit('request_return',request_index=i,status='recorded');name=f'request_{i:02d}.json';h=store.put(name,row,final=True)
        rows.append(dict(index=i,status='recorded',file=name,sha256=h,resources=row['resources'],predict_API_resources=row['predict_API_resources']));contexts.append(context)
    cells=m.seal(prepared,contexts);selection=dict(file='selections.json',sha256=store.put('selections.json',cells))
    status='all32_forward_and_cells_sealed_science_blocked';store.emit('run_finished',status=status)
    report=dict(schema=m.NAME+'.report',status=status,contract=m.contract(),rows=rows,errors=[],selection=selection,
        prepared=str(prep),prepare_sha256=expected,model_before=prepared['expected_model_hash'],model_after=prepared['expected_model_hash'],elapsed_seconds=1.,
        journal_chain_tail=store.previous,artifact_sha256={f.name:m.sha(f) for f in output.iterdir()},fixed_request_denominator=32,fixed_cell_denominator=32,
        AD=0,SVD=0,new_visual=0,new_physics=0,full_upstream_rebuilt_here=False,authority=prepared['authority'],P1_admission=False,P2_admission=False,physical_release=False)
    store.put('report.json',report,final=True)
    return output,m.sha(output/'report.json'),prepared


def test_complete_saved_consume_file_arrays_journal_and_all32_seals(durable,monkeypatch):
    output,h,p=durable;monkeypatch.setattr(m,'verify',lambda *a:p)
    result=m.consume(output,h)
    assert len(result['cells'])==32 and result['actual_new_modules']['predict']==dict(attempted=32,completed=32)
    assert result['full_upstream_rebuilt_here'] is False and result['fair_budget']['guard_extra_predict']==0
    assert result['physical_metrics'] is None and result['new_physics']==result['consume_model_calls']==0


def test_rehashed_unknown_payload_not_filtered_by_consumer(durable,tmp_path,monkeypatch):
    import shutil
    original,_,p=durable;output=tmp_path/(m.NAME+'_unknown');shutil.copytree(original,output)
    r=m.physical.read(output/'report.json');events=m.physical.journal(output/'journal.jsonl',r['journal_chain_tail'])
    events[1]['unbudgeted_field']=True;previous=None
    for e in events:
        e['previous_sha256']=previous;e['event_sha256']=m.kernel.source.digest({k:v for k,v in e.items() if k!='event_sha256'});previous=e['event_sha256']
    (output/'journal.jsonl').write_bytes(b''.join(m.kernel.raw(e) for e in events))
    r['journal_chain_tail']=previous;r['artifact_sha256']['journal.jsonl']=m.sha(output/'journal.jsonl')
    (output/'report.json').write_bytes(m.raw(r));monkeypatch.setattr(m,'verify',lambda *a:p)
    assert len(m.physical.journal(output/'journal.jsonl',previous))==len(events)
    with pytest.raises(ValueError,match='canonical hashchain and payload ABI'):m.consume(output,m.sha(output/'report.json'))


@pytest.mark.parametrize('fault',['extra_file','array_bytes','missing_row','fake_FLOPs','unknown_journal','selection','partial'])
def test_saved_output_tamper_rejected(durable,tmp_path,monkeypatch,fault):
    import shutil
    original,_,p=durable;output=tmp_path/(m.NAME+'_mutated');shutil.copytree(original,output)
    monkeypatch.setattr(m,'verify',lambda *a:p);r=m.physical.read(output/'report.json')
    if fault=='extra_file':(output/'extra').write_bytes(b'x')
    elif fault=='array_bytes':next(output.glob('*.npy')).write_bytes(b'bad')
    elif fault=='missing_row':r['rows'].pop()
    elif fault=='fake_FLOPs':r['contract']['equal_FLOPs']=True
    elif fault=='partial':r['status']='partial_or_failed'
    elif fault=='unknown_journal':(output/'journal.jsonl').write_bytes((output/'journal.jsonl').read_bytes()+b'{}\n')
    else:
        f=output/'selections.json';cells=m.physical.read(f);cells[0]['selection']['model_Q']+=1;f.write_bytes(m.raw(cells))
        r['artifact_sha256']['selections.json']=m.sha(f);r['selection']['sha256']=m.sha(f)
    (output/'report.json').write_bytes(m.raw(r))
    with pytest.raises((ValueError,KeyError)):m.consume(output,m.sha(output/'report.json'))


def test_timeout_failure_final_reserve_keeps32_and_no_fake_completion(tmp_path,prepared,monkeypatch):
    clock=[0.];monkeypatch.setattr(m,'time',NS(monotonic=lambda:clock[0]))
    monkeypatch.setattr(m,'verify',lambda *a:prepared);monkeypatch.setattr(m.bridge,'verify_bindings',lambda *a:None)
    fake=NS();attention=NS(sdpa_kernel=lambda *a:nullcontext(),SDPBackend=NS(MATH='MATH'))
    for key,value in (('torch',fake),('torch.nn',NS()),('torch.nn.attention',attention),
        ('experiments.checkpoint_compat',NS(load_compatible_pretrained=lambda *a,**kw:pytest.fail('model must not load')))):
        monkeypatch.setitem(sys.modules,key,value)
    def denied(*a):clock[0]=1000.;raise ValueError('RAM simulated allocation/time failure')
    monkeypatch.setitem(sys.modules,'experiments.gpu_preflight',NS(require_cuda_allocation=denied))
    prep=tmp_path/(m.NAME+'_prepare_RAM.json');prep.write_bytes(m.raw(prepared));output=tmp_path/(m.NAME+'_failed')
    with pytest.raises(ValueError):m.run(prep,m.sha(prep),output,m.AUTH)
    r=m.physical.read(output/'report.json')
    assert r['status']=='partial_or_failed' and r['selection'] is None and len(r['rows'])==32 and r['errors']
    assert all(x['status']=='not_started' and x['resources']['predict']['attempted']==0 for x in r['rows'])
    assert m.physical.journal(output/'journal.jsonl',r['journal_chain_tail'])[-1]['kind']=='run_finished'


def test_run_dispatch_hooks_all32_and_saved_consume_without_neural(tmp_path,prepared,monkeypatch):
    class Module(visual.Module):
        def register_forward_hook(self,fn,*,always_call=False):
            assert always_call is True
            return super().register_forward_hook(fn)
    class Model:
        def __init__(self):
            self.encoder=Module(lambda *a:pytest.fail('visual forbidden'))
            self.projector=Module(lambda *a:pytest.fail('visual forbidden'))
            self.action_encoder=Module(lambda x:visual.Tensor(x.a.copy()))
            def predict(lat,ae):
                out=lat.a.copy();out[0,-1]+=np.float32(ae.a[0,-1].sum()*.001)
                return visual.Tensor(out)
            self.predictor=Module(predict);self.pred_proj=Module(lambda x:visual.Tensor(x.a.copy()));self.calls=0
        def eval(self):return self
        def requires_grad_(self,value):assert value is False;return self
        def predict(self,lat,ae):
            result=self.predictor(lat,ae);return self.pred_proj(visual.Tensor(result.a.reshape(3,192)))
    class Backend:
        def __init__(self,model,device):self.model=model;assert device.type=='cuda'
        def step(self,c,commands):
            q=prepared['requests'][self.model.calls];self.model.calls+=1
            incoming=m.F.planner.incoming(m.branch(q),c,commands)
            ae=self.model.action_encoder(visual.Tensor(np.asarray(incoming['action_history'],np.float32)[None]))
            out=self.model.predict(visual.Tensor(np.asarray(incoming['latent_history'],np.float32)[None]),ae)
            return dict(c,latent_history=c['latent_history'][1:]+[out.a[-1].astype(float).tolist()],action_history=incoming['action_history'][-2:])
    torch=visual.fake_torch();torch.svd=lambda *a:pytest.fail('SVD forbidden');torch.linalg=NS(svd=torch.svd,svdvals=torch.svd)
    torch.device=lambda *a:NS(type='cuda');torch.cuda.synchronize=lambda *a:None
    torch.inference_mode=lambda:nullcontext();torch.autocast=lambda **kw:nullcontext()
    torch.backends=NS(cuda=NS(matmul=NS(allow_tf32=True)),cudnn=NS(allow_tf32=True))
    model=Model();allocations=[]
    for key,value in (('torch',torch),('torch.nn',NS()),('torch.nn.attention',NS(sdpa_kernel=lambda *a:nullcontext(),SDPBackend=NS(MATH='MATH'))),
        ('experiments.checkpoint_compat',NS(load_compatible_pretrained=lambda *a,**kw:model)),
        ('experiments.gpu_preflight',NS(require_cuda_allocation=lambda *a:allocations.append('RAM')))):
        monkeypatch.setitem(sys.modules,key,value)
    monkeypatch.setattr(m,'verify',lambda *a:prepared);monkeypatch.setattr(m.F.d2p,'TorchBackend',Backend)
    monkeypatch.setattr(m.bridge.native,'frozen',lambda *a:None);monkeypatch.setattr(m.bridge.native,'tensor_hash',lambda *a:prepared['expected_model_hash'])
    prep=tmp_path/(m.NAME+'_prepare_RAM.json');prep.write_bytes(m.raw(prepared));output=tmp_path/(m.NAME+'_run_RAM')
    r=m.run(prep,m.sha(prep),output,m.AUTH);result=m.consume(output,r['report_sha256'])
    assert model.calls==32 and allocations==['RAM'] and len(result['cells'])==32
    assert torch.backends.cuda.matmul.allow_tf32 is torch.backends.cudnn.allow_tf32 is True
    assert all(not x.pre and not x.post for x in (model.encoder,model.projector,model.action_encoder,model.predictor,model.pred_proj))


def test_AUTH_before_all_IO_and_exact_limits():
    with pytest.raises(ValueError):m.run('/NO','0'*64,'/NO','unauthorized')
    c=m.contract();assert c['requests']==c['cells']==32 and c['AD']==c['SVD']==c['physics']==0
    assert (c['wall_seconds'],c['output_cap_bytes'],c['final_reserve_bytes'],c['disk_floor_bytes'])==(900,64*1024**2,2*1024**2,200*1024**2)
