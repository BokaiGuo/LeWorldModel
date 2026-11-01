"""RAM/mock only: never call real prepare, torch, model, CUDA or physics."""
from copy import deepcopy
import gzip
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import zlib
import numpy as np
import pytest
from experiments.spectral_reachability import p1_moving_overlay_recompile_cuda_20261027_v1 as m


def helper(stem):
    spec=importlib.util.spec_from_file_location(stem,Path(__file__).with_name(stem+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


h=helper('test_moving_history_planners_cuda_20261027_v1')


class RAM(m.Store):
    def __init__(self):
        self.output=Path('/RAM/run');self.files={};self.comp=zlib.compressobj(6,zlib.DEFLATED,31)
        self.closed=self.broken=False;self.check=lambda:None;self._write('events.jsonl.gz',b'')
    def _write(self,name,raw,append=False,final=False):
        assert append or name not in self.files
        assert sum(map(len,self.files.values()))+len(raw)<=m.CAP-(0 if final else m.FINAL)
        self.files[name]=self.files.get(name,b'')+raw


def fixture():
    packets={};exports=[]
    for s in (0,1):
        artifacts={}
        for ci,c in enumerate(m.CONDITIONS):
            rgb=np.stack([np.full((224,224,3),20*s+10*ci+j,np.uint8) for j in range(3)])
            goal=np.full((224,224,3),20*s+10*ci+4,np.uint8)
            packet=dict(rgb_history=rgb,goal_rgb=goal,raw_action_history=m.moving.fixed_raw())
            path=f's{s}_{c}_blind_inputs.npz';hs=m.hash_bytes(path.encode())
            packets['/RAM/source/'+path]=packet
            artifacts[c]=dict(path=path,sha256=hs,fields=['rgb_history','raw_action_history','goal_rgb'],
                rgb_history_pixel_sha256=[m.hash_bytes(x.tobytes()) for x in rgb],goal_pixel_sha256=m.hash_bytes(goal.tobytes()),
                raw_action_sha256=m.hash_bytes(packet['raw_action_history'].tobytes()))
        exports.append(dict(source_slot=s,artifacts=artifacts))
    c=dict(flags=[False,True,False],no_goal_pose_edit=True,warmup=10,horizon=10,slots=2,source_triplets=[0,5,10,15,20],caps=dict(public_step=120))
    r=dict(schema='p1_moving_target_overlay_source_20261027_v1.report',status='paired_moving_overlay_recorded',
        full_protocol=True,source_verified=True,errors=[],first_error=None,fixed_denominator=2,
        expected_episodes=6,expected_frames=186,contract=c,counts=dict(public_step=dict(attempted=120,completed=120)),blind_exports=exports)
    def load(path,expected):return deepcopy(packets[str(path)])
    slots=m.packets(r,Path('/RAM/source'),load)
    return dict(slots=slots,contract=m.contract(),expected_model_hash='a'*64,
        encoder_output_shape=m.encoder_output_shape(dict(encoder=dict(image_size=224,patch_size=14),projector=dict(input_dim=192)))),r,packets,load


def test_packet_firewall_fixed_four_real_conditions_and_metadata():
    p,r,_,load=fixture()
    assert [(s['root_slot'],s['condition']) for s in p['slots']]==[(s,c) for s in (0,1) for c in ('off','on')]
    assert m.contract()['encoder']==16 and m.contract()['predict']==576
    assert m.contract()['per_root_condition'][m.ARMS[0]]['predict']==78
    assert m.contract()['per_root_condition'][m.ARMS[1]]['predict']==66
    assert m.contract()['wall_seconds']==900 and m.CAP==64*1024**2
    assert not m.contract()['edit_goal_pose'] and not m.contract()['edit_state']


@pytest.mark.parametrize('bad',['partial','root','condition','path','raw_sha','pixel','count','goal_edit','warmup'])
def test_metadata_mismatch_fails_before_any_neural_work(bad):
    p,r,packets,load=fixture()
    if bad=='partial':r['full_protocol']=False
    elif bad=='root':r['blind_exports'][1]['source_slot']=0
    elif bad=='condition':del r['blind_exports'][0]['artifacts']['on']
    elif bad=='path':r['blind_exports'][0]['artifacts']['on']['path']='../witness.npz'
    elif bad=='raw_sha':r['blind_exports'][0]['artifacts']['off']['raw_action_sha256']='0'*64
    elif bad=='pixel':r['blind_exports'][0]['artifacts']['on']['goal_pixel_sha256']='0'*64
    elif bad=='count':r['counts']['public_step']['attempted']=119
    elif bad=='goal_edit':r['contract']['no_goal_pose_edit']=False
    else:packets['/RAM/source/s0_on_blind_inputs.npz']['raw_action_history'][:]=0
    with pytest.raises((ValueError,KeyError)):m.packets(r,Path('/RAM/source'),load)


def export(condition='on'):
    p,_,_,load=fixture();slot=p['slots'][m.CONDITIONS.index(condition)]
    packet=load(slot['blind_path'],slot['blind_sha256'])
    return m.task_export(p,'1'*64,slot,packet,[np.full(192,.1*k,np.float32) for k in range(4)])


def test_own_on_interface_never_calls_old_off_only_loaders(monkeypatch):
    monkeypatch.setattr(m.moving,'load_task',lambda *a:pytest.fail('off-only moving loader'))
    monkeypatch.setattr(m.blind,'load_export',lambda *a:pytest.fail('old zero/off loader'))
    monkeypatch.setattr(m.blind,'interface',lambda *a:pytest.fail('old zero-only gate'))
    on,off=export(),export('off')
    assert m.task_interface(on)['task_id']!=m.task_interface(off)['task_id']
    assert on['task_contract']['with_target'] is True and off['task_contract']['with_target'] is False
    assert np.array_equal(on['task']['root_raw_action_history'],m.moving.fixed_raw())
    assert np.array_equal(on['task']['projection'],np.eye(192))
    assert set(m.task_interface(on)['d2p_packet'])=={'root_context','goal'}


@pytest.mark.parametrize('bad',['condition','flag','Q','std','zero','horizon','id','precision'])
def test_own_task_contract_has_no_permission_or_numeric_relaxation(bad):
    e=export();t,c=e['task'],e['task_contract']
    if bad=='condition':c['condition']='other'
    elif bad=='flag':c['with_target']=False
    elif bad=='Q':t['projection'][0][0]=0
    elif bad=='std':t['action_std']=[1.,1.]
    elif bad=='zero':t['root_raw_action_history']=np.zeros((10,2)).tolist()
    elif bad=='horizon':t['horizon']=1
    elif bad=='id':e['task_id']='2'*64
    else:t['goal'][0]=.1
    with pytest.raises(ValueError):m.task_interface(e)


def simulate():
    """Real array-only GN/D2P kernels, synthetic module outputs and explicit mock AD receipts."""
    p,r,packets,load=fixture();store=RAM();rows=m.initial_rows();images=[];ledger=[]
    def module(kind,scope,inputs,output):
        e=dict(index=len(ledger),resource=kind,scope=scope,completed=False)
        store.event(dict(kind='tensor_module_attempt',entry=deepcopy(e)))
        e['inputs']=[store.array(f'module_{e["index"]:04}_input{j}',x) for j,x in enumerate(inputs)]
        e['output']=store.array(f'module_{e["index"]:04}_output',output);e['completed']=True
        ledger.append(e);store.event(dict(kind='tensor_module_completed',entry=deepcopy(e)));return e
    def encode(rgb,image):
        # Synthetic RAM pixels-only preprocessing, not a claim of actual native preprocessing.
        pre=np.transpose(rgb.astype(np.float32),(2,0,1))[None,None]
        z=np.full(192,float(rgb[0,0,0])*.01,np.float32);out=np.broadcast_to(z,tuple(p['encoder_output_shape'])).copy()
        scope=dict(image_index=image['image_index'])
        image['preprocessed']=store.array(f'image_{image["image_index"]:02}_preprocessed',pre)
        image['modules']=[module('encoder',scope,[pre.reshape(1,3,224,224)],out),module('projector',scope,[out[:,0]],z[None])]
        return z
    def runner(arm,i,sink,row):
        events=[]
        def journal(e):events.append(deepcopy(e));sink(e)
        if arm==m.ARMS[0]:result=h.mock_runner(arm,i,journal,row)
        else:
            ledger=m.blind.D2PLedger(journal);context_index=0
            class FP32Mock(h.h.DBackend):
                def step(self,c,a):
                    nonlocal context_index
                    journal(dict(kind='native_context_attempt',index=context_index,context=m.history.incoming(i['gn_task'],c,a)))
                    context_index+=1
                    after=super().step(c,a)
                    after['latent_history'][-1]=np.asarray(after['latent_history'][-1],np.float32).astype(float).tolist()
                    return after
            result=m.blind.d2p.compile_arm(i['d2p_packet'],i['q'],FP32Mock(ledger),ledger.journal)
            row['resources']=ledger.counts()
            caps=m.blind.contract(1)['AD_API_per_root'][arm]
            for index,kind in enumerate([k for k,n in caps.items() for _ in range(n)]):
                e=dict(index=index,resource=kind,completed=False,is_grads_batched=True)
                journal(dict(kind='AD_API_attempt',entry=deepcopy(e)))
                journal(dict(kind='AD_API_completed',entry=dict(e,completed=True)))
            row['AD_API_resources']={k:dict(attempted=v,completed=v) for k,v in caps.items()}
        choice=m.blind.selected(arm,i,result)
        contexts=[e['context'] for e in events if e['kind']=='native_context_attempt'];modules=[]
        for j,c in enumerate(contexts):
            scope=dict(plan_index=row['index']);ae=np.zeros((1,3,192),np.float32)
            output=np.zeros((3,192),np.float32)
            if j==len(contexts)-1:output[-1]=choice['prediction']['latent_history'][-1]
            modules.extend([module('action_encoder',scope,[np.asarray(c['action_history'],np.float32)[None]],ae),
                module('predictor',scope,[np.asarray(c['latent_history'],np.float32)[None],ae],output[None]),
                module('pred_proj',scope,[output],output)])
        row['modules']=modules
        return result
    store.put('binding.json',dict(prepared='/RAM/prepared.json',prepare_sha256='1'*64))
    m.execute(p,'1'*64,store,encode,runner,rows,images,load=load);store.close()
    report=dict(schema=m.NAME+'.report',status='overlay_recompiled_plans_sealed_not_admitted',rows=rows,images=images,
        errors=[],model_unchanged=True,source_verified=True,commands_released=True,fixed_denominator=8,
        prepared='/RAM/prepared.json',prepare_sha256='1'*64,contract=p['contract'],model_hash_before='a'*64,model_hash_after='a'*64,
        module_ledger=ledger,artifact_sha256={n:m.hash_bytes(v) for n,v in store.files.items()})
    store.put('report.json',report,final=True)
    return p,store,report,load


@pytest.fixture(scope='module')
def full():return simulate()


def ram_consume(monkeypatch,full):
    p,store,report,load=full
    monkeypatch.setattr(m,'verify',lambda *a:deepcopy(p))
    monkeypatch.setattr(m,'load_packet',load)
    monkeypatch.setattr(m,'sha',lambda path:m.hash_bytes(store.files[Path(path).name]))
    def read(path,*a):
        raw=store.files[Path(path).name]
        if str(path).endswith('.gz'):raw=gzip.decompress(raw)
        return m.blind.decode(raw)
    monkeypatch.setattr(m,'read',read)
    original=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda path:store.files[path.name] if path.parent==store.output else original(path))
    monkeypatch.setattr(Path,'iterdir',lambda path:iter([store.output/n for n in store.files]))
    return m.consume(store.output,m.hash_bytes(store.files['report.json']))


def test_full8_two_kernels_counts_and_stable_consumer_interface(monkeypatch,full):
    p,store,report,load=full
    plans=ram_consume(monkeypatch,full)
    assert len(plans)==8 and [(x['root_slot'],x['condition'],x['arm']) for x in plans]==list(m.KEYS)
    assert len(report['images'])==16
    assert len(store.files['report.json'])<=m.FINAL
    for plan in plans:
        assert {'root_slot','condition','arm','commands','selection_digest','task_id','blind_path','blind_sha256',
                'source_report_sha256','planner_report_sha256'}<=set(plan)
        assert np.asarray(plan['commands']).shape==(10,2) and Path(plan['blind_path']).is_absolute()
        assert len(plan['selection_digest'])==len(plan['planner_report_sha256'])==64
        assert plan['selection_uses_physics'] is False


@pytest.mark.parametrize('index',[0,3,7])
def test_failure_preserves_fixed8_and_does_not_retry_or_encode_more(index):
    p,r,packets,load=fixture();store=RAM();rows=m.initial_rows();images=[];calls=[]
    def encode(rgb,image):
        image['modules']=[dict(resource=k,completed=True) for k in ('encoder','projector')]
        return np.zeros(192,np.float32)
    def runner(arm,i,sink,row):
        calls.append(row['index']);row['resources']=dict(predict=dict(attempted=1,completed=0))
        raise RuntimeError('mock failed after actual attempt')
    # Start failure at a later row without executing previous kernels: isolate scheduler suffix.
    if index:
        p['slots']=p['slots'][index//2:]
        target=2*p['slots'][0]['order']
    else:target=0
    with pytest.raises(RuntimeError):m.execute(p,'1'*64,store,encode,runner,rows,images,load=load)
    assert calls==[target] and len(rows)==8 and rows[target]['resources']['predict']==dict(attempted=1,completed=0)
    assert all(r['status']=='not_started' for r in rows[target+1:]) and len(images)==4


@pytest.mark.parametrize('bad',['partial','row_order','model','extra','tensor_bytes','selection_bytes'])
def test_consume_never_releases_partial_or_tampered_output(monkeypatch,full,bad):
    p,old,r,load=full;store=RAM();store.files=deepcopy(old.files);report=deepcopy(r)
    if bad=='partial':report['status']='partial_or_failed'
    elif bad=='row_order':report['rows'].reverse()
    elif bad=='model':report['model_unchanged']=False
    elif bad=='extra':store.files['unknown']=b'bad'
    elif bad=='tensor_bytes':store.files['module_0000_output.npy.gz']+=b'bad'
    else:store.files['plan_00.plan.json']+=b'bad'
    store.files['report.json']=m.canonical(report)
    with pytest.raises((ValueError,KeyError)):ram_consume(monkeypatch,(p,store,report,load))


def test_traced_scope_preserves_original_error_and_partial_receipt():
    trace=NS(scope=None,entries=[dict(scope=dict(plan_index=0),resource='predictor',completed=False)],
        finish=lambda s:(_ for _ in ()).throw(ValueError('incomplete')))
    row={}
    with pytest.raises(RuntimeError,match='original') as exc:
        with m.traced_scope(trace,dict(plan_index=0),row):raise RuntimeError('original')
    assert trace.scope is None and row['modules'][0]['completed'] is False and exc.value.__notes__


def test_no_actual_prepare_or_runtime_called_in_tests(monkeypatch):
    monkeypatch.setattr(m,'build',lambda:pytest.fail('real prepare/build forbidden in this regression'))
    assert m.initial_rows()[7]['arm']==m.ARMS[1]
    assert m.contract()['Qfit'] is False and m.contract()['no_CPU_neural'] is True


def test_patch14_config_derives257_and_real257_cls_wiring():
    shape=m.encoder_output_shape(dict(encoder=dict(image_size=224,patch_size=14),projector=dict(input_dim=192)))
    assert shape==[1,257,192]
    pre=np.zeros((1,1,3,224,224),np.float32)
    eout=np.arange(257*192,dtype=np.float32).reshape(1,257,192)
    z=np.full(192,3,np.float32)
    m.visual_bridge(pre,pre[0],eout,eout[:,0],z[None],z,shape)


@pytest.mark.parametrize('shape',[(1,197,192),(1,256,192),(1,258,192),(1,257,191)])
def test197_and_any_non257_encoder_shape_rejected(shape):
    pre=np.zeros((1,1,3,224,224),np.float32);out=np.zeros(shape,np.float32);z=np.zeros(192,np.float32)
    with pytest.raises(ValueError,match='257'):
        m.visual_bridge(pre,pre[0],out,out[:,0],z[None],z,[1,257,192])


@pytest.mark.parametrize('patch',[16,7,28,True])
def test_config_cannot_relax_frozen_patch14(patch):
    with pytest.raises(ValueError):
        m.encoder_output_shape(dict(encoder=dict(image_size=224,patch_size=patch),projector=dict(input_dim=192)))


def test_main_preregistration_is_in_source_closure_without_prepare():
    assert m.PREREG in m.source_files()
    assert m.PREREG.name=='D2P_MOVING_OVERLAY_RECOMPILE_PROTOCOL_20261027.md'
    assert len(m.PREREG_SHA)==64
