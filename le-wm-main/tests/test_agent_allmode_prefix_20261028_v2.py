"""RAM-only algebra/state-machine/native-interface stand-ins, no neural models."""
from copy import deepcopy
from io import BytesIO
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from experiments.spectral_reachability import agent_allmode_prefix_20261028_v2 as m

spec=importlib.util.spec_from_file_location('allmode_arrays',Path(__file__).with_name('test_moving_history_observed_suffix_cuda_20261027_v1.py'))
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


def slot(index=0):
    history=np.linspace(-.1,.1,576,dtype=np.float32).reshape(3,192).tolist()
    raw=np.asarray([[.1,0]]*5+[[0,.1]]*5,np.float32)
    context=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
        latent_history=history,action_history=m.kernel.normalized_history(raw).tolist())
    s,c=m.ORDER[index]
    return dict(index=index,root_slot=s,condition=c,task_id=str(index+1)*64,blind_sha256='b'*64,task_sha256='c'*64,
        packet=dict(root_context=context,goal=np.linspace(.1,-.1,192,dtype=np.float32).tolist()),q=np.eye(192).tolist())


class RAM:
    def __init__(self):self.files={};self.events=[]
    def _write(self,name,data,**kwargs):
        assert name not in self.files;self.files[name]=data
    def event(self,e):self.events.append(deepcopy(e))


class ArrayBackend:
    def __init__(self,row,store,modules=None,fail=None,flat=False):
        self.row,self.store,self.modules,self.fail=row,store,modules,fail
        row['forwards']=[];row['AD_API']=[]
        self.j=np.zeros((192,10),np.float32)
        if not flat:self.j[:10]=np.diag(np.linspace(.2,1.1,10,dtype=np.float32))
    def forward(self,c,raw,phase):
        raw=np.asarray(raw,np.float32);n=len(self.row['forwards'])
        e=dict(index=n,phase=phase,input=deepcopy(c),raw=raw.tolist(),completed=False)
        self.row['forwards'].append(e);self.store.event(dict(kind='forward_attempt',root=self.row['index'],entry=deepcopy(e)))
        e['raw_array']=m.capture.prior.save_array(self.store,f'root_{self.row["index"]}.raw_{n:03}.npy',raw)
        incoming=dict(latent_history=deepcopy(c['latent_history']),action_history=np.concatenate([c['action_history'],m.kernel.normalized_history(raw)]).tolist())
        z=(np.asarray(c['latent_history'][-1],np.float32)+self.j@raw.reshape(10)).astype(np.float32)
        out=dict(c,latent_history=deepcopy(c['latent_history'][1:])+[z.tolist()],action_history=incoming['action_history'][-2:])
        if self.modules:
            self.modules.meter.stage=n;self.modules.meter.expected_input=incoming
            ae=np.zeros((1,3,192),np.float32);ae[0,:,:10]=incoming['action_history']
            lat=np.asarray(c['latent_history'],np.float32)[None];po=lat.copy();po[0,-1]=z
            for kind,ins,result in (('action_encoder',[np.asarray(incoming['action_history'],np.float32)[None]],ae),
                    ('predict',[lat,ae],po),('pred_proj',[po.reshape(3,192)],po.reshape(3,192))):
                args=tuple(h.Tensor(a) for a in ins);self.modules.pre(kind)(kind,args)
                if n==self.fail and kind=='predict':
                    self.modules.post(kind,args,None);raise RuntimeError('RAM forward fault')
                self.modules.post(kind,args,h.Tensor(result))
        elif n==self.fail:raise RuntimeError('RAM forward fault')
        e.update(completed=True,output=out);self.store.event(dict(kind='forward_return',root=self.row['index'],entry=deepcopy(e)))
        return out
    def step(self,c,raw):return self.forward(c,raw,'step')
    def nominal(self,c):return np.asarray(self.forward(c,np.zeros((5,2),np.float32),'nominal')['latent_history'][-1])
    def jacobian(self,c,operation):
        def calculate():
            e=dict(index=len(self.row['AD_API']),completed=False,shape=[192,10],VJP_directions=192,
                functional_jacobian_calls=1,vectorize=True,engine_internal_calls='not independently intercepted')
            self.row['AD_API'].append(e);self.store.event(dict(kind='AD_attempt',root=self.row['index'],entry=deepcopy(e)))
            self.forward(c,np.zeros((5,2),np.float32),'jacobian_zero_forward')
            e['jacobian']=m.capture.prior.save_array(self.store,f'root_{self.row["index"]}.J_{e["index"]:02}.npy',self.j)
            e['completed']=True;self.store.event(dict(kind='AD_return',root=self.row['index'],entry=deepcopy(e)))
            return self.j.copy()
        return operation('jacobian',calculate)
    def replay(self,c,commands):
        for raw in np.asarray(commands,np.float32).reshape(-1,5,2):c=self.step(c,raw)
        return c


def execute(index=0,fail=None,flat=False):
    s=slot(index);store=RAM();row=dict(index=index,root_slot=s['root_slot'],condition=s['condition'],events=[])
    meter=NS(stage=None,expected_input=None,last_native=None)
    modules=m.Modules(NS(),store,f'root_{index}',meter)
    backend=ArrayBackend(row,store,modules,fail,flat)
    def sink(e):row['events'].append(deepcopy(e));store.event(dict(kind='compiler_event',root=index,entry=e))
    try:row['result']=m.compile_arm(s['packet'],s['q'],backend,sink)
    finally:row['module_ledger']=deepcopy(modules.entries);row['resources']=modules.counts()
    row.update(status='plan_sealed',plan=m.model_plan(s,row['result']))
    return s,row,store


def load(store,a):
    raw=store.files[a['file']];assert m.overlay.hash_bytes(raw)==a['sha256']
    return np.load(BytesIO(raw),allow_pickle=False)


@pytest.mark.parametrize('index',range(4))
def test_full42_and_real_shape_native_schedule258_no_extra_evaluation(index):
    s,r,st=execute(index);w=m.validate_execution(s,r,lambda a:load(st,a))
    assert len(r['forwards'])==258 and len(r['module_ledger'])==774 and len(r['AD_API'])==43
    assert r['plan']['commands']==w['commands'] and len(r['result']['candidates'])==42
    assert r['resources']=={k:dict(attempted=258,completed=258) for k in m.Modules.ORDER}
    assert len(r['result']['operations'])==86
    assert sum(e.get('SVD_complete_if_returned',0) for e in r['result']['operations'])==44
    assert r['plan']['condition']==m.ORDER[index][1]
    assert sum(map(len,st.files.values()))<5*1024**2


def test_all10mode_order_radii_signs_and_original_first2_subset():
    rng=np.random.default_rng(918);j=rng.normal(size=(192,10));n=np.zeros(192);g=np.ones(192);q=np.eye(192)
    b=m.all_mode_bank(j,n,g,q);old=m.compiler.spectral_bank(j,n,g,q)
    assert len(b['params'])==42 and b['params'][:2]==old['params'][:2]
    assert b['right_vectors']==old['right_vectors'] and b['singular_values']==old['singular_values']
    for ri,radius in enumerate(m.compiler.RADII):
        for mode in range(10):
            for si,sign in enumerate((-1,1)):
                index=2+ri*20+mode*2+si
                assert np.allclose(b['params'][index],sign*radius*np.asarray(b['right_vectors'])[mode])
                assert b['kinds'][index]==f'spectral_mode_{mode}_radius_{radius:.8f}_sign_{sign}'
    assert [b['params'][i] for i in (0,1,2,3,4,5,22,23,24,25)]==old['params']


def test_zero_singular_directions_and_duplicate_actions_kept_no_pruning():
    b=m.all_mode_bank(np.zeros((192,10)),np.zeros(192),np.zeros(192),np.eye(192))
    assert len(b['params'])==42 and len(b['singular_values'])==10
    s,r,st=execute(flat=True);m.validate_execution(s,r,lambda a:load(st,a))
    assert r['result']['winner']['index']==0
    assert len(r['forwards'])==258


@pytest.mark.parametrize('bad',['bank','dropmode','cost','stage','params','suffix','commands','history','prefix','context','winner','replay','operation'])
def test_saved_nonwinner_and_global_selection_mutations_rejected(bad):
    s=slot();row=dict(index=0);st=RAM();b=ArrayBackend(row,st)
    r=m.compile_arm(s['packet'],s['q'],b,lambda e:None);c=r['candidates'][9]
    if bad=='bank':r['bank']['params'][30][0]+=.01
    elif bad=='dropmode':r['candidates'].pop()
    elif bad=='cost':c['nonlinear_terminal_cost']+=1
    elif bad=='stage':c['stage_cost']+=1
    elif bad=='params':c['prefix_params'][0]+=.01
    elif bad=='suffix':c['suffix_solve']['own_reach_value']+=1
    elif bad=='commands':c['commands'][0][0]+=.01
    elif bad=='history':c['context']['action_history'][0][0]+=.001
    elif bad=='prefix':c['predicted_prefix'][0]+=.01
    elif bad=='context':c['context']['latent_history'][0][0]+=.01
    elif bad=='winner':r['winner']=deepcopy(c)
    elif bad=='replay':r['whole_replay_context']['latent_history'][-1][0]+=.01
    else:r['operations'].pop()
    with pytest.raises(ValueError):m.validate_compilation(s['packet'],s['q'],r)


@pytest.mark.parametrize('at',[0,1,2,3,4,5,7,253,257])
def test_failure_retains_attempt_native_inputs_and_entire_planned_denominator(at):
    s=slot();st=RAM();row=dict(index=0,events=[]);meter=NS(stage=None,expected_input=None,last_native=None)
    modules=m.Modules(NS(),st,'root_0',meter);b=ArrayBackend(row,st,modules,fail=at)
    with pytest.raises(RuntimeError,match='RAM forward fault'):
        m.compile_arm(s['packet'],s['q'],b,lambda e:row['events'].append(deepcopy(e)))
    assert len(row['forwards'])==at+1 and row['forwards'][-1]['completed'] is False
    assert len(modules.entries[-1]['inputs'])==2 and modules.entries[-1]['resource']=='predict'
    assert modules.counts()['predict']==dict(attempted=at+1,completed=at)
    assert st.files and len(m.ORDER)==4
    if at>=1:assert any(e['kind']=='nominal' for e in row['events'])
    if at>=4:assert any(e['kind']=='prefix' for e in row['events'])


def test_no_global_math_patch_and_fixed_contract():
    original=np.linalg.svd;s=slot();r=dict(index=0);st=RAM()
    m.compile_arm(s['packet'],s['q'],ArrayBackend(r,st),lambda e:None)
    assert np.linalg.svd is original
    c=m.contract();assert c['total']['predict']==1032 and c['total']['jacobians']==172
    assert c['total']['VJP_directions']==33024 and c['total']['SVD_successful_frozen_helper_calls']==176
    assert c['equal_FLOPs'] is c['P1_admission'] is c['P3_admission'] is False


def test_invalid_authority_stops_before_output_or_neural(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *args:(_ for _ in ()).throw(ValueError('source drift')))
    monkeypatch.setattr(m,'Store',lambda *a:pytest.fail('output before input verified'))
    with pytest.raises(ValueError,match='MAIN'):m.run('x','a'*64,'x','wrong')
    with pytest.raises(ValueError,match='source drift'):m.run('x','a'*64,'x',m.AUTH)


def test_cli_help_no_runtime(capsys):
    with pytest.raises(SystemExit) as e:m.main(['--help'])
    assert e.value.code==0 and 'consume' in capsys.readouterr().out


def test_RAM_full4_saved_consume_and_output_bound(monkeypatch):
    import gzip,hashlib,json
    files={};rows=[];slots=[];events=[]
    for i in range(4):
        s,r,st=execute(i);slots.append(s);rows.append(r)
        events.append(dict(kind='root_started',slot=deepcopy(s)));events.extend(st.events)
        events.append(dict(kind='root_sealed',index=i,selection_digest=r['plan']['selection_digest']))
        files.update(st.files);files[f'root_{i}.json.gz']=gzip.compress(m.canonical(r),mtime=0)
    p=dict(slots=slots,expected_model_hash='a'*64,contract=m.contract())
    files['binding.json']=m.canonical(dict(prepared='/RAM/prepare',prepare_sha256='b'*64))
    files['events.jsonl.gz']=gzip.compress(b''.join(m.canonical(e) for e in events),mtime=0)
    hashes={k:hashlib.sha256(v).hexdigest() for k,v in files.items()}
    report=dict(schema=m.NAME+'.report',status='all4_allmode_plans_sealed_not_admitted',errors=[],first_error=None,
        commands_released=True,model_unchanged=True,source_verified=True,contract=m.contract(),
        model_hash_before='a'*64,model_hash_after='a'*64,prepared='/RAM/prepare',prepare_sha256='b'*64,
        rows=rows,artifact_sha256=hashes,math_resources=m.math_resources(rows),resources={k:dict(attempted=1032,completed=1032) for k in m.Modules.ORDER})
    files['report.json']=m.canonical(report)
    assert len(files['report.json'])<m.FINAL and sum(map(len,files.values()))<m.CAP
    print('RAM full4 bytes',sum(map(len,files.values())),'report',len(files['report.json']))
    class VPath:
        def __init__(self,p):self.p=str(p)
        def __str__(self):return self.p
        def absolute(self):return self
        @property
        def name(self):return self.p.rsplit('/',1)[-1]
        def __truediv__(self,n):return VPath(self.p+'/'+str(n))
        def read_bytes(self):return files[self.name]
        def stat(self):return NS(st_size=len(self.read_bytes()))
        def is_symlink(self):return False
        def iterdir(self):return [self/n for n in files]
    def read(path,*a):
        raw=path.read_bytes();return json.loads(gzip.decompress(raw) if path.name.endswith('.gz') else raw)
    monkeypatch.setattr(m,'Path',VPath)
    monkeypatch.setattr(m,'sha',lambda path:hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(m.overlay,'read',read)
    monkeypatch.setattr(m,'verify',lambda *a:p)
    digest=hashlib.sha256(files['report.json']).hexdigest()
    value=m.consume('/RAM/out',digest)
    assert len(value['plans'])==4 and value['status']=='all4_allmode_commands_verified_not_admitted'
    assert [(x['root_slot'],x['condition']) for x in value['plans']]==m.ORDER
    assert all(x['whole_run_report_sha256']==digest for x in value['plans'])


def test_RAM_CLI_prepare_verify(monkeypatch,capsys):
    import json
    p=dict(schema=m.NAME+'.prepare',slots=[slot(i) for i in range(4)])
    monkeypatch.setattr(m,'build',lambda:deepcopy(p))
    path='/RAM/'+m.NAME+'_prepare.json'
    def write(path,raw,cap):
        assert path=='/RAM/'+m.NAME+'_prepare.json' and raw==m.canonical(p) and cap==m.PREP_CAP
        return dict(file=path,sha256='a'*64,bytes=len(raw))
    monkeypatch.setattr(m.capture,'write_new',write)
    assert m.main(['prepare','--output',path])==0
    assert json.loads(capsys.readouterr().out)['sha256']=='a'*64
    monkeypatch.setattr(m,'sha',lambda path:'a'*64)
    monkeypatch.setattr(m.overlay,'read',lambda *a:deepcopy(p))
    assert m.main(['verify','--prepared','/RAM/p','--prepare-sha','a'*64])==0
    assert json.loads(capsys.readouterr().out)['slots']==4


def test_real_saved_build_canonical_serialization_no_prepare(capsys):
    import json
    p=m.build();raw=m.canonical(p)
    assert json.loads(raw)==p and len(raw)<=m.PREP_CAP
    assert [(s['root_slot'],s['condition']) for s in p['slots']]==m.ORDER
    assert all(type(s['q']) is list and s['q']==np.eye(192).tolist() for s in p['slots'])
    for f in (Path(m.__file__).resolve(),m.ROOT/'tests'/('test_'+m.NAME+'.py'),m.ROOT/'innerdocs'/(m.NAME+'.md')):
        assert p['bindings'][str(f)]==m.sha(f)
    print('actual saved-only build bytes',len(raw),'slots',len(p['slots']))


@pytest.mark.parametrize('path,raw',[('/RAM/other_prepare.json',b'{}'),
    ('/RAM/agent_allmode_prefix_20261028_v2_prepare.gz',b'{}'),
    ('/RAM/agent_allmode_prefix_20261028_v2_prepare.json','{}'),
    ('/RAM/agent_allmode_prefix_20261028_v2_prepare.json',b'x'*(2*1024**2+1))])
def test_own_prepare_writer_prefix_bytes_and_cap_before_any_write(monkeypatch,path,raw):
    monkeypatch.setattr(m.capture,'write_new',lambda *a:pytest.fail('wrong stem/type/cap reached writer'))
    with pytest.raises(ValueError,match='prefix'):m.write_new(path,raw)


def test_v2_changes_only_engineering_capacity_contract():
    from experiments.spectral_reachability import agent_allmode_prefix_20261028_v1 as previous
    old=previous.contract(); new=m.contract()
    new['output_cap_bytes']=old['output_cap_bytes']
    new['final_reserve_bytes']=old['final_reserve_bytes']
    assert new==old
    assert m.CAP==512*1024**2 and m.FINAL==256*1024**2


def test_dense_real_row_upper_bound_fits_full4_final_report():
    import gzip,json
    old=m.ROOT/'results/agent_allmode_prefix_20261028_v1_cuda_MAIN01'
    rows=[json.loads(gzip.decompress((old/f'root_{i}.json.gz').read_bytes())) for i in range(3)]
    def bound(v):
        if v is None:return 4
        if type(v) is bool:return 5
        if type(v) in (float,int):return 32
        if isinstance(v,str):return len(json.dumps(v,ensure_ascii=True).encode())
        if isinstance(v,list):return 2+len(v)+sum(bound(x) for x in v)
        assert isinstance(v,dict)
        return 2+2*len(v)+sum(bound(k)+bound(x) for k,x in v.items())
    # Full duplicate-rich live rows, not low-entropy diagonal-J fixtures.
    projected_report=4*max(bound(r) for r in rows)+16*1024**2
    actual_artifacts=sum(p.stat().st_size for p in old.iterdir())
    assert projected_report<m.FINAL
    assert actual_artifacts*2+projected_report<m.CAP


def test_final_diagnostics_can_use_reserved_region_after_normal_write_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'CAP',8192);monkeypatch.setattr(m,'FINAL',4096);monkeypatch.setattr(m,'FLOOR',0)
    store=object.__new__(m.Store);store.output=tmp_path
    store._write('payload',b'x'*4000)
    with pytest.raises(ValueError,match='reserve'):store._write('blocked',b'x'*100)
    store._write('failure.json',b'{"commands_released":false}',final=True)
    assert not (tmp_path/'blocked').exists()
    assert (tmp_path/'failure.json').read_bytes()==b'{"commands_released":false}'

