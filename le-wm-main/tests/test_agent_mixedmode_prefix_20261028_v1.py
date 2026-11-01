"""RAM arithmetic/interface stand-ins and saved metadata, never neural/physics."""
from copy import deepcopy
from io import BytesIO
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from experiments.spectral_reachability import agent_mixedmode_prefix_20261028_v1 as m

spec=importlib.util.spec_from_file_location('sealed_allmode_RAM',Path(__file__).with_name('test_agent_allmode_prefix_20261028_v2.py'))
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


def execute(index=0,fail=None,flat=False):
    s=h.slot(index);st=h.RAM();r=m.census(dict(slots=[h.slot(i) for i in range(4)]))[index];r['events']=[]
    meter=NS(stage=None,expected_input=None,last_native=None);mods=m.Modules(NS(),st,f'root_{index}',meter)
    b=h.ArrayBackend(r,st,mods,fail,flat)
    def sink(e):r['events'].append(deepcopy(e));st.event(dict(kind='compiler_event',root=index,entry=e))
    try:r['result']=m.compile_arm(s['packet'],s['q'],b,sink)
    finally:r['module_ledger']=deepcopy(mods.entries);r['resources']=mods.counts()
    v=dict(attempted=True,completed=False,SVD_successful_helpers=0,internal_SVD_on_failure='unknown',neural_calls=0)
    r['saved_validation']=v;st.event(dict(kind='saved_validation_attempt',root=index,entry=deepcopy(v)))
    m.validate_compilation(s['packet'],s['q'],r['result'])
    v.update(completed=True,SVD_successful_helpers=224)
    st.event(dict(kind='saved_validation_return',root=index,entry=deepcopy(v)))
    r.update(status='plan_sealed',plan=m.model_plan(s,r['result']))
    return s,r,st


def load(st,a):
    raw=st.files[a['file']];assert hashlib.sha256(raw).hexdigest()==a['sha256']
    return np.load(BytesIO(raw),allow_pickle=False)


@pytest.fixture(scope='module')
def complete():return execute()


def test_frozen_contract_no_changes_old_kernels():
    c=m.contract();assert c['total']==dict(predict=5352,action_encoder=5352,pred_proj=5352,predictor_samples=5352,
        AD_forward=892,jacobians=892,VJP_directions=171264,SVD_successful_helpers=896,suffix_solves=888)
    assert m.Backend.jacobian is m.base.Backend.jacobian and m.Backend.replay is m.base.Backend.replay
    assert m.Modules.post is m.base.Modules.post and m.Modules.__enter__ is m.base.Modules.__enter__
    assert c['equal_FLOPs'] is c['P1_admission'] is c['P3_admission'] is False
    assert c['beta']==1 and c['damping']==.002 and c['retries']==c['padding']==0


@pytest.mark.parametrize('rank',[0,3,10])
def test_exact_original42_and_fixed180_pairwise_bank(rank):
    rng=np.random.default_rng(20261028);j=rng.normal(size=(192,10));j[:,rank:]=0
    n=rng.normal(size=192);g=rng.normal(size=192);q=np.eye(192)
    old=m.base.all_mode_bank(j,n,g,q);b=m.mixed_bank(j,n,g,q)
    assert b['params'][:42]==old['params'] and b['kinds'][:42]==old['kinds']
    assert b['right_vectors']==old['right_vectors'] and len(b['params'])==222
    right=np.asarray(b['right_vectors']);k=42
    for i in range(9):
        for j in range(i+1,10):
            for si in (-1,1):
                for sj in (-1,1):
                    w=b['parameter_witnesses'][k];before=m.compiler.CAP*(si*right[i]+sj*right[j])/np.sqrt(2.)
                    # Sealed rowwise screen uses sum over the last axis, not
                    # BLAS dot's reduction order used by norm(single_vector).
                    row_norm=np.sqrt(np.sum(before*before))
                    want=before*min(1.,m.compiler.CAP/max(row_norm,np.finfo(float).tiny))
                    assert (w['i'],w['j'],w['si'],w['sj'])==(i,j,si,sj)
                    assert np.array_equal(w['params_before'],before) and np.array_equal(w['params_after'],want)
                    assert w['params_after']==b['params'][k] and np.linalg.norm(want)<=m.compiler.CAP+1e-14
                    assert np.allclose(right@want,[(si if a==i else sj if a==j else 0)*m.compiler.CAP/np.sqrt(2.) for a in range(10)])
                    k+=1
    assert k==222 and len(b['parameter_witnesses'])==222


def test_complete222_actual_shape_full_native_schedule_and_all_journals(complete):
    s,r,st=complete;w=m.validate_execution(s,r,lambda a:load(st,a));m.validate_events(r,st.events)
    assert len(r['forwards'])==1338 and len(r['module_ledger'])==4014 and len(r['AD_API'])==223
    assert len(r['result']['candidates'])==222 and r['plan']['commands']==w['commands']
    assert r['resources']=={k:dict(attempted=1338,completed=1338) for k in m.Modules.ORDER}
    assert m.base.math_resources([r])['SVD_completed_from_successful_frozen_helpers']==224
    assert r['plan']['schema']==m.NAME+'.model_plan' and r['plan']['arm']==m.ARM


def test_original42_compilations_unchanged_and_no_global_patching(complete):
    s,r,st=complete;other=dict(index=0);store=h.RAM();svd=np.linalg.svd;oldbank=m.base.all_mode_bank
    old=m.base.compile_arm(s['packet'],s['q'],h.ArrayBackend(other,store),lambda e:None)
    assert old['candidates']==r['result']['candidates'][:42]
    assert np.linalg.svd is svd and m.base.all_mode_bank is oldbank


@pytest.mark.parametrize('bad',['bank','witness','drop','raw','history','J','score','stage','T','winner','replay','ops','diagnostics'])
def test_saved_nonwinner_corruption_rejected(complete,bad):
    s,row,st=complete;r=deepcopy(row['result']);c=r['candidates'][190]
    if bad=='bank':r['bank']['params'][190][0]+=.01
    elif bad=='witness':r['bank']['parameter_witnesses'][190]['si']*= -1
    elif bad=='drop':r['candidates'].pop()
    elif bad=='raw':c['commands'][0][0]+=.01
    elif bad=='history':c['context']['action_history'][0][0]+=.01
    elif bad=='J':c['jacobian'][0][0]+=.01
    elif bad=='score':c['objective']+=.01
    elif bad=='stage':c['stage_cost']+=.01
    elif bad=='T':c['nonlinear_terminal_cost']+=.01
    elif bad=='winner':r['winner']=deepcopy(c)
    elif bad=='replay':r['whole_replay_context']['latent_history'][-1][0]+=.01
    elif bad=='ops':r['operations'].pop()
    else:r['diagnostics']['greedy_index']=999
    with pytest.raises(ValueError):m.validate_compilation(s['packet'],s['q'],r)


@pytest.mark.parametrize('kind',['module_attempt','module_inputs_saved','module_return','forward_attempt','forward_return','AD_attempt','AD_return',
    'saved_validation_attempt','saved_validation_return'])
def test_each_resource_missing_attempt_or_return_rejected(complete,kind):
    s,r,st=complete;events=list(st.events);events.pop(next(i for i,e in enumerate(events) if e['kind']==kind))
    with pytest.raises(ValueError):m.validate_events(r,events)


@pytest.mark.parametrize('at',[0,1,4,258,1337])
def test_failure_retains_partial_arrays_and_fixed_denominator(at):
    s=h.slot();st=h.RAM();r=dict(index=0,events=[]);mods=m.Modules(NS(),st,'root_0',NS(stage=None,expected_input=None,last_native=None))
    b=h.ArrayBackend(r,st,mods,fail=at)
    with pytest.raises(RuntimeError,match='RAM forward fault'):
        m.compile_arm(s['packet'],s['q'],b,lambda e:r['events'].append(deepcopy(e)))
    assert len(r['forwards'])==at+1 and r['forwards'][-1]['completed'] is False
    assert mods.counts()['predict']==dict(attempted=at+1,completed=at)
    assert mods.entries[-1]['resource']=='predict' and len(mods.entries[-1]['inputs'])==2
    census=m.census(dict(slots=[h.slot(i) for i in range(4)]))
    assert len(census)==4 and all(e['candidate_denominator']==222 and e['status']=='not_started' for e in census)
    assert st.files


def test_flat_null_modes_duplicate_actions_no_pruning():
    s=h.slot();r=dict(index=0);st=h.RAM()
    out=m.compile_arm(s['packet'],s['q'],h.ArrayBackend(r,st,flat=True),lambda e:None)
    assert len(out['candidates'])==222 and len(r['forwards'])==1338 and out['winner']['index']==0


def test_input_and_authorization_fail_before_output(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:(_ for _ in ()).throw(ValueError('source drift')))
    monkeypatch.setattr(m,'Store',lambda *a:pytest.fail('output before source check'))
    with pytest.raises(ValueError,match='MAIN'):m.run('x','a'*64,'x','wrong')
    with pytest.raises(ValueError,match='source drift'):m.run('x','a'*64,'x',m.AUTH)


def test_actual_backend_predispatch_cap_without_tensor_or_neural():
    b=object.__new__(m.Backend);b.torch=NS();b.check=lambda:None;b.row=dict(forwards=[None]*1338)
    with pytest.raises(ValueError,match='1338 predict cap'):b.forward({},None,'step')


def test_module_predispatch_cap_and_no_further_native_call():
    st=h.RAM();mods=m.Modules(NS(),st,'root_0',NS(stage=1338))
    mods.entries=[None]*(3*1338)
    with pytest.raises(ValueError,match='1338 native forwards'):mods.pre('action_encoder')('unused',())
    assert st.events==[]


def test_failed_journal_return_does_not_mask_first_backend_error():
    s=h.slot();row=dict(index=0);st=h.RAM();b=h.ArrayBackend(row,st,fail=1)
    def journal(e):
        if 'error' in e:raise OSError('completion journal failed')
    with pytest.raises(RuntimeError,match='RAM forward fault') as out:m.compile_arm(s['packet'],s['q'],b,journal)
    assert any(isinstance(e,OSError) for e in out.value.compiler_secondary_errors)


def test_final_region_writer_uses_reserved_bytes_tiny_tmp(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'CAP',8192);monkeypatch.setattr(m,'FINAL',4096);monkeypatch.setattr(m,'FLOOR',0)
    st=m.Store(tmp_path/(m.NAME+'_tiny'))
    st._write('payload',b'x'*4000)
    with pytest.raises(ValueError,match='reserve'):st._write('blocked',b'x'*100)
    st._write('failure.json',b'{"commands_released":false}',final=True)
    assert set(st.file_sizes)=={'events.jsonl.gz','payload','failure.json'}
    assert st.used_bytes==st.audit_files()==4000+len(b'{"commands_released":false}')


def test_O1_regular_writes_append_exact_bytes_no_per_write_iterdir(tmp_path,monkeypatch):
    st=m.Store(tmp_path/(m.NAME+'_tiny'));original=Path.iterdir
    monkeypatch.setattr(Path,'iterdir',lambda *a:pytest.fail('per-write directory census'))
    st._write('part',b'abc');st._write('part',b'defg',append=True)
    assert st.used_bytes==7 and st.file_sizes['part']==7
    with pytest.raises(ValueError,match='exclusive'):st._write('part',b'overwrite')
    monkeypatch.setattr(Path,'iterdir',original)
    assert st.audit_files()==7 and (st.output/'part').read_bytes()==b'abcdefg'


@pytest.mark.parametrize('mode',['short','raises_after_write','fsync_after_write'])
def test_actual_partial_write_or_error_bytes_are_accounted(tmp_path,monkeypatch,mode):
    st=m.Store(tmp_path/(m.NAME+'_tiny'));write=m.os.write;fsync=m.os.fsync
    def partial(fd,raw):
        n=write(fd,raw[:3])
        if mode=='raises_after_write':raise OSError('write returned error after partial')
        return n
    if mode=='fsync_after_write':monkeypatch.setattr(m.os,'fsync',lambda *a:(_ for _ in ()).throw(OSError('fsync failed')))
    else:monkeypatch.setattr(m.os,'write',partial)
    with pytest.raises((ValueError,OSError)):st._write('partial',b'x'*21)
    monkeypatch.setattr(m.os,'write',write);monkeypatch.setattr(m.os,'fsync',fsync)
    want=21 if mode=='fsync_after_write' else 3
    assert (st.output/'partial').stat().st_size==st.file_sizes['partial']==st.used_bytes==want
    assert st.audit_files()==want


def test_append_partial_growth_count_not_requested_length(tmp_path,monkeypatch):
    st=m.Store(tmp_path/(m.NAME+'_tiny'));st._write('part',b'abcde');write=m.os.write
    monkeypatch.setattr(m.os,'write',lambda fd,raw:write(fd,raw[:2]))
    with pytest.raises(ValueError,match='short write'):st._write('part',b'z'*20,append=True)
    monkeypatch.setattr(m.os,'write',write)
    assert st.audit_files()==st.used_bytes==7 and (st.output/'part').read_bytes()==b'abcdezz'


@pytest.mark.parametrize('drift',['unknown','resize','replace','missing'])
def test_terminal_census_rejects_unknown_or_changed_file(tmp_path,drift):
    st=m.Store(tmp_path/(m.NAME+'_tiny'));st._write('owned',b'abc')
    # Fixture mutations through apply-independent tiny test files, never results.
    if drift=='unknown':
        with (st.output/'extra').open('xb') as f:f.write(b'x')
    elif drift=='resize':
        with (st.output/'owned').open('ab') as f:f.write(b'x')
    elif drift=='replace':
        (st.output/'owned').rename(tmp_path/'old_inode_fixture')
        with (st.output/'owned').open('xb') as f:f.write(b'abc')
    else:(st.output/'owned').rename(tmp_path/'moved_fixture')
    with pytest.raises(ValueError):st.audit_files()
    with pytest.raises(ValueError):st._write('report.json',b'{}',final=True)


def test_observed_append_drift_is_persistent_not_silently_recertified(tmp_path):
    st=m.Store(tmp_path/(m.NAME+'_tiny'));st._write('owned',b'abc')
    with (st.output/'owned').open('ab') as f:f.write(b'external')
    with pytest.raises(ValueError,match='drift'):st._write('owned',b'ours',append=True)
    assert st.drifted
    with pytest.raises(ValueError,match='drift'):st.audit_files()
    with pytest.raises(ValueError,match='drift'):st._write('report.json',b'{}',final=True)


def test_cumulative_uncompressed_journal_cap_before_compression(tmp_path,monkeypatch):
    st=m.Store(tmp_path/(m.NAME+'_tiny'));e=dict(kind='repeat',value='0'*4096)
    size=len(m.canonical(e));monkeypatch.setattr(m,'JOURNAL_CAP',size)
    st.event(e);assert st.journal_plain_bytes==size
    compressed=(st.output/'events.jsonl.gz').stat().st_size
    assert compressed<size
    with pytest.raises(ValueError,match='uncompressed'):st.event(e)
    assert st.journal_plain_bytes==size and st.broken
    assert (st.output/'events.jsonl.gz').stat().st_size==compressed


@pytest.mark.parametrize('final_report_fails',[False,True])
def test_run_failure_census_first_error_and_bounded_terminal_RAM(monkeypatch,final_report_fails):
    # Pure interface objects. No torch import, model/tensor operation or disk I/O.
    import sys
    from contextlib import nullcontext
    from types import ModuleType
    p=dict(slots=[h.slot(i) for i in range(4)],policy='RAM',expected_model_hash='a'*64)
    calls=[]
    def verify(*args):
        calls.append('verify')
        if len(calls)>1:raise FileNotFoundError('final source missing')
        return p
    class Memory:
        def __init__(self,*a):self.output=Path('/RAM/out');self.files={};self.events=[]
        def event(self,e):self.events.append(deepcopy(e))
        def put(self,n,v,**kw):
            if n=='report.json' and final_report_fails:raise OSError('final report write failed')
            self.files[n]=deepcopy(v)
        def _write(self,n,raw,**kw):self.files[n]=json.loads(raw)
        def close(self):raise OSError('journal close failed')
        def audit_files(self):return 0
    st=Memory()
    class FakeModel:
        def eval(self):return self
        def requires_grad_(self,flag):assert flag is False;return self
    class FakeModules:
        ORDER=m.Modules.ORDER
        def __init__(self,*a):self.entries=[dict(resource='predict',completed=False)]
        def __enter__(self):return self
        def __exit__(self,*a):return False
        def counts(self):return {k:dict(attempted=int(k=='predict'),completed=0) for k in self.ORDER}
    def fail_compile(*a):raise RuntimeError('first compute error')
    torch=ModuleType('torch');torch.device=lambda *a:NS(type='cuda')
    torch.backends=NS(cuda=NS(matmul=NS(allow_tf32=None)),cudnn=NS(allow_tf32=None))
    torch.autocast=lambda **kw:nullcontext()
    att=ModuleType('torch.nn.attention');att.sdpa_kernel=lambda *a:nullcontext();att.SDPBackend=NS(MATH='MATH')
    ck=ModuleType('experiments.checkpoint_compat');ck.load_compatible_pretrained=lambda *a,**k:FakeModel()
    gpu=ModuleType('experiments.gpu_preflight');gpu.require_cuda_allocation=lambda *a:dict(interface_mock=True)
    for name,value in [('torch',torch),('torch.nn.attention',att),('experiments.checkpoint_compat',ck),('experiments.gpu_preflight',gpu)]:
        monkeypatch.setitem(sys.modules,name,value)
    monkeypatch.setattr(m,'verify',verify);monkeypatch.setattr(m,'Store',lambda *a:st)
    monkeypatch.setattr(m,'Modules',FakeModules);monkeypatch.setattr(m,'Backend',lambda *a:NS())
    monkeypatch.setattr(m,'compile_arm',fail_compile)
    monkeypatch.setattr(m.overlay.moving.low,'frozen',lambda *a:None)
    count=[]
    def tensor_hash(*a):
        count.append(1)
        if len(count)>1:raise ValueError('final hash failed')
        return 'a'*64
    monkeypatch.setattr(m.overlay.moving.low,'tensor_hash',tensor_hash)
    with pytest.raises(ValueError,match='first compute error'):m.run('/RAM/p','b'*64,'/RAM/out',m.AUTH)
    report=st.files['failure.json' if final_report_fails else 'report.json']
    assert report['first_error']['message']=='first compute error' and report['commands_released'] is False
    assert [r['status'] for r in report['census' if final_report_fails else 'rows']]==['failed_partial']+['not_started']*3
    assert len(report['errors'])>=4 and len(calls)==2
    if final_report_fails:assert report['candidate_denominator']==888
    else:assert report['resources']['predict']==dict(attempted=1,completed=0)


def test_real_saved_build_canonical_no_prepare():
    p=m.build();raw=m.canonical(p)
    assert json.loads(raw)==p and len(raw)<m.PREP_CAP
    assert [(s['root_slot'],s['condition']) for s in p['slots']]==m.ORDER
    assert all(type(s['q']) is list and s['q']==np.eye(192).tolist() for s in p['slots'])
    for f in [Path(m.__file__).resolve(),m.PREREG,m.ROOT/'tests'/('test_'+m.NAME+'.py'),m.ROOT/'innerdocs'/(m.NAME+'.md')]:
        assert p['bindings'][str(f)]==m.sha(f)
    print('actual saved build canonical bytes',len(raw))


def test_cli_help_build_prepare_verify_RAM(monkeypatch,capsys):
    with pytest.raises(SystemExit) as e:m.main(['--help'])
    assert e.value.code==0 and 'consume' in capsys.readouterr().out
    p=dict(schema=m.NAME+'.prepare',slots=[h.slot(i) for i in range(4)])
    monkeypatch.setattr(m,'build',lambda:deepcopy(p))
    monkeypatch.setattr(m.capture,'write_new',lambda path,raw,cap:dict(sha256='a'*64,bytes=len(raw)))
    assert m.main(['prepare','--output','/RAM/'+m.NAME+'_prepare.json'])==0
    assert json.loads(capsys.readouterr().out)['sha256']=='a'*64
    monkeypatch.setattr(m,'sha',lambda path:'a'*64);monkeypatch.setattr(m.overlay,'read',lambda *a:deepcopy(p))
    assert m.main(['verify','--prepared','/RAM/p','--prepare-sha','a'*64])==0
    assert json.loads(capsys.readouterr().out)['slots']==4


@pytest.mark.parametrize('name,raw',[('wrong.json',b'{}'),(m.NAME+'_x.gz',b'{}'),(m.NAME+'_x.json','{}'),(m.NAME+'_x.json',b'x'*(m.PREP_CAP+1))])
def test_own_exclusive_prepare_prefix_type_cap(monkeypatch,name,raw):
    monkeypatch.setattr(m.capture,'write_new',lambda *a:pytest.fail('invalid payload reached writer'))
    with pytest.raises(ValueError):m.write_new('/RAM/'+name,raw)


def test_dense_saved_real_shape_capacity_projection():
    # Dense ACTUAL saved row, not diagonal-J/zero-latent fixture. No output files.
    path=m.ROOT/'results/agent_allmode_prefix_20261028_v2_cuda_MAIN01/root_0.json.gz'
    r=json.loads(gzip.decompress(path.read_bytes()))
    def bound(v):
        if v is None:return 4
        if type(v) is bool:return 5
        if type(v) in (float,int):return 32
        if isinstance(v,str):return len(json.dumps(v,ensure_ascii=True).encode())
        if isinstance(v,list):return 2+len(v)+sum(map(bound,v))
        return 2+2*len(v)+sum(bound(k)+bound(x) for k,x in v.items())
    # Six times the entire42 dense row (including all repeated numerical fields)
    # exceeds222/42; include32MiB for new bank receipts/expanded paths/metadata.
    report_upper=4*6*bound(r)+32*1024**2
    assert report_upper<m.FINAL
    old=path.parent
    # Measured full dense saved journal+NPY+gz rows, projected by a factor6
    # (larger than222/42); this is a stress projection, NOT a compression theorem.
    regular_projection=6*sum(p.stat().st_size for p in old.iterdir() if p.name!='report.json')+32*1024**2
    assert regular_projection<m.CAP-m.FINAL
    assert regular_projection+report_upper<m.CAP
    # Entire REAL dense event census including intermediate candidate snapshots,
    # module inputs/returns and J arrays. Stream/decode, no materialized new run.
    with gzip.open(old/'events.jsonl.gz','rb') as stream:
        journal_old_upper=sum(bound(json.loads(line)) for line in stream)
    journal_upper=6*journal_old_upper+64*1024**2
    assert journal_upper<m.JOURNAL_CAP
    print('dense report conservative numeric32 bound',report_upper,'ordinary dense saved projection',regular_projection,
        'uncompressed dense journal upper',journal_upper)


def test_RAM_full4_consume_with_all_arrays_and_actual_cli(monkeypatch,capsys):
    files={};rows=[];slots=[];events=[]
    for i in range(4):
        s,r,st=execute(i);slots.append(s);rows.append(r)
        events.append(dict(kind='root_started',slot=deepcopy(s)));events.extend(st.events)
        events.append(dict(kind='root_sealed',index=i,selection_digest=r['plan']['selection_digest']))
        files.update(st.files);files[f'root_{i}.json.gz']=gzip.compress(m.canonical(r),mtime=0)
    p=dict(slots=slots,expected_model_hash='a'*64,contract=m.contract())
    files['binding.json']=m.canonical(dict(prepared='/RAM/p',prepare_sha256='b'*64))
    journal_plain=b''.join(m.canonical(e) for e in events)
    files['events.jsonl.gz']=gzip.compress(journal_plain,mtime=0)
    report=dict(schema=m.NAME+'.report',status='all4_mixedmode_plans_sealed_not_admitted',errors=[],first_error=None,
        commands_released=True,model_unchanged=True,source_verified=True,contract=m.contract(),
        model_hash_before='a'*64,model_hash_after='a'*64,prepared='/RAM/p',prepare_sha256='b'*64,
        rows=rows,artifact_sha256={k:hashlib.sha256(v).hexdigest() for k,v in files.items()},
        journal_plain_bytes=len(journal_plain),
        artifact_bytes_before_report=sum(map(len,files.values())),
        math_resources=m.base.math_resources(rows),resources={k:dict(attempted=5352,completed=5352) for k in m.Modules.ORDER})
    files['report.json']=m.canonical(report)
    assert len(files['report.json'])<m.FINAL and sum(map(len,files.values()))<m.CAP
    print('RAM full4 files bytes',sum(map(len,files.values())),'report',len(files['report.json']))
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
    monkeypatch.setattr(m,'Path',VPath);monkeypatch.setattr(m,'sha',lambda path:hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(m.overlay,'read',read);monkeypatch.setattr(m,'verify',lambda *a:p)
    expected=hashlib.sha256(files['report.json']).hexdigest()
    assert m.main(['consume','--output','/RAM/out','--report-sha',expected])==0
    result=json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result['status']=='all4_mixedmode_commands_verified_not_admitted' and len(result['plans'])==4
    assert [(v['root_slot'],v['condition']) for v in result['plans']]==m.ORDER
    assert all(v['whole_run_report_sha256']==expected for v in result['plans'])
