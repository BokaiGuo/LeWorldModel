"""RAM-only fake env/PNG/byte ledger. No simulator, neural, prepare or result."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
import stat
import numpy as np
from PIL import Image
import pytest
from experiments.spectral_reachability import agent_mixedmode_capability_physical_20261028_v1 as m
from test_agent_allmode_physical_20261028_v1 import fixture as old_fixture, fake_place, Env


def h(value):return hashlib.sha256(str(value).encode()).hexdigest()


class MemoryStore(m.Store):
    def __init__(self):
        self.path=Path('/RAM');self.blobs={};self.used=0;self.previous=None;self.index=0
        self.terminal=False;self.free=m.CAP+m.FLOOR;self.events=[]
    def check(self,size=4096,final=False):
        m.require(self.used+size<=m.CAP-(0 if final or self.terminal else m.FINAL) and
            self.free>=m.FLOOR+size,'RAM cap/reserve/floor')
    def write(self,name,raw,append=False,final=False):
        self.check(len(raw),final)
        assert append or name not in self.blobs
        if name=='journal.jsonl':m.require(len(self.blobs.get(name,b''))+len(raw)<=m.JOURNAL_CAP,'RAM journal64MiB')
        self.blobs.setdefault(name,bytearray()).extend(raw);self.used+=len(raw)
    def emit(self,kind,**kw):
        e=dict(index=self.index,previous_sha256=self.previous,kind=kind,**deepcopy(kw))
        e['event_sha256']=m.probe.digest(e);self.write('journal.jsonl',m.canonical(e)+b'\n',append=True)
        self.index+=1;self.previous=e['event_sha256'];self.events.append(e)
    def frame(self,name,a):
        buf=io.BytesIO();Image.fromarray(a).save(buf,format='PNG');self.write(name,buf.getvalue())
        return hashlib.sha256(self.blobs[name]).hexdigest()
    def arrays(self,name,**arrays):
        buf=io.BytesIO();np.savez_compressed(buf,**arrays);self.write(name,buf.getvalue())
        return hashlib.sha256(self.blobs[name]).hexdigest()
    def census(self):return {k:hashlib.sha256(v).hexdigest() for k,v in self.blobs.items()}


def fixture():
    old,data,r,compiled,consumed=old_fixture()
    compiled['schema']='agent_mixedmode_prefix_20261028_v1.prepare'
    consumed['status']='all4_mixedmode_commands_verified_not_admitted'
    for slot in compiled['slots']:slot['packet']={'complete_history':'SYNTHETIC'}
    p=m.probe.assemble(compiled,consumed,m.adapter.root_metadata(data))
    p.update(source_slots=old['source_slots'],warm_traces=old['warm_traces'])
    package=dict(schema=m.NAME+'.prepare',contract=m.contract(),probe_digest=m.probe.digest(p),
        probe_prepared='/RAM/probe_prepare.json',probe_prepare_sha256=h('probe prepare'),
        bindings={str(Path(m.__file__).resolve()):h('executor source')})
    return package,p,data


def execute(monkeypatch,*,factory=Env,package=None,p=None,data=None,store=None,started=None):
    fp,pp,dd=fixture();package=fp if package is None else package;p=pp if p is None else p;data=dd if data is None else data
    store=MemoryStore() if store is None else store;envs=[]
    def make():env=factory();envs.append(env);return env
    monkeypatch.setattr(m,'place',fake_place);monkeypatch.setattr(m,'validate_env',lambda body:None)
    monkeypatch.setattr(m.probe,'verify_bindings',lambda _:None)
    monkeypatch.setattr(m,'sha',lambda path:hashlib.sha256(store.blobs[Path(path).name]).hexdigest())
    result=m.execute(package,p,data,store,make,authorization=m.AUTH,prepared='/RAM/own_prepare.json',prepare_sha=h('own prepare'),started=started)
    return result,store,envs,p,package


@lru_cache(maxsize=1)
def completed():
    with pytest.MonkeyPatch.context() as mp:return execute(mp)


def test_all40_true_pre_post_counts_and_full_denominator():
    r,s,envs,p,_=completed()
    assert r['status']=='complete',r['errors'][:3]
    assert len(envs)==len(r['rows'])==40
    assert r['counts']=={('internal_physics_step' if k=='physics_step' else k):dict(attempted=v,completed=v) for k,v in m.CAPS.items()}
    assert len([n for n in s.blobs if n.endswith('.png')])==2340
    assert len(s.blobs['journal.jsonl'])<m.JOURNAL_CAP
    assert len(m.canonical(r))<m.REPORT_CAP
    assert r['P1_admission'] is False and r['time_confound'] is True
    for j,row,env in zip(p['jobs'],r['rows'],envs,strict=True):
        assert row['status']=='completed' and row['start_verified'] and len(row['frames'])==6
        assert row['attempted']==row['completed']==5 and env.closed
        commands=np.asarray(env.commands,np.float32)
        expected=np.concatenate([m.moving.warmup_commands(),m.probe.raw(j['prefix_commands'],len(j['prefix_commands'])),m.probe.raw(j['probe_commands'],5)])
        assert commands.tobytes()==expected.tobytes()
        assert len(commands)==(15 if j['role']=='before' else 20)
        assert env.with_target is False and all(flag is False for flag in env.dispatch_flags)
        assert '_set_state' not in env.__dict__ and 'reset' not in env.__dict__


def test_emitted_hashchain_and_all_receipts():
    r,s,*_=completed();tail=None
    for i,e in enumerate(s.events):
        assert e['index']==i and e['previous_sha256']==tail
        assert e['event_sha256']==m.probe.digest({k:v for k,v in e.items() if k!='event_sha256'})
        tail=e['event_sha256']
    assert tail==r['journal_tail'] and r['journal']['sha256']==hashlib.sha256(s.blobs['journal.jsonl']).hexdigest()
    assert len([e for e in s.events if e['kind']=='probe_returned'])==200
    assert len([e for e in s.events if e['kind']=='causal_replay_returned'])==500
    assert len([e for e in s.events if e['kind']=='fresh_observation'])==2340


def test_actual_probe_consume_ABI_RAM(monkeypatch):
    r,s,envs,p,package=completed()
    original_open=Image.open
    monkeypatch.setattr(m.probe,'verify',lambda *_:p)
    monkeypatch.setattr(m.probe,'verify_bindings',lambda _:None)
    monkeypatch.setattr(m.probe,'read',lambda _:r)
    monkeypatch.setattr(m.probe,'sha',lambda path:h('executor source') if Path(path).name==Path(m.__file__).name else
        h('report') if Path(path).name=='report.json' else hashlib.sha256(s.blobs[Path(path).name]).hexdigest())
    monkeypatch.setattr(Path,'is_symlink',lambda _:False)
    monkeypatch.setattr(Path,'stat',lambda _:SimpleNamespace(st_size=1))
    monkeypatch.setattr(Path,'read_bytes',lambda path:bytes(s.blobs[path.name]))
    monkeypatch.setattr(Image,'open',lambda path:original_open(io.BytesIO(s.blobs[Path(path).name])))
    result=m.probe.consume('/RAM/probe',package['probe_prepare_sha256'],'/RAM/report.json',h('report'),m.__file__,h('executor source'))
    assert len(result['contrasts'])==20 and result['fixed_denominator']==40


def test_own_trace_complete_and_partial_census():
    r,s,_,p,_=completed();m.validate_trace(p,r,s.events)
    bad=deepcopy(r);bad['counts']['public_step']['attempted']-=1
    with pytest.raises(ValueError):m.validate_trace(p,bad,s.events)
    broken=deepcopy(r);broken['rows'][0]['context_digest']='wrong'
    with pytest.raises(ValueError):m.validate_trace(p,broken,s.events)


def test_own_trace_all_unstarted_is_honest_partial(monkeypatch):
    r,s,_,p,_=execute(monkeypatch,started=time.monotonic()-m.WALL-1)
    m.validate_trace(p,r,s.events)
    assert r['counts']=={} and r['status']=='partial'


def test_reject_authority_before_reads_factory_or_store(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('must not call')
    monkeypatch.setattr(m,'verify',forbidden);monkeypatch.setattr(m,'Store',forbidden)
    with pytest.raises(ValueError,match='authority'):m.run('x','x','x',forbidden,authorization='NO')


def test_limits_nonphysical_close_only_and_attempt_before_callback():
    s=MemoryStore();meter=m.Meter(s);meter.where={'job_digest':'a'*64};order=[]
    meter.call('constructor',lambda:order.append(s.events[-1]['kind']))
    assert order==['operation_attempted']
    meter.counts['public_step']['attempted']=m.CAPS['public_step']
    with pytest.raises(ValueError):meter.call('public_step',lambda:order.append('bad'))
    with pytest.raises(ValueError):meter.call('reset',lambda:None,cleanup=True)
    meter.fault=True;meter.call('close',lambda:order.append('closed'),cleanup=True)
    assert order[-1]=='closed'


def test_callback_exception_keeps_attempt_and_cleanup():
    s=MemoryStore();meter=m.Meter(s);meter.where={'job_digest':'a'*64}
    def bad():raise RuntimeError('primary')
    with pytest.raises(RuntimeError,match='primary'):meter.call('public_step',bad)
    assert meter.counts['public_step']==dict(attempted=1,completed=0)
    assert s.events[-1]['kind']=='operation_failed'
    meter.call('close',lambda:None,cleanup=True)


def test_stale_wall_no_callback_but_close():
    s=MemoryStore();meter=m.Meter(s,time.monotonic()-m.WALL-1);meter.where={'job_digest':'a'*64}
    with pytest.raises(ValueError):meter.call('constructor',lambda:None)
    meter.call('close',lambda:None,cleanup=True)


def test_full_denominator_preserved_before_first_constructor(monkeypatch):
    r,s,envs,p,_=execute(monkeypatch,started=time.monotonic()-m.WALL-1)
    assert r['status']=='partial' and len(r['rows'])==40 and not envs
    assert all(x['status']=='unstarted' and x['end_state'] is None for x in r['rows'])


def test_source_final_failure_does_not_mask_execution_primary(monkeypatch):
    package,p,data=fixture();s=MemoryStore()
    def badfactory():raise RuntimeError('FIRST_EXECUTION_ERROR')
    monkeypatch.setattr(m.probe,'verify_bindings',lambda _:(_ for _ in ()).throw(FileNotFoundError('LATER_SOURCE')))
    # Do not use execute helper which replaces verify_bindings.
    r=m.execute(package,p,data,s,badfactory,authorization=m.AUTH,prepared='/RAM/own',prepare_sha=h('p'))
    assert r['errors'][0]['message']=='FIRST_EXECUTION_ERROR'
    assert any(e['phase']=='final_source_hash' for e in r['errors'])
    assert len(r['rows'])==40 and r['status']=='partial'


def test_lossless_info_arrays_not_expanded_into_json():
    s=MemoryStore();a=np.arange(10,dtype=np.float64);image=np.zeros((224,224,3),np.uint8)
    r=m.save_info(s,'info.npz',{'goal':image,'state':a,'flag':True})
    assert len(m.canonical(r))<1024
    with np.load(io.BytesIO(s.blobs['info.npz']),allow_pickle=False) as z:
        assert z['a0'].dtype==np.uint8 and z['a1'].dtype==np.float64
        assert z['a1'].tobytes()==a.tobytes()


def test_no_pickle_info():
    with pytest.raises(ValueError):m.save_info(MemoryStore(),'info.npz',{'x':np.asarray([object()],dtype=object)})


def test_cli_help_and_refusal(capsys):
    with pytest.raises(SystemExit) as e:m.main(['--help'])
    assert e.value.code==0 and 'consume' in capsys.readouterr().out
    with pytest.raises(ValueError):m.main(['run','--prepared','x','--prepare-sha','x','--output','x'])


class FakeIO:
    """File descriptor bytes/fstat in RAM; exercise actual Store.write logic."""
    def __init__(self):self.data={};self.current=None;self.partial=False;self.inode=7
    def open(self,path,flags,mode):
        name=Path(path).name
        if flags & m.os.O_EXCL:
            if name in self.data:raise FileExistsError(name)
            self.data[name]=bytearray()
        self.current=name;return 91
    def fstat(self,fd):
        return SimpleNamespace(st_mode=stat.S_IFREG,st_dev=1,st_ino=self.inode,st_size=len(self.data[self.current]))
    def fdopen(self,*a,**kw):return self
    def __enter__(self):return self
    def __exit__(self,*a):return False
    def write(self,value):
        if self.partial:
            self.data[self.current].extend(value[:3]);raise OSError('partial write retained')
        self.data[self.current].extend(value);return len(value)
    def flush(self):return None


def file_store(monkeypatch):
    s=object.__new__(m.Store);s.path=Path('/RAM');s.used=0;s.files={};s.terminal=False;s.drift=False
    fs=FakeIO()
    monkeypatch.setattr(s,'check',lambda *a,**kw:None)
    monkeypatch.setattr(m.os,'open',fs.open);monkeypatch.setattr(m.os,'fstat',fs.fstat)
    monkeypatch.setattr(m.os,'fdopen',fs.fdopen);monkeypatch.setattr(m.os,'close',lambda fd:None)
    monkeypatch.setattr(m.os,'fsync',lambda fd:None);monkeypatch.setattr(m.adapter.paired,'fsync_dir',lambda _:None)
    return s,fs


def test_owned_append_byte_account_is_actual(monkeypatch):
    s,fs=file_store(monkeypatch)
    s.write('journal.jsonl',b'first\n',append=True);s.write('journal.jsonl',b'second\n',append=True)
    assert s.used==13 and s.files['journal.jsonl'][2]==13
    with pytest.raises(ValueError):s.write('journal.jsonl',b'overwrite')


def test_partial_bytes_and_exception_stay_charged(monkeypatch):
    s,fs=file_store(monkeypatch);fs.partial=True
    with pytest.raises(OSError,match='partial'):s.write('partial.bin',b'abcdef')
    assert s.used==3 and s.files['partial.bin'][2]==3 and bytes(fs.data['partial.bin'])==b'abc'


def test_foreign_inode_drift_not_recertifiable(monkeypatch):
    s,fs=file_store(monkeypatch);s.write('journal.jsonl',b'first\n',append=True);fs.inode=99
    with pytest.raises(ValueError,match='inode'):s.write('journal.jsonl',b'bad',append=True)
    assert s.drift
    with pytest.raises(ValueError,match='drift'):s.census()


def test_unknown_file_final_census_rejected(monkeypatch):
    s=object.__new__(m.Store);s.path=Path('/RAM');s.used=0;s.files={};s.drift=False
    foreign=SimpleNamespace(name='foreign',lstat=lambda:SimpleNamespace(st_mode=stat.S_IFREG,st_dev=1,st_ino=7,st_size=1))
    monkeypatch.setattr(Path,'iterdir',lambda _:[foreign])
    with pytest.raises(ValueError,match='census'):s.census()


def test_store_rejects_symlink_or_nested_output_name(monkeypatch):
    s,fs=file_store(monkeypatch)
    with pytest.raises(ValueError):s.write('../outside',b'bad')
    with pytest.raises(ValueError):s.write('dir/file',b'bad')


def test_terminal_journal_failure_preserves_primary(monkeypatch):
    class BrokenStore(MemoryStore):
        def emit(self,kind,**kw):
            if kind=='run_finished':raise OSError('terminal journal error')
            return super().emit(kind,**kw)
    def failed():raise RuntimeError('branch primary')
    r,s,_,_,_=execute(monkeypatch,factory=failed,store=BrokenStore())
    assert r['status']=='partial' and r['errors'][0]['message']=='branch primary'
    assert any(e['phase']=='terminal_journal' for e in r['errors'])
    assert 'report.json' in s.blobs


def test_capacity_and_wall_are_prespecified_not_inherited4():
    c=m.contract()
    assert c['caps']['reset']==80 and c['caps']['physics_step']==7160
    assert c['frames']==2340 and c['wall_seconds']==1800
    assert c['output_cap_bytes']==512*1024**2 and c['journal_uncompressed_cap_bytes']==64*1024**2
    assert c['probe_schema'].endswith('_v2.prepare') and not c['P1_admission']


@pytest.mark.parametrize('fault',['terminal','truncate','throw','substitute'])
def test_native_stop_or_failed_step_retains40_no_padding(monkeypatch,fault):
    class FaultEnv(Env):
        def step(self,c):
            result=super().step(c)
            if fault=='substitute' and len(self.commands)==11:
                # Zero probes/prefixes must also have a genuinely unequal fake
                # actual action; the inherited fake's zero substitution is a no-op there.
                self.latest_action=np.asarray(c,np.float32)+np.float32(.125)
            return result
    r,s,envs,p,_=execute(monkeypatch,factory=lambda:FaultEnv(fault=fault,at=11))
    assert r['status']=='partial' and len(r['rows'])==40 and len(envs)==40
    assert all(len(e.commands)==11 and e.closed for e in envs)
    assert all(row['end_state'] is None for row in r['rows'])
    if fault in ('terminal','truncate'):
        assert all(row['status']=='native_stopped' for row in r['rows'])
    else:
        assert all(row['status']=='failed' for row in r['rows'])
    assert all(row['last_observation']['png'] in s.blobs for row in r['rows'])


def test_existing_rgb_guard_receipts_not_state_substitution():
    r,s,*_=completed()
    for row in r['rows']:
        assert row['guards']['warmup']['primitive_commands_authorized']==10
        n=5 if row['role']=='before' else 10
        assert row['guards']['task']['primitive_commands_authorized']==n
        assert len(row['guards']['task']['observations'])==n+1
    assert len([e for e in s.events if e['kind']=='rgb_guard_observed'])==780


@pytest.mark.skipif(os.environ.get('RUN_MIXEDMODE_PHYSICAL_SAVED_MATERIAL')!='1',reason='explicit saved-only material opt-in')
def test_actual_saved_material_size_no_prepare():
    path=m.ROOT/'results/agent_mixedmode_capability_probe_20261028_v2_prepare_MAIN01.json'
    package,p,data=m.material(path,'e8e4554ea2448a8c4bcf4e916934800fa8c7afe3eda335174056a551620da3d5')
    size=len(m.canonical(package))
    print('ACTUAL_PHYSICAL_MATERIAL_BYTES',size,'BINDINGS',len(package['bindings']))
    assert size<=m.PREP_CAP and len(package['bindings'])>44000 and len(p['jobs'])==40
    assert len(data['roots'])==2
