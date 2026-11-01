"""Pure NumPy/mock compression, B8 budget and real-source provenance tests."""
from contextlib import contextmanager
from copy import deepcopy
import gzip
import hashlib
from pathlib import Path
from types import SimpleNamespace
import zlib
import numpy as np
import pytest

from experiments.spectral_reachability import p1_observed_direct_cuda_20261027_v2 as m

PACK=m.ROOT/'results/p1_observed_history_source_20261027_v1_main01'
PREP=m.ROOT/'results/p1_observed_history_source_20261027_v1_prepare01.json'
PACK_SHA='0b89d77494acc6cf03bb0110e87231c1c240a253e844326acc668b7e4f93d546'
PREP_SHA='ecc5f4a488290b6d0eade4d3ff2034d256729de94c4d3a14b12d6eebad783385'


class DenseBackend:
    """Dense float64 contexts/J stress the full native event shapes, not a NN."""
    def __init__(self):self.b=np.random.default_rng(498).normal(size=(192,10))*.02
    @contextmanager
    def guard(self,meter):yield
    def rollout(self,t,raw,meter,mode):
        c=deepcopy(t['root_context']);out=[]
        for k,token in enumerate(m.screen.core.normalized_tokens(t,raw)):
            meter.call('action_encoder',lambda:None)
            z=meter.call('predict',lambda:.7*np.asarray(c['latent_history'][-1])+self.b@token)
            c=dict(c,latent_history=c['latent_history'][1:]+[z.tolist()],action_history=c['action_history'][1:]+[token.tolist()])
            out.append(deepcopy(c))
            if mode=='AD':meter.emit('jacobian_forward_packet',packet=k,context=c)
            else:meter.emit('native_packet',mode=mode,packet=k,context=c)
        return out
    def jacobian(self,t,inc,meter):
        terminal=self.rollout(t,inc['commands'],meter,'AD')[-1]['latent_history'][-1]
        meter.emit('jacobian_forward_complete',terminal=terminal,logits=inc['logits'])
        std=np.tile(np.asarray(t['action_std'],np.float32),5).astype(float)
        raw=np.concatenate((.7*self.b/std,self.b/std),axis=1)
        for i in range(192):
            row=meter.call('vjp_directions',lambda:raw[i])
            meter.emit('jacobian_row',row=i,raw_control_derivative=row.tolist())
        j=raw*(1-np.asarray(inc['commands']).reshape(20)**2)
        meter.emit('joint_logit_jacobian',terminal=terminal,jacobian=j.tolist())
        return terminal,j


def test_full_four_B8_dense_lossless_mock_scale(tmp_path):
    data=m.original.material(PACK,PACK_SHA,PREP,PREP_SHA)
    store=m.Store(tmp_path/'complete');totals={k:0 for k in m.TOTAL};uncompressed=0
    rng=np.random.default_rng(812)
    image_sink,path=store.journal('image_mock')
    for i,packet in enumerate(data['packets']):
        def encode(rgb):
            value=rng.normal(size=192).astype(np.float32).tolist()
            image_sink(dict(kind='native_image_attempted',resource='encoder',samples=1))
            image_sink(dict(kind='native_image_attempted',resource='projector',samples=1))
            image_sink(dict(kind='native_image_completed',counts=dict(encoder=1,projector=1),latent=value,
                            rgb_sha256=m.observed.image_sha(rgb)))
            return value
        t=m.original.encode_observed(packet,m.original.template(data,'a'*64,'b'*64,'c'*64),encode)
        store.put(f'task_{i}.json.gz',t)
        for method in m.METHODS:
            name=f'root_{i}_{method}';sink,journal=store.journal(name)
            expected=hashlib.sha256();nbytes=0;kinds={}
            def emit(event):
                nonlocal nbytes
                raw=(m.observed.source.canonical(event)+'\n').encode()
                expected.update(raw);nbytes+=len(raw);kinds[event['kind']]=kinds.get(event['kind'],0)+1
                sink(event)
            result=m.screen.run_direct(t['task'],m.original.config(method),DenseBackend(),emit)
            assert result['status']=='direct_component_completed_not_admitted' and result['error'] is None
            assert len(result['direct_runs'][0])==9 and kinds['jacobian_row']==1536
            assert kinds['trajectory_step']==9 and kinds['joint_logit_jacobian']==8
            assert kinds['native_packet']==54 and kinds['jacobian_forward_packet']==16
            store.put(name+'/report.json.gz',result);store.finish(journal)
            assert m.read_gzip(store.output/name/'report.json.gz')==result
            raw=gzip.decompress(journal.read_bytes())
            assert len(raw)==nbytes and hashlib.sha256(raw).digest()==expected.digest()
            events,eof=m.recover_events(journal)
            assert eof and len(events)==sum(kinds.values())
            for resource,cap in result['budget'].items():
                es=[e for e in result['extended_journal'] if e['resource']==resource]
                assert len(es)==cap and all(e['completed'] for e in es)
                totals[resource]+=len(es)
            uncompressed+=nbytes+len((m.observed.source.canonical(result)+'\n').encode())
    store.finish(path);assert store.close()==[]
    store.put('source_audit.json',data['summary']);store.put('report.json',dict(mock=True,resources=totals),final=True)
    assert totals==m.TOTAL
    size=m.used_bytes(store.output)
    print(f'\nFULL_B8_DENSE_MOCK compressed_bytes={size} uncompressed_arm_bytes={uncompressed} cap_bytes={m.CAP}')
    assert size<=m.CAP//2, 'must retain >=2x conservative complete-output headroom'


def test_v1_protocol_unchanged_except_storage_and_new_binding():
    inputs=dict(pack=PACK,source_sha=PACK_SHA,source_prepare=PREP,source_prepare_sha=PREP_SHA)
    old=m.original.build(**inputs);new=m.build(**inputs)
    for k in old:
        if k not in ('schema','source_sha256','output_cap_bytes'):assert old[k]==new[k]
    assert all(new['source_sha256'][f]==h for f,h in old['source_sha256'].items())
    assert new['output_cap_bytes']==24*1024**2 and old['output_cap_bytes']==56*1024**2
    assert new['direct_resources']==m.TOTAL==dict(predict=280,action_encoder=280,jacobian=32,vjp_directions=6144,
        svd=16,linear_solve=16,proposal=32,selection=32)


def test_actual_small_prepare_verify_no_closure(tmp_path):
    path=tmp_path/'prepare.json'
    p=m.prepare(path,pack=PACK,source_sha=PACK_SHA,source_prepare=PREP,source_prepare_sha=PREP_SHA)
    assert path.stat().st_size<256*1024 and list(tmp_path.iterdir())==[path]
    assert m.verify(path,m.sha(path))==p
    with pytest.raises(ValueError):m.verify(path,'0'*64)


@pytest.mark.parametrize('method',m.METHODS)
def test_identical_B4_B8_fixed_budget(method):
    config=m.original.config(method);b=m.screen.budget(config)
    assert config['B1']==4 and config['B2']==8 and config['extra_line_search_trials']==0
    assert b['predict']==b['action_encoder']==70 and b['jacobian']==8 and b['vjp_directions']==1536


def test_start_gate_before_neural_or_output(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:{})
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=m.FLOOR+m.CAP-1))
    with pytest.raises(ValueError,match='224MiB'):m.run('fixture','a'*64,tmp_path/'output')
    assert not list(tmp_path.iterdir())


def test_every_event_fsync_lossless_prefix_and_footer(tmp_path,monkeypatch):
    fsync=[];real=m.os.fsync
    def record(fd):fsync.append(fd);real(fd)
    monkeypatch.setattr(m.os,'fsync',record)
    s=m.Store(tmp_path/'output');emit,p=s.journal('component')
    for i in range(5):
        value=dict(kind='dense',index=i,array=np.random.default_rng(i).normal(size=(192,20)).tolist(),tiny=1e-300)
        emit(value)
        events,eof=m.recover_events(p);assert events[-1]==value and not eof
    assert len(fsync)==6
    s.finish(p);assert m.recover_events(p)[1] and len(fsync)==7


def test_partial_write_no_false_footer_and_finalization_survives(tmp_path,monkeypatch):
    s=m.Store(tmp_path/'output');emit,p=s.journal('component');emit(dict(saved=1))
    original=s.write
    def fail(path,raw,**kwargs):
        if kwargs.get('append'):raise OSError('mock_storage_error')
        return original(path,raw,**kwargs)
    monkeypatch.setattr(s,'write',fail)
    with pytest.raises(OSError):emit(dict(unsaved=2))
    assert s.close() and m.recover_events(p)==([dict(saved=1)],False)
    s.put('report.json',dict(status='partial_or_failed'),final=True)


def test_quota_floor_exclusive_and_report_lossless(tmp_path,monkeypatch):
    s=m.Store(tmp_path/'output');value=dict(array=np.nextafter(1.,2.),negative=-0.,tiny=1e-300)
    s.put('report.json.gz',value);assert m.read_gzip(s.output/'report.json.gz')==value
    with pytest.raises(FileExistsError):s.put('report.json.gz',value)
    monkeypatch.setattr(m,'used_bytes',lambda _:m.CAP-m.FINAL)
    with pytest.raises(ValueError,match='cap'):s.put('too_large',{})
    s.put('final.json',{},final=True)
    monkeypatch.setattr(m,'used_bytes',lambda _:0)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=m.FLOOR))
    with pytest.raises(ValueError,match='floor'):s.put('floor',{})


def test_corrupt_gzip_not_complete(tmp_path):
    s=m.Store(tmp_path/'output');emit,p=s.journal('component');emit(dict(one=1));s.finish(p)
    raw=bytearray(p.read_bytes());raw[-5]^=1
    bad=tmp_path/'bad.gz';bad.write_bytes(raw)
    with pytest.raises((zlib.error,gzip.BadGzipFile)):m.recover_events(bad)


def test_import_no_neural_runtime():
    import subprocess,sys
    subprocess.run([sys.executable,'-c','import sys;from experiments.spectral_reachability import '+m.NAME+
        ';assert not any(x in sys.modules for x in ("torch","gymnasium","pymunk","stable_worldmodel"))'],check=True)


def test_direct_v1_v2_sink_every_event_every_array_identical(tmp_path):
    old_root=tmp_path/'v1';old_root.mkdir()
    old=m.original.SharedJournal(old_root,old_root/'component')
    new=m.Store(tmp_path/'v2');emit,path=new.journal('component')
    rng=np.random.default_rng(19)
    events=[dict(kind='jacobian_row',row=191,raw_control_derivative=rng.normal(size=20).tolist()),
            dict(kind='joint_logit_jacobian',terminal=rng.normal(size=192).tolist(),jacobian=rng.normal(size=(192,20)).tolist()),
            dict(kind='trajectory_step',restart=0,step=dict(iteration=8,incumbent=dict(
                contexts=rng.normal(size=(2,3,192)).tolist(),commands=rng.normal(size=(10,2)).tolist()),
                derivative=rng.normal(size=(192,20)).tolist())),
            dict(kind='direct_finished',report=dict(extended_journal=[dict(resource='vjp_directions',completed=True)]))]
    for event in events:
        old(event);emit(event)
        assert m.original.read_lines(old.path)==m.recover_events(path)[0]
    new.finish(path)
    assert old.path.read_bytes()==gzip.decompress(path.read_bytes())
    assert m.recover_events(path)==(events,True)
