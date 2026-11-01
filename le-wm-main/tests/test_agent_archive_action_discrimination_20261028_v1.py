"""RAM / saved byte fixtures only. No torch, environment or model loader."""
from copy import deepcopy
import gzip
from io import BytesIO
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments.spectral_reachability import agent_archive_action_discrimination_20261028_v1 as m


META={'mean':[-.0087,.0068],'std':[.2019,.2002]}


def base(i=0):
    row=m.census()[4*i]
    raw=(np.arange(30,dtype=np.float32).reshape(3,5,2)-10)/np.float32(100)
    return dict(episode_ordinal=row['episode_ordinal'],condition=row['condition'],kind=row['kind'],
        raw_packets=raw,latent_history=np.full((3,192),i,np.float32),tokens=m.normalize(raw,META))


@pytest.mark.parametrize('arm',m.ARMS)
def test_mutation_preserves_history(arm):
    b=base();snapshot=deepcopy(b);r=m.mutate(b,arm,META)
    assert m.bits(r['latent_history'],b['latent_history'])
    assert m.bits(r['raw_packets'][:2],b['raw_packets'][:2])
    assert m.bits(r['tokens'][:2],b['tokens'][:2])
    expected={'factual':b['raw_packets'][-1], 'zero_outgoing':np.zeros((5,2),np.float32),
        'reverse_time_outgoing':b['raw_packets'][-1,::-1], 'swap_xy_outgoing':b['raw_packets'][-1,:,::-1]}[arm]
    assert m.bits(r['raw_packets'][-1],expected)
    assert m.bits(r['tokens'],m.normalize(r['raw_packets'],META))
    for k in ('latent_history','raw_packets','tokens'):assert m.bits(b[k],snapshot[k])


@pytest.mark.parametrize('bad',('future','H2','dtype','std'))
def test_invalid_inputs_closed(bad):
    b=base();meta=deepcopy(META)
    if bad=='future':b['goal']=np.ones(192,np.float32)
    if bad=='H2':b['kind']='fresh_H2_15_to_20'
    if bad=='dtype':b['raw_packets']=b['raw_packets'].astype(np.float64)
    if bad=='std':meta['std'][0]=0
    with pytest.raises(ValueError):m.mutate(b,'factual',meta)


def test_literal_zero_not_normalized_zero():
    r=m.mutate(base(),'zero_outgoing',META)
    assert np.count_nonzero(r['raw_packets'][-1])==0
    assert np.count_nonzero(r['tokens'][-1])==10


@pytest.mark.parametrize('failure',('trailing','member','truncated','bound'))
def test_gzip_strict(failure):
    raw=gzip.compress(b'x'*100,mtime=0);limit=100
    if failure=='trailing':raw+=b'x'
    if failure=='member':raw+=gzip.compress(b'other')
    if failure=='truncated':raw=raw[:-1]
    if failure=='bound':limit=99
    with pytest.raises(ValueError):m.gunzip(raw,limit)


def test_gzip_exact():assert m.gunzip(gzip.compress(b'abc',mtime=0),3)==b'abc'


class Tensor:
    """Numpy-only trace protocol fixture, NOT CPU neural execution."""
    device=NS(type='cuda')
    def __init__(self,x):self.x=np.asarray(x,np.float32)
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.x


class Module:
    def __init__(self,fn):self.fn=fn;self.pre=[];self.post=[]
    def register_forward_pre_hook(self,fn,**kw):
        self.pre.append(fn);return NS(remove=lambda:self.pre.remove(fn))
    def register_forward_hook(self,fn,**kw):
        self.post.append(fn);return NS(remove=lambda:self.post.remove(fn))
    def __call__(self,*args,**kw):
        for f in self.pre:f(self,args,kw)
        out=self.fn(*args,**kw)
        for f in self.post:f(self,args,kw,out)
        return out


class Model:
    def __init__(self):
        self.encoder=Module(lambda x:x);self.projector=Module(lambda x:x)
        self.action_encoder=Module(lambda x:Tensor(np.repeat(x.x.sum(-1,keepdims=True),192,-1)))
        self.predictor=Module(lambda z,a:Tensor(z.x+a.x))
        self.pred_proj=Module(lambda x:x)
    def predict(self,z,a):
        y=self.predictor(z,a);q=self.pred_proj(Tensor(y.x.reshape(3,192)))
        return Tensor(q.x.reshape(1,3,192))


def fixture_prediction(r):
    return r['latent_history']+np.repeat(r['tokens'].sum(-1,keepdims=True),192,-1)


@pytest.fixture
def assets(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    upstream=m.Store(tmp_path/'results'/(m.NAME+'_fixture_upstream'))
    bases=[];refs=[]
    for i in range(12):
        b=base(i);identity={k:b[k] for k in ('episode_ordinal','condition','kind')}
        bases.append(dict(base_index=i,old_index=i,**identity,
            inputs={k:m.old.save_array(upstream,f'b{i}.{k}',b[k]) for k in ('latent_history','raw_packets','tokens')},
            factual_output=m.old.save_array(upstream,f'b{i}.out',fixture_prediction(b))))
        refs.append(dict(episode_ordinal=b['episode_ordinal'],condition=b['condition'],step=15 if b['kind']==m.KINDS[0] else 20,
            latent=m.old.save_array(upstream,f'b{i}.label',np.full(192,i/10,np.float32))))
    upstream.close()
    inv={f.name:m.sha(f) for f in upstream.output.iterdir()}
    p=dict(old_report=str(upstream.output/'report.json'),upstream_inventory=inv,bases=bases,label_refs=refs,
        normalization=META,expected_model_hash='fixture',inherited_resources={'fixture_only':True})
    monkeypatch.setattr(m,'OLD_REPORT',upstream.output/'report.json')
    monkeypatch.setattr(m,'verify',lambda *args:deepcopy(p))
    return p,tmp_path


def execute_fixture(p,store,rows,loader=None,corrupt=False):
    model=Model();trace=m.Trace(model,store,lambda:None,lambda:True)
    def predict(r,i):
        trace.active=('prediction',i,r)
        out=model.predict(Tensor(r['latent_history'][None]),model.action_encoder(Tensor(r['tokens'][None]))).x[0].copy()
        trace.finish('prediction',i)
        if corrupt:out[0,0]=np.nextafter(out[0,0],np.float32(np.inf))
        return out
    with trace:result=m.execute(p,predict,store,rows,loader or m.labels)
    assert 'predict' not in model.__dict__
    assert all(not mod.pre and not mod.post for mod in (model.encoder,model.projector,model.action_encoder,model.predictor,model.pred_proj))
    return result,trace


def make_report(p,root,store,rows,trace):
    store.close()
    report=dict(schema=m.NAME+'.report',status='all48_action_discrimination_saved_not_admitted',contract=m.contract(),
        prepared=str(root/'fixture_prepare.json'),prepare_sha256='fixture_sha',requests=rows,errors=[],forbidden_AD_attempts=[],
        first_error=None,source_verified=True,model_hash_before='fixture',model_hash_after='fixture',model_unchanged=True,
        resources=trace.counts(),module_ledger=trace.entries,AD=0,physics=0,training=False,P1_admission=False,
        training_lineage_verified=False,scientific_winner=None,wall_seconds=1.,
        limits=dict(elapsed_seconds=1.,peak_allocated_bytes=0,peak_reserved_bytes=0,minimum_observed_free_bytes=m.FLOOR),
        artifact_sha256={f.name:m.sha(f) for f in store.output.iterdir()},journal_chain_tail=store.tail,journal_events=store.event_index)
    store.put('report.json',report,final=True)
    return store.output/'report.json',report


def test_full48_native_saved_consume_and_seal(assets):
    p,root=assets;store=m.Store(root/'results'/(m.NAME+'_fixture_run'));rows=m.census()
    store.put('binding.json',dict(prepared=str(root/'fixture_prepare.json'),prepare_sha256='fixture_sha'))
    store.put('input_receipts.json',p)
    def label_loader(prepared):
        assert len(rows)==48 and all(r['status']=='saved' for r in rows)
        assert store.event_index==48*14+1 # attempt + 4*(attempt/input/return) + saved + seal
        return m.labels(prepared)
    analysis,trace=execute_fixture(p,store,rows,label_loader)
    assert len(analysis['rows'])==48 and len(analysis['contrasts'])==36 and len(analysis['render_relative_contrasts'])==24
    path,report=make_report(p,root,store,rows,trace)
    receipt=m.consume(root/'fixture_prepare.json','fixture_sha',path,m.sha(path))
    assert receipt['factual_full3_exact']==12 and receipt['module_entries']==192
    assert receipt['P1_admission'] is False
    # Save a self-consistently rehashed scalar mutation: consumer must recompute, not trust analysis.
    analysis['rows'][0]['MSE']+=1
    with (path.parent/'analysis.json').open('wb') as f:f.write(m.canonical(analysis))
    report['artifact_sha256']['analysis.json']=m.sha(path.parent/'analysis.json')
    with path.open('wb') as f:f.write(m.canonical(report))
    with pytest.raises(ValueError,match='independent recomputation'):m.consume(root/'fixture_prepare.json','fixture_sha',path,m.sha(path))


def test_factual_one_bit_failure_no_labels(assets):
    p,root=assets;store=m.Store(root/'results'/(m.NAME+'_fixture_failed'));rows=m.census();called=[]
    with pytest.raises(ValueError,match='exactly reproduce'):
        execute_fixture(p,store,rows,lambda p:called.append(True),corrupt=True)
    assert not called and rows[0]['status']=='attempted' and rows[1]['status']=='not_started'
    assert (store.output/rows[0]['output']['file']).exists() # failed output preserved
    store.close()


def test_trace_rejects_extra_or_encoder(assets):
    p,root=assets;store=m.Store(root/'results'/(m.NAME+'_fixture_guard'))
    trace=m.Trace(Model(),store,lambda:None,lambda:True);trace.active=('prediction',0,base())
    with pytest.raises(ValueError,match='no encoder'):trace.begin('encoder',(),{})
    trace.entries=[dict(resource='predict') for _ in range(48)]
    with pytest.raises(ValueError,match='predispatch48'):trace.begin('predict',(),{})
    store.close()


@pytest.mark.parametrize('case',('trailing_npy','raw_sha','path','nonfinite'))
def test_arrays_fail_closed(assets,case):
    p,root=assets;rec=deepcopy(p['bases'][0]['factual_output']);inv=deepcopy(p['upstream_inventory'])
    path=root/'results'/(m.NAME+'_fixture_upstream')/rec['file']
    if case=='raw_sha':rec['raw_sha256']='0'*64
    elif case=='path':rec['file']='../'+rec['file']
    else:
        raw=m.gunzip(path.read_bytes(),8*1024**2)
        if case=='trailing_npy':raw+=b'bad'
        else:
            buf=BytesIO();np.save(buf,np.full((3,192),np.nan,np.float32),allow_pickle=False);raw=buf.getvalue()
        with path.open('wb') as f:f.write(gzip.compress(raw,mtime=0))
        rec['sha256']=inv[rec['file']]=m.sha(path);rec['npy_sha256']=m.digest(raw)
    with pytest.raises(ValueError):m.Arrays(path.parent,inv).load(rec)


def test_nested_journal_return_order():
    entries=[dict(index=i,phase='prediction',sample=0,resource=k,completed=True,inputs=[],kwargs={},output=None) for i,k in enumerate(m.ORDER)]
    events=[]
    for i in (0,1,2,3):
        e=entries[i];events.append(dict(kind='native_module_attempt',entry={k:e[k] for k in ('index','phase','sample','resource')}|{'completed':False}))
        events.append(dict(kind='native_inputs_saved',entry={k:v for k,v in e.items() if k!='output'}|{'completed':False}))
        if i in (0,2,3):events.append(dict(kind='native_module_return',entry=e))
    events.append(dict(kind='native_module_return',entry=entries[1]))
    m.validate_events(events,entries)
    bad=deepcopy(events);bad[-1],bad[-2]=bad[-2],bad[-1]
    with pytest.raises(ValueError):m.validate_events(bad,entries)


def test_census_and_contract():
    rows=m.census();assert len(rows)==48 and len({(r['episode_ordinal'],r['condition'],r['kind']) for r in rows})==12
    assert m.contract()['counts']==m.EXPECTED and sum(m.EXPECTED.values())==192
    assert m.contract()['projected_token_samples']==144


def test_confirm_required_before_any_model(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('must reject before verify/load'))
    with pytest.raises(ValueError,match='authorization'):m.run('missing','missing','missing')


def test_duplicate_json_and_nonfinite(tmp_path):
    path=tmp_path/'bad.json'
    for raw in (b'{"a":1,"a":2}',b'{"a":NaN}'):
        with path.open('wb') as f:f.write(raw)
        with pytest.raises(ValueError):m.read(path)


def test_storage_caps_floor_and_no_overwrite(assets,monkeypatch):
    p,root=assets;path=root/'results'/(m.NAME+'_fixture_cap');store=m.Store(path)
    store.put('one.json',{})
    with pytest.raises(FileExistsError):store.put('one.json',{})
    with pytest.raises(ValueError):store._write('too_big',b'x'*m.CAP)
    store.close()
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:NS(free=m.FLOOR+m.CAP-1))
    with pytest.raises(ValueError):m.Store(root/'results'/(m.NAME+'_fixture_floor'))


def test_prepare_full_semantic_binding(tmp_path,monkeypatch):
    (tmp_path/'results').mkdir();monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(m,'material',lambda:dict(schema='fixture',all_bindings={'a':'b'}))
    path=tmp_path/'results'/(m.NAME+'_fixture_prepare.json');result=m.write_prepare(path)
    assert m.verify(path,result['prepare_sha256'])['all_bindings']=={'a':'b'}
    with pytest.raises(ValueError):m.write_prepare(path)
    monkeypatch.setattr(m,'material',lambda:dict(schema='fixture',all_bindings={'a':'changed'}))
    with pytest.raises(ValueError):m.verify(path,result['prepare_sha256'])
