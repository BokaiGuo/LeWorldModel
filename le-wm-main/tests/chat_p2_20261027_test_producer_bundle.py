"""Pure metadata/entry-hook tests. No real torch, neural, or physics operations."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import chat_p2_20261027_producer_bundle as b
from experiments.spectral_reachability.p2_full20_sampling_20261027_v2 import actual_entries


def row():
    return dict(root_index=0,root_sha256='a'*64,root=dict(root_id='a'*64,context=dict(history=[1,2,3]),
                goal_rgb_sha256='b'*64),action_domain=dict(low=[-1],high=[1]),metric_sha256='c'*64)


@pytest.mark.parametrize('kind',b.KINDS)
def test_identity_projection_preserves_full_history_goal_and_domain(kind):
    r = row()
    p = {'prepared':[r]} if kind == 'strong_sampling' else {'context':{'cases':[r]}}
    assert b.identity_rows(kind,p)[0] == r


@pytest.mark.parametrize('field',['root','action_domain','metric_sha256'])
def test_cross_producer_identity_mismatch_fails_before_snapshot(field,monkeypatch):
    first=row();second=deepcopy(first)
    second[field] = dict(second[field],extra='different') if isinstance(second[field],dict) else 'd'*64
    def native(kind,path):
        r = second if kind=='no_composition_sra' else first
        p = {'prepared':[r]} if kind == 'strong_sampling' else {'context':{'cases':[r]}}
        return SimpleNamespace(__file__=b.__file__),b.Path('/fixture'),p,dict(source_seals=[])
    monkeypatch.setattr(b,'native',native)
    with pytest.raises(ValueError,match='cross-producer'):
        b.prepare(dict.fromkeys(b.KINDS,'fixture'))


def test_native_hooks_measure_calls_batch_samples_tokens_and_failures_separately():
    class Encoder:
        def forward(self,x):
            return x
    class Model:
        def __init__(self):
            self.action_encoder=Encoder()
        def predict(self,x):
            self.action_encoder.forward(x)
            if x.fail:
                raise RuntimeError('retained failure')
            return x
    model=Model();original_predict=model.predict;original_encoder=model.action_encoder.forward
    with actual_entries(model,SimpleNamespace(_p2_scope={'candidate':0})) as entries:
        model.predict(SimpleNamespace(shape=(4,3),fail=False))
        with pytest.raises(RuntimeError,match='retained'):
            model.predict(SimpleNamespace(shape=(2,3),fail=True))
    totals=b.entry_counts(entries)['entry_streams']
    assert totals['predict']==dict(api_calls=2,batch_samples=6,context_tokens=18,failed_api_calls=1)
    assert totals['action_encoder']==dict(api_calls=2,batch_samples=6,context_tokens=18,failed_api_calls=0)
    assert model.predict==original_predict and model.action_encoder.forward==original_encoder
    assert len(entries)==4  # Failure is never removed/refunded.


@pytest.mark.parametrize('field,value',[('samples',True),('samples',0),('context_tokens',None),('completed',1),('kind','unknown')])
def test_malformed_entry_not_coerced_to_zero(field,value):
    entry=dict(kind='predict',samples=1,context_tokens=3,completed=True)
    entry[field]=value
    with pytest.raises(ValueError):
        b.entry_counts([entry])


def test_every_producer_retained_after_missing_composition(monkeypatch):
    monkeypatch.setattr(b,'verify',lambda *a:True)
    monkeypatch.setattr(b,'saved_metadata',lambda *a:{})
    def exporter(kind,path):
        if kind=='composition':
            raise FileNotFoundError('no report')
        return {}
    monkeypatch.setattr(b.life,'consume_producer',exporter)
    bundle=dict(producers={k:dict(path='fixture') for k in b.KINDS},root_denominator=4)
    report=b.run_saved(bundle,'a'*64)
    assert len(report['rows'])==report['producer_denominator']==3
    assert report['root_denominator']==4 and report['status']=='incomplete_native_exports'
    assert report['rows'][0]['error']['type']=='FileNotFoundError'
    assert all(r['status']=='native_full_export_verified' for r in report['rows'][1:])
    with pytest.raises(ValueError,match='complete composition'):
        b.consume(bundle,'a'*64,report)


def test_bundle_external_sha_rejected_before_native_load():
    with pytest.raises(ValueError,match='external bundle'):
        b.verify({},'a'*64)


def test_duplicate_root_not_filtered():
    r=row();s=deepcopy(r);s['root_index']=1
    with pytest.raises(ValueError,match='unique complete'):
        b.identity_rows('strong_sampling',dict(prepared=[r,s]))


def test_run_saved_propagates_interrupt_without_fake_success(monkeypatch):
    monkeypatch.setattr(b,'verify',lambda *a:True)
    def exporter(*a):
        raise KeyboardInterrupt('stop')
    monkeypatch.setattr(b.life,'consume_producer',exporter)
    with pytest.raises(KeyboardInterrupt):
        b.run_saved(dict(producers={k:dict(path='fixture') for k in b.KINDS}),'a'*64)


def test_started_failure_preserves_primary_and_blocks_retry(tmp_path,monkeypatch):
    monkeypatch.setattr(b.life,'PROJECT',tmp_path)
    (tmp_path/'results').mkdir()
    path=tmp_path/'results/chat_p2_20261027_fixture.json'
    out=tmp_path/'results/chat_p2_20261027_failed.json'
    b.life.write(path,{})
    primary=KeyboardInterrupt('retained primary')
    def fail(*a):
        raise primary
    monkeypatch.setattr(b,'run_saved',fail)
    with pytest.raises(KeyboardInterrupt) as raised:
        b.run_saved_file(path,'a'*64,out)
    assert raised.value is primary
    assert b.life.read(out.with_name(out.stem+'_failure.json'))['error_type']=='KeyboardInterrupt'
    with pytest.raises(FileExistsError):
        b.run_saved_file(path,'a'*64,out)


def test_sampling_metadata_keeps_both_arms_and_entry_streams(tmp_path):
    e=dict(kind='predict',samples=4,context_tokens=12,completed=False)
    arms=[dict(arm=k,attempts=[{},{}],actual_model_entries=[e]) for k in ('cem','random')]
    report=dict(rows=[dict(root_sha256='a'*64,rows=arms)])
    p=tmp_path/'report.json';p.write_bytes(b.life.graph.encode(report))
    metadata=b.saved_metadata('strong_sampling',tmp_path)['rows'][0]
    assert metadata['candidate_denominators']==dict(cem=2,random=2)
    assert metadata['actual_entries']['entry_streams']['predict']['batch_samples']==8
    assert metadata['actual_entries']['entry_streams']['predict']['failed_api_calls']==2
