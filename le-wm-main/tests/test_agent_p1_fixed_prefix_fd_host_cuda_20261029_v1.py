"""Temporary-disk/RAM native fixtures only; never execute production CUDA."""
from copy import deepcopy
import time
import pytest
from experiments.spectral_reachability import agent_p1_fixed_prefix_fd_host_cuda_20261029_v1 as h
import test_agent_p1_fixed_prefix_saved_handoff_20261029_v1 as fixture
import test_agent_p1_nonlinear_suffix_library_20261028_v2 as native

inputs=fixture.inputs
complete=fixture.complete
group=fixture.group
saved=fixture.saved
panel=fixture.panel


def specs(panel):
    p,r=panel; out=[]
    for q,row in zip(p['groups'],r['rows'],strict=True):
        n=sum(x['valid'] for x in h.fd.pairs(q,row['result']))
        out.append(dict(index=q['index'],request=q,source_result=row['result'],valid_pairs=n,
                        invalid_causes=h.invalid_causes(q,row['result']),resource_caps=h.F.caps(2*n,0)))
    return out


def fake_prepared(panel):
    s=specs(panel)
    return dict(specs=s,bindings={},expected_model_hash='a'*64,planned_valid_pairs=sum(v['valid_pairs'] for v in s))


def publish_fixture(store,rows,status,errors,p):
    store.finalizing=True;output=store.path
    artifacts={str(f.relative_to(output)):h.sha(f) for f in output.rglob('*') if f.is_file()}
    precision=dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,math=True,flash=False,efficient=False,cudnn_sdp=False)
    model=dict(original_state_dict='a'*64)
    report=dict(schema=h.NAME+'.report',status=status,contract=h.contract(),prepared='/fixture/prepare',prepare_sha256='b'*64,
        rows=rows,errors=errors,group_denominator=64,pair_denominator=2560,source_verified=True,source_manifest_sha256=h.digest({}),
        model_before=model,model_after=model,host=None,precision_before=precision,precision_after=precision,
        CUDA_peak_bytes=[0,0],wall_seconds=1,artifact_sha256=artifacts,P1_admission=False,freshness=False,physical_release=False)
    store.put('report.json',report,final=True);return h.sha(output/'report.json')


def test_partial_durable_consumer_and_tamper(tmp_path,monkeypatch,panel):
    (tmp_path/'results').mkdir();monkeypatch.setattr(h,'ROOT',tmp_path)
    p=fake_prepared(panel);monkeypatch.setattr(h,'verify',lambda *_:p)
    output=tmp_path/'results'/(h.NAME+'_partial_fixture');store=h.Store(output,time.monotonic())
    store.put('binding.json',dict(prepared='/fixture/prepare',prepare_sha256='b'*64));rows=h.roster();started=[]
    def factory(task,journal,ledger):
        started.append(1)
        if len(started)==2:raise RuntimeError('injected')
        return native.RAM(task,journal,ledger)
    with pytest.raises(RuntimeError,match='injected'):h.dispatch(p,store,rows,lambda:None,factory)
    assert rows[0]['status']=='sealed' and rows[1]['status']=='failed_partial'
    assert all(v['status']=='not_started' for v in rows[2:])
    sha=publish_fixture(store,rows,'partial_failclosed',[dict(error='injected')],p)
    receipt=h.consume(output,sha)
    assert not receipt['complete'] and receipt['pair_denominator']==2560 and receipt['audits'][0]['native_forwards']==80
    assert receipt['actual_new_compute']['completed']==h.F.caps(80,0)
    (output/'unexpected').write_bytes(b'x')
    with pytest.raises(ValueError,match='file census'):h.consume(output,sha)


def test_full_64_durable_fixture_and_saved_rebuild(tmp_path,monkeypatch,panel):
    (tmp_path/'results').mkdir();monkeypatch.setattr(h,'ROOT',tmp_path)
    p=fake_prepared(panel);monkeypatch.setattr(h,'verify',lambda *_:p)
    output=tmp_path/'results'/(h.NAME+'_complete_fixture');store=h.Store(output,time.monotonic())
    store.put('binding.json',dict(prepared='/fixture/prepare',prepare_sha256='b'*64));rows=h.roster()
    h.dispatch(p,store,rows,lambda:None,lambda task,journal,ledger:native.RAM(task,journal,ledger))
    sha=publish_fixture(store,rows,'complete_model_only',[],p);receipt=h.consume(output,sha)
    assert receipt['complete'] and len(receipt['audits'])==64 and receipt['pair_denominator']==2560
    assert receipt['actual_new_compute']['attempted']==receipt['actual_new_compute']['completed']==h.F.caps(5120,0)


def test_no_authorization_no_verify(monkeypatch):
    monkeypatch.setattr(h,'verify',lambda *_:pytest.fail('unexpected verify'))
    with pytest.raises(ValueError,match='authorization'):h.run('x','y','z','no')


def test_chart_boundary_explicit_reason(saved):
    import numpy as np
    q,r,_,_=deepcopy(saved);x=np.zeros(10,np.float64);x[0]=h.F.CAP
    q['suffix_logits']=x.tolist();r['iterates'][0]['logits']=x.tolist();r['iterates'][0]['raw_bits']=h.F.bits(h.F.chart(x))
    reasons=h.invalid_causes(q,r)
    assert any(v['space']=='chart_logit' and v['coordinate']==0 and 'chart_ball_boundary' in v['reasons'] for v in reasons)
