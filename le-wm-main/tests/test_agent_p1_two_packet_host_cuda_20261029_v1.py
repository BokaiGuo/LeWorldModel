import time
from copy import deepcopy
import pytest
from experiments.spectral_reachability import agent_p1_two_packet_host_cuda_20261029_v1 as m
import test_agent_p1_two_packet_uphill_generator_20261029_v1 as fixture
import test_agent_p1_nonlinear_suffix_library_20261028_v2 as native

inputs=fixture.inputs
complete=fixture.complete
requests=fixture.requests


@pytest.mark.parametrize('failure',[False,True,'after_paid_gradient'])
def test_disk_eight_arm_consume_and_partial(requests,tmp_path,monkeypatch,failure):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    p=dict(schema=m.NAME+'.prepare',production_ready=True,bindings={},arguments=dict(source_report_sha='c'*64),
           authority=dict(rows=[dict(request=q) for q in requests],expected_model_hash='d'*64))
    rows=m.roster();store=m.Store(tmp_path/'results'/(m.NAME+'_RAM'),time.monotonic());calls=0
    def factory(task,journal,ledger):
        nonlocal calls
        calls+=1
        if failure is True and calls==4:raise RuntimeError('injected before native')
        if failure=='after_paid_gradient' and calls==4:
            class Broken(native.RAM):
                def gradient(self,c,a,label):
                    d=super().gradient(c,a,label)
                    if label.endswith('_gradient_2'):raise RuntimeError('injected after paid gradient')
                    return d
            return Broken(task,journal,ledger)
        return native.RAM(task,journal,ledger)
    if failure:
        with pytest.raises(RuntimeError):m.dispatch(p,store,rows,lambda:None,factory)
        assert [[a['status'] for a in r['arms']] for r in rows]==[
            ['sealed','sealed'],['sealed','failed_partial'],['not_started','not_started'],['not_started','not_started']]
    else:
        m.dispatch(p,store,rows,lambda:None,factory)
        assert all(a['status']=='sealed' for r in rows for a in r['arms'])
    prepared=tmp_path/'RAM_prepare.json';prepared.write_bytes(m.raw(p));h=m.sha(prepared)
    store.put('binding.json',dict(prepared=str(prepared),prepare_sha256=h))
    monkeypatch.setattr(m,'verify',lambda path,expected:p if str(path)==str(prepared) and expected==h else m.require(False,'fixture prepare'))
    precision=dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,math=True,flash=False,efficient=False,cudnn_sdp=False)
    artifacts={str(f.relative_to(store.path)):m.sha(f) for f in store.path.rglob('*') if f.is_file()}
    report=dict(schema=m.NAME+'.report',status='partial_failclosed' if failure else 'complete_model_only',contract=m.contract(),
        prepared=str(prepared),prepare_sha256=h,rows=rows,errors=['injected'] if failure else [],fixed_denominator=4,arm_denominator=8,
        candidate_denominator=72,source_verified=True,source_manifest_sha256=m.digest({}),model_before=dict(original_state_dict='d'*64),
        model_after=dict(original_state_dict='d'*64),host=None,precision_before=precision,precision_after=precision,
        CUDA_peak_bytes=[0,0],wall_seconds=1.,artifact_sha256=artifacts,P1_admission=False,freshness=False,physical_release=False)
    store.put('report.json',report,final=True)
    result=m.consume(store.path,m.sha(store.path/'report.json'))
    assert result['complete'] is (not failure) and len(result['rows'])==4
    assert all(a['scalar_objective']['checked_scalar_losses']==16 for row in result['audits'] for a in row if a is not None)
    assert result['actual_new_compute']['completed']['predict_API']==(115 if failure=='after_paid_gradient' else 108 if failure else 288)
    assert result['actual_new_compute']['completed']['grad_API']==(51 if failure=='after_paid_gradient' else 48 if failure else 128)
    (store.path/'unexpected').write_bytes(b'x')
    with pytest.raises(ValueError):m.consume(store.path,m.sha(store.path/'report.json'))


def test_fixed_contract_pins_and_authorization(tmp_path):
    c=m.contract();assert c['fixed_slots']==4 and c['fixed_arms']==8 and c['fixed_candidate_denominator']==72
    assert c['per_arm']==m.F.caps(36,16) and c['total']==m.F.caps(288,128)
    for module,h in m.PINS:assert m.sha(module.__file__)==h
    assert m.sha(m.scalar_audit.__file__)==m.SCALAR_SOURCE_SHA
    with pytest.raises(ValueError):m.run('missing','bad',str(tmp_path/'x'),'wrong')
