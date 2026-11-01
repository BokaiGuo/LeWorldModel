import time
import pytest
from experiments.spectral_reachability import agent_e3_continuous_host_launcher_20261029_v1 as m


def binding():
    return dict(cell=0, manifest_path='/fixture/manifest.json', manifest_sha='1'*64)


def test_disk_authority_before_dispatch_and_final_guard(tmp_path):
    order=[]
    def authority():
        order.append('authority');return binding()
    def perform(b, store):
        order.append('execute')
        m._record(store, 'fixture_cleanup_return', {})
        order.append('guard_after')
        return dict(status='completed20')
    result=m._run_lifecycle(tmp_path/'job',time.monotonic(),authority,perform)
    assert order == ['authority','execute','guard_after']
    seal=result['publication']
    assert m.pub.consume(tmp_path/'job',seal['report_sha256'],seal['completion_sha256'])['complete']
    events=[m.pub.base.decode(x) for x in (tmp_path/'job/events.jsonl').read_bytes().splitlines()]
    assert [x['resource'] for x in events] == ['launcher_started','launcher_authority_verified',
                                                'fixture_cleanup_return','launcher_guarded_return']


@pytest.mark.parametrize('stage', ['authority','execute','after_cleanup'])
def test_failure_persisted_without_false_success(tmp_path,stage):
    def authority():
        if stage == 'authority': raise RuntimeError('authority fixture')
        return binding()
    def perform(b,store):
        if stage == 'after_cleanup': m._record(store,'fixture_cleanup_return',{})
        raise RuntimeError(stage)
    result=m._run_lifecycle(tmp_path/'job',time.monotonic(),authority,perform)
    assert result['complete'] is False and result['error_type']=='RuntimeError'
    seal=result['publication']
    saved=m.pub.consume(tmp_path/'job',seal['report_sha256'],seal['completion_sha256'])
    assert not saved['complete']
    events=[m.pub.base.decode(x) for x in (tmp_path/'job/events.jsonl').read_bytes().splitlines()]
    assert events[-1]['resource']=='launcher_failure' and events[-1]['detail']['message']==stage.replace('authority','authority fixture')
    assert all(e['resource']!='launcher_guarded_return' for e in events)


def test_expired_before_authority_does_not_dispatch(tmp_path):
    called=[]
    with pytest.raises(ValueError):
        m._run_lifecycle(tmp_path/'job',time.monotonic()-1801,
                         lambda: called.append('authority'),lambda b,s: called.append('execute'))
    assert called == []


def test_partial_disk_consumer_no_forged_authority(tmp_path):
    from experiments.spectral_reachability import agent_e3_continuous_saved_consumer_20261029_v1 as saved
    def fail():raise RuntimeError('authority broken')
    result=m._run_lifecycle(tmp_path/'job',time.monotonic(),fail,lambda b,s:None)
    seal=result['publication']
    audit=saved.consume(tmp_path/'job',seal['report_sha256'],seal['completion_sha256'],
                        '/unread/authority.json','0'*64)
    assert audit['status']=='partial' and audit['error']['message']=='authority broken'
    assert audit['authority_verified_here'] is False and audit['full_saved_strict'] is False
