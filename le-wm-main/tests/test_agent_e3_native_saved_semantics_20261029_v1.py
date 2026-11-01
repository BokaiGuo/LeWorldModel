from copy import deepcopy
from pathlib import Path
import pytest
from experiments.spectral_reachability import agent_e3_native_saved_semantics_20261029_v1 as s

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def actual():
    data=(ROOT/'results/agent_e3_chunked_identity_host_20261029_v1_MAIN05/events.jsonl').read_bytes()
    return [s.pub.base.decode(x) for x in data.splitlines()]


def ev(index,resource,detail,kind='record'):
    return dict(index=index,origin='physical',resource=resource,detail=detail,kind=kind,token=index)


def test_actual_persisted_semantics(actual):
    out=s.audit(actual,'cell00',True)
    assert out['observations']==32 and out['snapshot_occurrences']==37
    assert out['links']==['warm10_original_root','root_to_plan0_snapshot','cp5_guard','encode_10','encode_15']
    assert out['dispatched']==out['returned']==20 and out['cleanup']=='returned'
    assert out['full_failure_semantics'] is False


def test_tampered_root_snapshot_is_rejected(actual):
    events=deepcopy(actual)
    root=next(e for e in events if e['resource']=='native_original_root_accepted')
    root['detail']['snapshot']['bodies']['block']['float64_bits']='00'
    with pytest.raises(ValueError,match='body dynamics bits'):s.audit(events,'cell00',True)


def test_mismatched_cp5_rejected(actual):
    events=deepcopy(actual)
    cp=next(e for e in events if e['resource']=='native_actual_cp5_guard')
    cp['detail']['observation']['pixel_sha256']='bad'
    with pytest.raises(ValueError,match='cp5 exact'):s.audit(events,'cell00',True)


def test_failed_dispatch_remains_unreturned_and_cleanup_reported():
    events=[ev(0,'native_authorized_dispatch',dict(episode_id='e',step=1,raw_bits='00')),
            ev(1,'native_episode_failure',dict(error=dict(type='RuntimeError',message='broken'))),
            ev(2,'native_cleanup_intent',dict(episode_id='e')),
            ev(3,'native_cleanup_failure',dict(episode_id='e')),
            ev(4,'launcher_failure',dict(type='RuntimeError',message='broken'))]
    out=s.audit(events,'e',False)
    assert (out['dispatched'],out['returned'])==(1,0)
    assert out['unreturned_dispatch']['step']==1 and out['cleanup']=='failed'
    assert out['native_failure']['error']['message']=='broken' and not out['full_failure_semantics']


def test_no_dispatch_after_stop():
    events=[ev(0,'native_authorized_dispatch',dict(episode_id='e',step=1,raw_bits='aa')),
            ev(1,'native_control_return',dict(episode_id='e',step=1,raw_bits='aa',terminated=True,truncated=False)),
            ev(2,'native_authorized_dispatch',dict(episode_id='e',step=2,raw_bits='bb'))]
    with pytest.raises(ValueError,match='after stop'):s.audit(events,'e',False)


def test_missing_or_duplicated_cleanup_does_not_become_success():
    events=[ev(0,'native_cleanup_return',dict(episode_id='e'))]
    with pytest.raises(ValueError,match='cleanup order'):s.audit(events,'e',False)
    events=[ev(0,'native_cleanup_intent',dict(episode_id='e')),
            ev(1,'native_cleanup_return',dict(episode_id='e')),
            ev(2,'native_cleanup_failure',dict(episode_id='e'))]
    with pytest.raises(ValueError,match='exclusive'):s.audit(events,'e',False)


def test_complete_flag_cannot_be_invented():
    with pytest.raises(ValueError,match='prerequisites'):s.audit([],'e',True)
