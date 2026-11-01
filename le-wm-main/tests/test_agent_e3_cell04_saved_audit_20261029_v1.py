from copy import deepcopy
import pytest
from experiments.spectral_reachability import agent_e3_cell04_saved_audit_20261029_v1 as m


def ev(resource,detail,kind='record'):
    return dict(origin='physical',resource=resource,kind=kind,detail=detail)


def sample(monkeypatch):
    goal=[0.]*7;state=[0.]*7
    row=dict(status='native_initial_success_control5',initial_success=True,initial_native_distance=0.,native_success=True,final_state=state)
    monkeypatch.setattr(m.native.terminal,'read',lambda _:row)
    events=[ev('launcher_authority_verified',dict(cell=4,manifest_path=m.AUTH,manifest_sha256=m.AUTH_SHA)),
            ev('native_actual_root_guard',dict(episode_id='cell04',new_goal=goal,root_observation={'state':state})),
            ev('native_actual_cp5_guard',dict(observation={'state':state})),
            ev('native_initial_native_evaluation',dict(episode_id='cell04',boundary=5,success=True,distance=0.)),
            ev('terminal_row',{},'return')]
    return events,dict(transport_complete=True)


def test_initial_success_is_not_padded(monkeypatch):
    events,audit=sample(monkeypatch);result=m.project(events,audit)
    assert result['initial_success'] and result['status']=='native_initial_success_control5'


def test_changed_initial_or_cell_rejected(monkeypatch):
    events,audit=sample(monkeypatch);events[3]['detail']['success']=False
    with pytest.raises(ValueError):m.project(events,audit)
    events,audit=sample(monkeypatch);events[0]['detail']['cell']=0
    with pytest.raises(ValueError):m.project(events,audit)


def test_partial_missing_does_not_become_complete():
    out=m.project([],dict(transport_complete=False))
    assert out['status'] is None and out['initial_success'] is None and out['final_success'] is None


def test_ack_before_any_artifact_reads():
    with pytest.raises(ValueError,match='terminal confirmation'):m.build('a'*64,'b'*64,'no')
