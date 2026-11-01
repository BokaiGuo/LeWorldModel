"""Saved-output wrapper mocks, no model or simulator."""
import json
import pytest
from experiments.spectral_reachability import consume_observed_CEM_MAIN_20261028_v1 as m


def test_persists_exact_verified_return_without_changing_schema(tmp_path,monkeypatch):
    result=dict(status='observed_CEM_commands_verified_not_admitted',plans=[dict(model_Q=0.)])
    monkeypatch.setattr(m.source,'consume',lambda directory,sha:result)
    output=tmp_path/'accepted.json';r=m.run('/RAM','a'*64,output)
    assert json.loads(output.read_bytes())==result
    assert r['plans']==1 and r['model_Q']==[0.]
    assert r['new_neural_calls']==r['new_physics_calls']==0


def test_consumer_failure_does_not_create_release_file(tmp_path,monkeypatch):
    def bad(*a):raise ValueError('actual module bridge')
    monkeypatch.setattr(m.source,'consume',bad)
    output=tmp_path/'accepted.json'
    with pytest.raises(ValueError,match='actual module bridge'):m.run('/RAM','a'*64,output)
    assert not output.exists()


def test_never_overwrite_or_reconsume_existing_release(tmp_path,monkeypatch):
    output=tmp_path/'accepted.json';output.write_text('sealed')
    monkeypatch.setattr(m.source,'consume',lambda *a:pytest.fail('read after overwrite guard'))
    with pytest.raises(ValueError,match='exclusive'):m.run('/RAM','a'*64,output)
    assert output.read_text()=='sealed'
