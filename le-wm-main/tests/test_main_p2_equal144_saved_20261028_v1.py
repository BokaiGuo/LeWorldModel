import pytest
from experiments.spectral_reachability import main_p2_equal144_saved_20261028_v1 as m


def plans():
    return [dict(source_slot=s,arm=a,task_id=str(s),model_terminal_cost=float(3-i),commands=[[i,s]])
            for s in (0,1) for i,a in enumerate(m.native.ARMS)]


def test_descriptive_only(monkeypatch):
    monkeypatch.setattr(m.native,'consume',lambda *args:dict(plans=plans()))
    report=m.analyze()
    assert len(report['rows']) == 2
    assert all(r['spectral_minus_beam_model_cost'] == -1 for r in report['rows'])
    assert all(r['physical_comparison'] is None for r in report['rows'])
    assert report['scientific_admission'] is False and report['task_benefit'] is None


def test_missing_arm_rejected(monkeypatch):
    monkeypatch.setattr(m.native,'consume',lambda *args:dict(plans=plans()[:-1]))
    with pytest.raises(ValueError):
        m.analyze()


def test_strict_failure_not_swallowed(monkeypatch):
    def fail(*args):
        raise ValueError('strict failure')
    monkeypatch.setattr(m.native,'consume',fail)
    with pytest.raises(ValueError,match='strict failure'):
        m.analyze()
