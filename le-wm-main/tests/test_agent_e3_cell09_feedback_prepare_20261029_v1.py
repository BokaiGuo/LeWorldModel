from copy import deepcopy

import pytest

from experiments.spectral_reachability import agent_e3_cell09_feedback_prepare_20261029_v1 as m
from experiments.spectral_reachability import agent_e3_frozen_model_task_resolver_20261029_v1 as resolver


def test_fixed_roster():
    slot = resolver.roster()[9]
    assert (slot['cell_id'], slot['root'], slot['goal_index'], slot['horizon_packets'], slot['domain']) == (m.TASK, 1, 0, 1, 'Dx')


@pytest.mark.parametrize('mismatch', [None, 'cell', 'resolver', 'prepare', 'report', 'receipt', 'source', 'new_source'])
def test_build_rejects_mismatched_inputs(monkeypatch, mismatch):
    prior = dict(schema='fixed', cell=0, resolver={'sha256': 'r'},
                 control5={'prepare': {'sha256': 'p'}}, additional_sources={'/a': 'a'})
    monkeypatch.setattr(m.host, 'authority', lambda *args: None)
    monkeypatch.setattr(m.host.launch.host, 'read', lambda *args: deepcopy(prior))
    monkeypatch.setattr(m.host.launch.host, 'sha', lambda *args: m.BASE_SHA)

    def fresh(refs, resolver_ref, cell):
        value = dict(schema='fixed', cell=cell, resolver=deepcopy(resolver_ref),
                     control5=deepcopy(refs), additional_sources={'/a': 'a'})
        if mismatch == 'cell': value['cell'] = 8
        elif mismatch == 'resolver': value['resolver']['sha256'] = 'bad'
        elif mismatch == 'prepare': value['control5']['prepare']['sha256'] = 'bad'
        elif mismatch in ('report', 'receipt'): value['control5'][mismatch]['sha256'] = 'bad'
        elif mismatch == 'source': value['additional_sources']['/a'] = 'bad'
        elif mismatch == 'new_source': value['additional_sources']['/new'] = 'bad'
        return value

    monkeypatch.setattr(m.identity, 'build', fresh)
    if mismatch is None:
        assert m.build()['cell'] == 9
    else:
        with pytest.raises(Exception):
            m.build()
