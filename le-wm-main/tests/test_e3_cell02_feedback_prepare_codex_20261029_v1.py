from copy import deepcopy
import pytest
from experiments.spectral_reachability import e3_cell02_feedback_prepare_codex_20261029_v1 as m


def sample():
    base = dict(schema='fixed', cell=0, resolver={'path':'resolver','sha256':'x'},
                control5={'prepare':{'path':'prepare','sha256':'y'}}, additional_sources={'/a':'a','/b':'b'})
    fresh = dict(schema='fixed', cell=2, resolver=deepcopy(base['resolver']),
                 control5=m.references(base), additional_sources={'/a':'a'})
    return base, fresh


def test_exact_binding_and_no_mutation():
    base, fresh = sample()
    actual = m.combine(base, fresh)
    assert actual['cell'] == 2 and actual['control5'] == fresh['control5']
    assert actual['additional_sources'] == base['additional_sources'] and base['cell'] == 0


@pytest.mark.parametrize('kind', ['cell', 'schema', 'resolver', 'prepare', 'receipt', 'source', 'extra'])
def test_reject_wrong_bindings(kind):
    base, fresh = sample()
    if kind == 'cell': fresh['cell'] = 3
    if kind == 'schema': fresh['schema'] = 'different'
    if kind == 'resolver': fresh['resolver']['sha256'] = 'bad'
    if kind == 'prepare': fresh['control5']['prepare']['sha256'] = 'bad'
    if kind == 'receipt': fresh['control5']['receipt']['sha256'] = 'bad'
    if kind == 'source': fresh['additional_sources']['/a'] = 'bad'
    if kind == 'extra': fresh['additional_sources']['/new'] = 'bad'
    with pytest.raises(ValueError): m.combine(base, fresh)
