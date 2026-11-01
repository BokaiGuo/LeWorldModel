from copy import deepcopy

import pytest
from experiments.spectral_reachability import e3_cell03_feedback_prepare_codex_20261030_v1 as m


def sample():
    prior = dict(schema='fixed', cell=0, resolver={'path': 'resolver', 'sha256': 'x'},
                 control5={'prepare': {'path': 'prepare', 'sha256': 'y'}},
                 additional_sources={'/a': 'a', '/b': 'b'})
    fresh = dict(schema='fixed', cell=3, resolver=deepcopy(prior['resolver']),
                 control5=m.references(prior), additional_sources={'/a': 'a'})
    return prior, fresh


def test_exact_binding_without_mutation():
    prior, fresh = sample()
    result = m.combine(prior, fresh)
    assert result['cell'] == 3 and result['control5'] == fresh['control5']
    assert result['additional_sources'] == prior['additional_sources'] and prior['cell'] == 0


@pytest.mark.parametrize('kind', ['cell', 'schema', 'resolver', 'prepare', 'receipt', 'source', 'extra'])
def test_bad_binding_is_rejected(kind):
    prior, fresh = sample()
    if kind == 'cell': fresh['cell'] = 2
    if kind == 'schema': fresh['schema'] = 'different'
    if kind == 'resolver': fresh['resolver']['sha256'] = 'bad'
    if kind == 'prepare': fresh['control5']['prepare']['sha256'] = 'bad'
    if kind == 'receipt': fresh['control5']['receipt']['sha256'] = 'bad'
    if kind == 'source': fresh['additional_sources']['/a'] = 'bad'
    if kind == 'extra': fresh['additional_sources']['/new'] = 'bad'
    with pytest.raises(ValueError): m.combine(prior, fresh)


def test_exclusive_publication_does_not_overwrite(tmp_path):
    path = tmp_path / 'authority.json'
    digest = m.startup.write_new(str(path), {'cell': 3})
    original = path.read_bytes()
    assert len(digest) == 64
    with pytest.raises((ValueError, FileExistsError)):
        m.startup.write_new(str(path), {'cell': 2})
    assert path.read_bytes() == original
