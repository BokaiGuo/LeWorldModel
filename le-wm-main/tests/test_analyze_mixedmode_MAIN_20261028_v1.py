"""Saved-array arithmetic fixtures only, NOT mixedmode experimental results."""
from copy import deepcopy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability.analyze_mixedmode_MAIN_20261028_v1 import summarize


@pytest.fixture(scope='module')
def saved_fixture():
    root = Path(__file__).resolve().parents[1]
    row = json.loads(gzip.decompress((root / 'results/agent_allmode_prefix_20261028_v2_cuda_MAIN01/root_0.json.gz').read_bytes()))
    prepared = json.loads((root / 'results/agent_allmode_prefix_20261028_v2_prepare_MAIN01.json').read_bytes())
    # Deliberately repeat known candidates for an analysis-unit fixture, not
    # a new compiler run or claim that mixed candidates actually repeat them.
    expanded = dict(row, result=dict(row['result']))
    expanded['result']['candidates'] = row['result']['candidates'] + [
        dict(row['result']['candidates'][i % 42], index=42+i) for i in range(180)]
    return expanded, row, prepared['slots'][0]


def call(fixture):
    expanded, original, slot = fixture
    return summarize(dict(rows=[expanded]*4), dict(rows=[original]*4), dict(slots=[slot]*4))


def test_saved_arithmetic_fixture_preserves_original42(saved_fixture):
    rows = call(saved_fixture)
    assert len(rows) == 4
    assert all(r['original42_exact'] and r['selected_raw_unchanged'] for r in rows)
    assert np.isfinite([r['selected_terminal_Q'] for r in rows]).all()


@pytest.mark.parametrize('fault', ['original42', 'root_J', 'stage', 'reach', 'terminal', 'objective', 'count', 'order'])
def test_arithmetic_or_identity_damage_rejected(saved_fixture, fault):
    expanded, original, slot = saved_fixture
    changed = dict(expanded, result=dict(expanded['result']))
    changed['result']['candidates'] = list(changed['result']['candidates'])
    index = 0 if fault == 'original42' else 190
    changed['result']['candidates'][index] = deepcopy(changed['result']['candidates'][index])
    candidate = changed['result']['candidates'][index]
    if fault == 'root_J':
        changed['result']['root_jacobian'] = np.zeros((192, 10)).tolist()
    elif fault == 'count':
        changed['result']['candidates'].pop()
    elif fault == 'order':
        candidate['index'] = 191
    else:
        key = {'original42': 'stage_cost', 'stage': 'stage_cost', 'reach': 'own_reach_value',
               'terminal': 'nonlinear_terminal_cost', 'objective': 'objective'}[fault]
        candidate[key] += .1
    with pytest.raises(ValueError):
        call((changed, original, slot))
