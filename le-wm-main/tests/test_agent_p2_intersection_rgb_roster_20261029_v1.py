from copy import deepcopy
import json
from pathlib import Path

import pytest
from experiments.spectral_reachability import agent_p2_intersection_rgb_roster_20261029_v1 as m

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'results'


@pytest.fixture(scope='module')
def actual():
    p = json.loads((RESULTS/'agent_p2_intersection_physical_20261029_v2_prepare_MAIN01.json').read_bytes())
    r = json.loads((RESULTS/'agent_p2_intersection_physical_20261029_v2_consume_MAIN01.json').read_bytes())
    return p, r, RESULTS/'agent_p2_intersection_physical_20261029_v2_MAIN01'


def test_actual_roster_fixed_and_missing(actual):
    rows = m.assemble(*actual)
    assert len(rows) == 24
    assert sum(r['frame'] is None for r in rows) == 1
    assert rows[-1]['slot'] == 5 and rows[-1]['checkpoint'] == 15 and rows[-1]['frame'] is None
    assert all(r['prediction_role'] == 'observed_root_context_not_forecast' for r in rows[::4])


@pytest.mark.parametrize('field', ['raw_float32_bits','task_id','selection_digest'])
def test_mutated_plan_rejected(actual, field):
    p,r,d = deepcopy(actual)
    p['plans'][0][field] = 'wrong'
    with pytest.raises(ValueError): m.assemble(p,r,d)


def test_missing_cannot_be_padded(actual):
    p,r,d = deepcopy(actual)
    r['rows'][5]['missing_checkpoints'] = []
    with pytest.raises(ValueError): m.assemble(p,r,d)


def test_native_state_identity(actual):
    p,r,d = deepcopy(actual)
    r['rows'][0]['checkpoints'][1]['state'][0] += 1
    with pytest.raises(ValueError): m.assemble(p,r,d)
